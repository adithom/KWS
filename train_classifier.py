import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.onnx
import random

from torch.autograd import Variable, Function

from preprocessing import create_dataloaders
from training_config import TrainingConfig
from model import *

class KeywordSpotter(nn.Module):
    """ This baseclass provides the PyTorch Module pattern for defining and training keyword spotters """

    def __init__(self):
        """
        Initialize the KeywordSpotter with the following parameters:
        input_dim - the size of the input audio frame in # samples
        num_keywords - the number of predictions to come out of the model.
        """
        super(KeywordSpotter, self).__init__()

        self.training = False
        self.tracking = False

        self.init_hidden()

    def name(self):
        return "KeywordSpotter"

    def init_hidden(self):
        """ Clear any  hidden state """
        pass

    def forward(self, input):
        """ Perform the forward processing of the given input and return the prediction """
        raise Exception("need to implement the forward method")

    def export(self, name, device):
        """ Export the model to the ONNX file format """
        self.init_hidden()
        self.tracking = True
        dummy_input = Variable(torch.randn(1, 1, self.input_dim))
        if device:
            dummy_input = dummy_input.to(device)
        torch.onnx.export(self, dummy_input, name, verbose=True)
        self.tracking = False

    def batch_accuracy(self, scores, labels):
        """ Compute the training accuracy of the results of a single mini-batch """
        batch_size = scores.shape[0]
        passed = 0
        results = []
        for i in range(batch_size):
            expected = labels[i]
            actual = scores[i].argmax()
            results += [int(actual)]
            if expected == actual:
                passed += 1
        return (float(passed) * 100.0 / float(batch_size), passed, results)

    def configure_optimizer(self, options):
        initial_rate = options.learning_rate
        oo = options.optimizer_options

        if options.optimizer == "Adadelta":
            optimizer = optim.Adadelta(self.parameters(), lr=initial_rate, weight_decay=oo.weight_decay,
                                       rho=oo.rho, eps=oo.eps)
        elif options.optimizer == "Adagrad":
            optimizer = optim.Adagrad(self.parameters(), lr=initial_rate, weight_decay=oo.weight_decay,
                                      lr_decay=oo.lr_decay)
        elif options.optimizer == "Adam":
            optimizer = optim.Adam(self.parameters(), lr=initial_rate, weight_decay=oo.weight_decay,
                                   betas=oo.betas, eps=oo.eps)
        elif options.optimizer == "Adamax":
            optimizer = optim.Adamax(self.parameters(), lr=initial_rate, weight_decay=oo.weight_decay,
                                     betas=oo.betas, eps=oo.eps)
        elif options.optimizer == "ASGD":
            optimizer = optim.ASGD(self.parameters(), lr=initial_rate, weight_decay=oo.weight_decay,
                                   lambd=oo.lambd, alpha=oo.alpha, t0=oo.t0)
        elif options.optimizer == "RMSprop":
            optimizer = optim.RMSprop(self.parameters(), lr=initial_rate, weight_decay=oo.weight_decay,
                                      eps=oo.eps, alpha=oo.alpha, momentum=oo.momentum, centered=oo.centered)
        elif options.optimizer == "Rprop":
            optimizer = optim.Rprop(self.parameters(), lr=initial_rate, etas=oo.etas,
                                    step_sizes=oo.step_sizes)
        elif options.optimizer == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=initial_rate, weight_decay=oo.weight_decay,
                                  momentum=oo.momentum, dampening=oo.dampening, nesterov=oo.nesterov)
        return optimizer

    def configure_lr(self, options, optimizer, ticks, total_iterations):
        num_epochs = options.max_epochs
        learning_rate = options.learning_rate
        lr_scheduler = options.lr_scheduler
        lr_min = options.lr_min
        lr_peaks = options.lr_peaks
        gamma = options.lr_gamma
        if not lr_min:
            lr_min = learning_rate
        scheduler = None
        if lr_scheduler == "TriangleLR":
            steps = lr_peaks * 2 + 1
            stepsize = num_epochs / steps
            scheduler = TriangularLR(optimizer, stepsize * ticks, lr_min, learning_rate, gamma)
        elif lr_scheduler == "CosineAnnealingLR":
            # divide by odd number to finish on the minimum learning rate
            cycles = lr_peaks * 2 + 1
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iterations / cycles,
                                                             eta_min=lr_min)
        elif lr_scheduler == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        elif lr_scheduler == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=options.lr_step_size, gamma=gamma)
        elif lr_scheduler == "ExponentialResettingLR":
            reset = (num_epochs * ticks) / 3  # reset at the 1/3 mark.
            scheduler = ExponentialResettingLR(optimizer, gamma, reset)
        return scheduler

    def fit(self, training_data, validation_data, options, sparsify=False, device=None, detail=False, run=None):
        """
        Perform the training.  This is not called "train" because
        the base class already defines that method with a different meaning.
        The base class "train" method puts the Module into "training mode".
        """
        print("Training {} using {} rows of featurized training input...".format(self.name(), training_data.num_rows))

        if training_data.mean is not None:
            mean = torch.from_numpy(np.array([[training_data.mean]])).to(device)
            std = torch.from_numpy(np.array([[training_data.std]])).to(device)
        else:
            mean = None
            std = None

        self.normalize(mean, std)

        self.training = True
        start = time.time()
        loss_function = nn.NLLLoss()
        optimizer = self.configure_optimizer(options)
        print(optimizer)

        num_epochs = options.max_epochs
        batch_size = options.batch_size
        trim_level = options.trim_level
        
        ticks = training_data.num_rows / batch_size  # iterations per epoch
        
        # Calculation of total iterations in non-rolling vs rolling training
        # ticks = num_rows/batch_size (total number of iterations per epoch)
        # Non-Rolling Training:
        # Total Iteration = num_epochs * ticks
        # Rolling Training:
        # irl = Initial_rolling_length (We are using 2)
        # If num_epochs <=  max_rolling_length:
        # Total Iterations = sum(range(irl, irl + num_epochs))
        # If num_epochs > max_rolling_length:
        # Total Iterations = sum(range(irl, irl + max_rolling_length)) + (num_epochs - max_rolling_length)*ticks
        if options.rolling:
            rolling_length = 2
            max_rolling_length = int(ticks)
            if max_rolling_length > options.max_rolling_length + rolling_length:
                max_rolling_length = options.max_rolling_length + rolling_length
            bag_count = 100
            hidden_bag_size = batch_size * bag_count
            if num_epochs + rolling_length < max_rolling_length:
                max_rolling_length = num_epochs + rolling_length
            total_iterations = sum(range(rolling_length, max_rolling_length))
            if num_epochs + rolling_length > max_rolling_length:
                epochs_remaining = num_epochs + rolling_length - max_rolling_length
                total_iterations += epochs_remaining * training_data.num_rows / batch_size
            ticks = total_iterations / num_epochs
        else:
            total_iterations = ticks * num_epochs

        scheduler = self.configure_lr(options, optimizer, ticks, total_iterations)

        # optimizer = optim.Adam(model.parameters(), lr=0.0001)
        log = []

        for epoch in range(num_epochs):
            self.train()
            if options.rolling:
                rolling_length += 1
                if rolling_length <= max_rolling_length:
                    self.init_hidden_bag(hidden_bag_size, device)
            for i_batch, (audio, labels) in enumerate(training_data.get_data_loader(batch_size)):
                if not self.batch_first:
                    audio = audio.transpose(1, 0)  # GRU wants seq,batch,feature

                if device:
                    self.move_to(device)
                    audio = audio.to(device)
                    labels = labels.to(device)

                # Also, we need to clear out the hidden state,
                # detaching it from its history on the last instance.
                if options.rolling:
                    if rolling_length <= max_rolling_length:
                        if (i_batch + 1) % rolling_length == 0:
                            self.init_hidden()
                            break

                    self.rolling_step()
                else:
                    self.init_hidden()

                self.to(device) # sparsify routines might move param matrices to cpu    

                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                optimizer.zero_grad()

                # Run our forward pass.
                keyword_scores = self(audio)

                # Compute the loss, gradients
                loss = loss_function(keyword_scores, labels)

                # Backward pass: compute gradient of the loss with respect to all the learnable
                # parameters of the model. Internally, the parameters of each Module are stored
                # in Tensors with requires_grad=True, so this call will compute gradients for
                # all learnable parameters in the model.
                loss.backward()
                # move to next learning rate
                if scheduler:
                    scheduler.step()

                # Calling the step function on an Optimizer makes an update to its parameters
                # applying the gradients we computed during back propagation
                optimizer.step()

                if sparsify:
                    if epoch >= num_epochs/3:
                        if epoch < (2*num_epochs)/3:
                            if i_batch % trim_level == 0:
                                self.sparsify()
                            else:
                                self.sparsifyWithSupport()
                        else:
                            self.sparsifyWithSupport()
                    self.to(device) # sparsify routines might move param matrices to cpu

                learning_rate = optimizer.param_groups[0]['lr']
                if detail:
                    learning_rate = optimizer.param_groups[0]['lr']
                    log += [{'iteration': iteration, 'loss': loss.item(), 'learning_rate': learning_rate}]
            # Find the best prediction in each sequence and return it's accuracy
            passed, total, rate = self.evaluate(validation_data, batch_size, device)
            learning_rate = optimizer.param_groups[0]['lr']
            current_loss = float(loss.item())
            print("Epoch {}, Loss {:.3f}, Validation Accuracy {:.3f}, Learning Rate {}".format(
                  epoch, current_loss, rate * 100, learning_rate))
            log += [{'epoch': epoch, 'loss': current_loss, 'accuracy': rate, 'learning_rate': learning_rate}]
            if run is not None:
                run.log('progress', epoch / num_epochs)
                run.log('epoch', epoch)
                run.log('accuracy', rate)
                run.log('loss', current_loss)
                run.log('learning_rate', learning_rate)

        end = time.time()
        self.training = False
        print("Trained in {:.2f} seconds".format(end - start))
        print("Model size {}".format(self.get_model_size()))
        return log

    def evaluate(self, test_data, batch_size, device=None, outfile=None):
        """
        Evaluate the given test data and print the pass rate
        """
        self.eval()
        passed = 0
        total = 0

        self.zero_grad()
        results = []
        with torch.no_grad():
            for i_batch, (audio, labels) in enumerate(test_data.get_data_loader(batch_size)):
                batch_size = audio.shape[0]
                audio = audio.transpose(1, 0)  # GRU wants seq,batch,feature
                if device:
                    audio = audio.to(device)
                    labels = labels.to(device)
                total += batch_size
                self.init_hidden()
                keyword_scores = self(audio)
                last_accuracy, ok, actual = self.batch_accuracy(keyword_scores, labels)
                results += actual
                passed += ok

        if outfile:
            print("Saving evaluation results in '{}'".format(outfile))
            with open(outfile, "w") as f:
                json.dump(results, f)

        return (passed, total, passed / total)

def create_model(model_config, input_size, num_keywords):
    ModelClass = get_model_class(KeywordSpotter)
    hidden_units_list = [model_config.hidden_units1, model_config.hidden_units2, model_config.hidden_units3]
    wRank_list = [model_config.wRank1, model_config.wRank2, model_config.wRank3]
    uRank_list = [model_config.uRank1, model_config.uRank2, model_config.uRank3]
    wSparsity_list = [model_config.wSparsity, model_config.wSparsity, model_config.wSparsity]
    uSparsity_list = [model_config.uSparsity, model_config.uSparsity, model_config.uSparsity]
    print(model_config.gate_nonlinearity, model_config.update_nonlinearity)
    return ModelClass(model_config.architecture, input_size, model_config.num_layers,
                      hidden_units_list, wRank_list, uRank_list, wSparsity_list,
                      uSparsity_list, model_config.gate_nonlinearity, 
                      model_config.update_nonlinearity, num_keywords)

def save_json(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f, indent=2)

def train(config, evaluate_only=False, outdir=".", detail=False):
    """Modified training function to use the normalized dataloaders."""
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # Set up device
    device = torch.device("cuda" if config.training.use_gpu and torch.cuda.is_available() else "cpu")
    
    # Create dataloaders with normalization
    train_loader, val_loader, test_loader = create_dataloaders(
        config.dataset.path,
        batch_size=config.training.batch_size
    )

    # Save normalization parameters
    if not evaluate_only:
        np.save(os.path.join(outdir, "mean.npy"), train_loader.dataset.mean)
        np.save(os.path.join(outdir, "std.npy"), train_loader.dataset.std)

    # Create and initialize model
    input_size = train_loader.dataset.data.shape[2]  # MFCC feature dimension
    num_classes = len(train_loader.dataset.label_encoder.classes_)
    model = create_model(config.model, input_size, num_classes)
    model.to(device)

    if not evaluate_only:
        start_time = time.time()
        log = model.fit(train_loader, val_loader, config.training, 
                       config.model.sparsify, device, detail)
        end_time = time.time()
        
        filename = config.model.filename or f"{config.model.architecture}_KeywordSpotter.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'mean': train_loader.dataset.mean,
            'std': train_loader.dataset.std,
            'label_encoder': train_loader.dataset.label_encoder
        }, os.path.join(outdir, filename))

    # Evaluate model
    accuracy = model.evaluate(test_loader, device)
    print(f"Test accuracy = {accuracy:.2f}%")

    if not evaluate_only:
        model.export(os.path.join(outdir, filename.replace(".pt", ".onnx")), device)
        
        log_data = {
            "final_accuracy": accuracy,
            "training_time": end_time - start_time,
            "log": log,
            "normalization_stats": {
                "mean": train_loader.dataset.mean.tolist(),
                "std": train_loader.dataset.std.tolist()
            }
        }
        with open(os.path.join(outdir, "train_results.json"), "w") as f:
            json.dump(log_data, f, indent=2)

    return accuracy, log


def str2bool(v):
    if v is None:
        return False
    lower = v.lower()
    return lower in ["t", "1", "true", "yes"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser("train a RNN based neural network for keyword spotting")

    # all the training parameters
    parser.add_argument("--epochs", help="Number of epochs to train", type=int)
    parser.add_argument("--trim_level", help="Number of batches before sparse support is updated in IHT", type=int)
    parser.add_argument("--lr_scheduler", help="Type of learning rate scheduler (None, TriangleLR, CosineAnnealingLR,"
                                               " ExponentialLR, ExponentialResettingLR)")
    parser.add_argument("--learning_rate", help="Default learning rate, and maximum for schedulers", type=float)
    parser.add_argument("--lr_min", help="Minimum learning rate for the schedulers", type=float)
    parser.add_argument("--lr_peaks", help="Number of peaks for triangle and cosine schedules", type=float)
    parser.add_argument("--batch_size", "-bs", help="Batch size of training", type=int)
    parser.add_argument("--architecture", help="Specify model architecture (FastGRNN)")
    parser.add_argument("--num_layers", type=int, help="Number of RNN layers (1, 2 or 3)")
    parser.add_argument("--hidden_units", "-hu", type=int, help="Number of hidden units in the FastGRNN layers")
    parser.add_argument("--hidden_units1", "-hu1", type=int, help="Number of hidden units in the FastGRNN 1st layer")
    parser.add_argument("--hidden_units2", "-hu2", type=int, help="Number of hidden units in the FastGRNN 2nd layer")
    parser.add_argument("--hidden_units3", "-hu3", type=int, help="Number of hidden units in the FastGRNN 3rd layer")
    parser.add_argument("--use_gpu", help="Whether to use fastGRNN for training", action="store_true")
    parser.add_argument("--normalize", help="Whether to normalize audio dataset", action="store_true")
    parser.add_argument("--rolling", help="Whether to train model in rolling fashion or not", action="store_true")
    parser.add_argument("--max_rolling_length", help="Max number of epochs you want to roll the rolling training"
                        " default is 100", type=int)
    parser.add_argument("--sample_non_kw", "-sl", type=str, help="Sample data for this label with probability sample_prob")
    parser.add_argument("--sample_non_kw_probability", "-spr", type=float, help="Sample from scl with this probability")

    # arguments for fastgrnn
    parser.add_argument("--wRank", "-wr", help="Rank of W in 1st layer of FastGRNN default is None", type=int)
    parser.add_argument("--uRank", "-ur", help="Rank of U in 1st layer of FastGRNN default is None", type=int)
    parser.add_argument("--wRank1", "-wr1", help="Rank of W in 1st layer of FastGRNN default is None", type=int)
    parser.add_argument("--uRank1", "-ur1", help="Rank of U in 1st layer of FastGRNN default is None", type=int)
    parser.add_argument("--wRank2", "-wr2", help="Rank of W in 2nd layer of FastGRNN default is None", type=int)
    parser.add_argument("--uRank2", "-ur2", help="Rank of U in 2nd layer of FastGRNN default is None", type=int)
    parser.add_argument("--wRank3", "-wr3", help="Rank of W in 3rd layer of FastGRNN default is None", type=int)
    parser.add_argument("--uRank3", "-ur3", help="Rank of U in 3rd layer of FastGRNN default is None", type=int)
    parser.add_argument("--wSparsity", "-wsp", help="Sparsity of W matrices", type=float)
    parser.add_argument("--uSparsity", "-usp", help="Sparsity of U matrices", type=float)
    parser.add_argument("--gate_nonlinearity", "-gnl", help="Gate Non-Linearity in FastGRNN default is sigmoid"
                        " use between [sigmoid, quantSigmoid, tanh, quantTanh]")
    parser.add_argument("--update_nonlinearity", "-unl", help="Update Non-Linearity in FastGRNN default is Tanh"
                        " use between [sigmoid, quantSigmoid, tanh, quantTanh]")

    # or you can just specify an options file.
    parser.add_argument("--config", help="Use json file containing all these options (as per 'training_config.py')")

    # and some additional stuff ...
    parser.add_argument("--azureml", help="Tells script we are running in Azure ML context")
    parser.add_argument("--eval", "-e", help="No training, just evaluate existing model", action='store_true')
    parser.add_argument("--filename", "-o", help="Name of model file to generate")
    parser.add_argument("--categories", "-c", help="Name of file containing keywords")
    parser.add_argument("--dataset", "-a", help="Path to the audio folder containing 'training.npz' file")
    parser.add_argument("--outdir", help="Folder in which to store output file and log files")
    parser.add_argument("--detail", "-d", help="Save loss info for every iteration not just every epoch",
                        action="store_true")
    args = parser.parse_args()

    config = TrainingConfig()
    if args.config:
        config.load(args.config)

    azureml = str2bool(args.azureml)

    # then any user defined options overrides these defaults
    if args.epochs:
        config.training.max_epochs = args.epochs
    if args.trim_level:
        config.training.trim_level = args.trim_level
    else:
        config.training.trim_level = 15
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.lr_min:
        config.training.lr_min = args.lr_min
    if args.lr_peaks:
        config.training.lr_peaks = args.lr_peaks
    if args.lr_scheduler:
        config.training.lr_scheduler = args.lr_scheduler
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.rolling:
        config.training.rolling = args.rolling
    if args.max_rolling_length:
        config.training.max_rolling_length = args.max_rolling_length
    if args.architecture:
        config.model.architecture = args.architecture
    if args.num_layers:
        config.model.num_layers = args.num_layers
    if args.hidden_units:
        config.model.hidden_units = args.hidden_units
    if args.hidden_units1:
        config.model.hidden_units = args.hidden_units
    if args.hidden_units2:
        config.model.hidden_units = args.hidden_units
    if args.hidden_units3:
        config.model.hidden_units = args.hidden_units
    if config.model.num_layers >= 1:
        if config.model.hidden_units1 is None:
            config.model.hidden_units1 = config.model.hidden_units
    if config.model.num_layers >= 2:
        if config.model.hidden_units2 is None:
            config.model.hidden_units2 = config.model.hidden_units1
    if config.model.num_layers == 3:
        if config.model.hidden_units3 is None:
            config.model.hidden_units3 = config.model.hidden_units2
    if args.filename:
        config.model.filename = args.filename
    if args.use_gpu:
        config.training.use_gpu = args.use_gpu
    if args.normalize:
        config.dataset.normalize = args.normalize
    if args.categories:
        config.dataset.categories = args.categories
    if args.dataset:
        config.dataset.path = args.dataset
    if args.sample_non_kw:
        config.dataset.sample_non_kw = args.sample_non_kw
        if args.sample_non_kw_probability is None:
            config.dataset.sample_non_kw_probability = 0.5
        else:
            config.dataset.sample_non_kw_probability = args.sample_non_kw_probability
    else:
        config.dataset.sample_non_kw = None

    if args.wRank:
        config.model.wRank = args.wRank
    if args.uRank:
        config.model.uRank = args.wRank
    if args.wRank1:
        config.model.wRank1 = args.wRank1
    if args.uRank1:
        config.model.uRank1 = args.wRank1
    if config.model.wRank1 is None:
        if config.model.wRank is not None:
            config.model.wRank1 = config.model.wRank
    if config.model.uRank1 is None:
        if config.model.uRank is not None:
            config.model.uRank1 = config.model.uRank
    if args.wRank2:
        config.model.wRank2 = args.wRank2
    if args.uRank2:
        config.model.uRank2 = args.wRank2
    if args.wRank3:
        config.model.wRank3 = args.wRank3
    if args.uRank3:
        config.model.uRank3 = args.wRank3
    if args.wSparsity:
        config.model.wSparsity = args.wSparsity
    else:
        config.model.wSparsity = 1.0
    if args.uSparsity:
        config.model.uSparsity = args.uSparsity
    else:
        config.model.uSparsity = 1.0
    if config.model.uSparsity < 1.0 or config.model.wSparsity < 1.0:
        config.model.sparsify = True
    else:
        config.model.sparsify = False
    if args.gate_nonlinearity:
        config.model.gate_nonlinearity = args.gate_nonlinearity
    if args.update_nonlinearity:
        config.model.update_nonlinearity = args.update_nonlinearity

    if not os.path.isfile("config.json"):
        config.save("config.json")

    train(config, args.eval, args.outdir, args.detail, azureml)