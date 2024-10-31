# fastgrnn_combined.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

# Define the RNN classes (assuming FastGRNN and FastGRNNCUDA are defined here)
class FastGRNN(nn.Module):
    # Define the FastGRNN class here
    pass

class FastGRNNCUDA(nn.Module):
    # Define the FastGRNNCUDA class here
    pass

# Define the RNNClassifierModel class
def get_model_class(inheritance_class=nn.Module):
    class RNNClassifierModel(inheritance_class):
        def __init__(self, rnn_name, input_dim, num_layers, hidden_units_list,
                     wRank_list, uRank_list, wSparsity_list, uSparsity_list,
                     gate_nonlinearity, update_nonlinearity, num_classes=None,
                     linear=True, batch_first=False, apply_softmax=True):
            self.input_dim = input_dim
            self.hidden_units_list = hidden_units_list
            self.num_layers = num_layers
            self.num_classes = num_classes
            self.wRank_list = wRank_list
            self.uRank_list = uRank_list
            self.wSparsity_list = wSparsity_list
            self.uSparsity_list = uSparsity_list
            self.gate_nonlinearity = gate_nonlinearity
            self.update_nonlinearity = update_nonlinearity
            self.linear = linear
            self.batch_first = batch_first
            self.apply_softmax = apply_softmax
            self.rnn_name = rnn_name

            if self.linear:
                if not self.num_classes:
                    raise Exception("num_classes need to be specified if linear is True")

            super(RNNClassifierModel, self).__init__()

            RNN = globals()[rnn_name]
            self.rnn_list = nn.ModuleList([
                RNN(self.input_dim if l==0 else self.hidden_units_list[l-1],
                    self.hidden_units_list[l],
                    gate_nonlinearity=self.gate_nonlinearity,
                    update_nonlinearity=self.update_nonlinearity,
                    wRank=self.wRank_list[l], uRank=self.uRank_list[l],
                    wSparsity=self.wSparsity_list[l],
                    uSparsity=self.uSparsity_list[l],
                    batch_first=self.batch_first)
                for l in range(self.num_layers)])

            if rnn_name == "FastGRNNCUDA":
                RNN_ = FastGRNN
                self.rnn_list_ = nn.ModuleList([
                    RNN_(self.input_dim if l==0 else self.hidden_units_list[l-1],
                         self.hidden_units_list[l],
                         gate_nonlinearity=self.gate_nonlinearity,
                         update_nonlinearity=self.update_nonlinearity,
                         wRank=self.wRank_list[l], uRank=self.uRank_list[l],
                         wSparsity=self.wSparsity_list[l],
                         uSparsity=self.uSparsity_list[l],
                         batch_first=self.batch_first)
                    for l in range(self.num_layers)])

            if self.linear:
                last_output_size = self.hidden_units_list[self.num_layers-1]
                self.hidden2keyword = nn.Linear(last_output_size, num_classes)
            self.init_hidden()

        def sparsify(self):
            for rnn in self.rnn_list:
                if self.rnn_name == "FastGRNNCUDA":
                    rnn.to(torch.device("cpu"))
                    rnn.sparsify()
                    rnn.to(torch.device("cuda"))
                else:
                    rnn.cell.sparsify()

        def sparsifyWithSupport(self):
            for rnn in self.rnn_list:
                if self.rnn_name == "FastGRNNCUDA":
                    rnn.to(torch.device("cpu"))
                    rnn.sparsifyWithSupport()
                    rnn.to(torch.device("cuda"))
                else:
                    rnn.cell.sparsifyWithSupport()

        def get_model_size(self):
            total_size = 4 * self.hidden_units_list[self.num_layers-1] * self.num_classes
            for rnn in self.rnn_list:
                if self.rnn_name == "FastGRNNCUDA":
                    total_size += rnn.get_model_size()
                else:
                    total_size += rnn.cell.get_model_size()
            return total_size

        def normalize(self, mean, std):
            self.mean = mean
            self.std = std

        def name(self):
            return "{} layer FastGRNN".format(self.num_layers)

        def move_to(self, device):
            for rnn in self.rnn_list:
                rnn.to(device)
            if hasattr(self, 'hidden2keyword'):
                self.hidden2keyword.to(device)

        def init_hidden_bag(self, hidden_bag_size, device):
            self.hidden_bag_size = hidden_bag_size
            self.device = device
            self.hidden_bags_list = []

            for l in range(self.num_layers):
                self.hidden_bags_list.append(
                   torch.from_numpy(np.zeros([self.hidden_bag_size, self.hidden_units_list[l]],
                                              dtype=np.float32)).to(self.device))

        def rolling_step(self):
            shuffled_indices = list(range(self.hidden_bag_size))
            np.random.shuffle(shuffled_indices)
            if self.hidden_states[0] is not None:
                batch_size = self.hidden_states[0].shape[0]
                temp_indices = shuffled_indices[:batch_size]
                for l in range(self.num_layers):
                    bag = self.hidden_bags_list[l]
                    bag[temp_indices, :] = self.hidden_states[l]
                    self.hidden_states[l] = bag[0:batch_size, :]

        def init_hidden(self):
            self.hidden_states = []
            for l in range(self.num_layers):
                self.hidden_states.append(None)

        def forward(self, input):
            if self.mean is not None:
                input = (input - self.mean) / self.std

            rnn_in = input
            if self.rnn_name == "FastGRNNCUDA":
                if self.tracking:
                    for l in range(self.num_layers):
                        rnn_ = self.rnn_list_[l]
                        model_output = rnn_(rnn_in, hiddenState=self.hidden_states[l])
                        self.hidden_states[l] = model_output.detach()[-1, :, :]
                        weights = self.rnn_list[l].getVars()
                        weights = [weight.clone() for weight in weights]
                        model_output = onnx_exportable_rnn(rnn_in, weights, rnn_.cell, output=model_output)
                        rnn_in = model_output
                else:
                    for l in range(self.num_layers):
                        rnn = self.rnn_list[l]
                        model_output = rnn(rnn_in, hiddenState=self.hidden_states[l])
                        self.hidden_states[l] = model_output.detach()[-1, :, :]
                        rnn_in = model_output
            else:
                for l in range(self.num_layers):
                    rnn = self.rnn_list[l]
                    if self.hidden_states[l] is not None:
                        self.hidden_states[l] = self.hidden_states[l].clone().unsqueeze(0)
                    model_output = rnn(rnn_in, hiddenState=self.hidden_states[l])
                    self.hidden_states[l] = model_output.detach()[-1, :, :]
                    if self.tracking:
                        weights = rnn.getVars()
                        model_output = onnx_exportable_rnn(rnn_in, weights, rnn.cell, output=model_output)
                    rnn_in = model_output

            if self.linear:
                model_output = self.hidden2keyword(model_output[-1, :, :])
            if self.apply_softmax:
                model_output = F.log_softmax(model_output, dim=1)
            return model_output
    return RNNClassifierModel

# Define the FastTrainer class
class FastTrainer:
    def __init__(self, model, optimizer, loss, accuracy, device, isDenseTraining=True):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.accuracy = accuracy
        self.device = device
        self.isDenseTraining = isDenseTraining
        self.model.to(self.device)

    def saveParams(self, currDir):
        torch.save(self.model.state_dict(), os.path.join(currDir, "model.pth"))

    def train(self, batchSize, totalEpochs, Xtrain, Xtest, Ytrain, Ytest,
              decayStep, decayRate, dataDir, currDir):
        fileName = str(self.model.rnn_name) + 'Results_pytorch.txt'
        resultFile = open(os.path.join(dataDir, fileName), 'a+')
        numIters = int(np.ceil(float(Xtrain.shape[0]) / float(batchSize)))
        totalBatches = numIters * totalEpochs

        counter = 0
        trimlevel = 15
        ihtDone = 0
        maxTestAcc = -10000
        if self.isDenseTraining:
            ihtDone = 1
            maxTestAcc = -10000
        header = '*' * 20
        self.timeSteps = int(Xtest.shape[1] / self.model.input_dim)
        Xtest = Xtest.reshape((-1, self.timeSteps, self.model.input_dim))
        Xtest = np.swapaxes(Xtest, 0, 1)
        Xtrain = Xtrain.reshape((-1, self.timeSteps, self.model.input_dim))
        Xtrain = np.swapaxes(Xtrain, 0, 1)

        for i in range(totalEpochs):
            print("\nEpoch Number: " + str(i), file=sys.stdout)

            if i % decayStep == 0 and i != 0:
                self.learningRate = self.learningRate * decayRate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learningRate

            shuffled = list(range(Xtrain.shape[1]))
            np.random.shuffle(shuffled)
            trainAcc = 0.0
            trainLoss = 0.0
            numIters = int(numIters)
            for j in range(numIters):
                if counter == 0:
                    msg = " Dense Training Phase Started "
                    print("\n%s%s%s\n" % (header, msg, header), file=sys.stdout)

                k = shuffled[j * batchSize:(j + 1) * batchSize]
                batchX = Xtrain[:, k, :]
                batchY = Ytrain[k]

                self.optimizer.zero_grad()
                logits = self.model(batchX.to(self.device))
                batchLoss = self.loss(logits, batchY.to(self.device))
                batchAcc = self.accuracy(logits, batchY.to(self.device))
                batchLoss.backward()
                self.optimizer.step()

                del batchX, batchY

                trainAcc += batchAcc.item()
                trainLoss += batchLoss.item()

                if (counter >= int(totalBatches / 3.0) and
                        (counter < int(2 * totalBatches / 3.0)) and
                        counter % trimlevel == 0 and
                        not self.isDenseTraining):
                    self.runHardThrsd()
                    if ihtDone == 0:
                        msg = " IHT Phase Started "
                        print("\n%s%s%s\n" % (header, msg, header), file=sys.stdout)
                    ihtDone = 1
                elif ((ihtDone == 1 and counter >= int(totalBatches / 3.0) and
                       (counter < int(2 * totalBatches / 3.0)) and
                       counter % trimlevel != 0 and
                       not self.isDenseTraining) or
                        (counter >= int(2 * totalBatches / 3.0) and
                            not self.isDenseTraining)):
                    self.runSparseTraining()
                    if counter == int(2 * totalBatches / 3.0):
                        msg = " Sparse Retraining Phase Started "
                        print("\n%s%s%s\n" % (header, msg, header), file=sys.stdout)
                counter += 1

            trainLoss /= numIters
            trainAcc /= numIters
            print("Train Loss: " + str(trainLoss) +
                  " Train Accuracy: " + str(trainAcc),
                  file=sys.stdout)

            logits = self.model(Xtest.to(self.device))
            testLoss = self.loss(logits, Ytest.to(self.device)).item()
            testAcc = self.accuracy(logits, Ytest.to(self.device)).item()

            if ihtDone == 0:
                maxTestAcc = -10000
                maxTestAccEpoch = i
            else:
                if maxTestAcc <= testAcc:
                    maxTestAccEpoch = i
                    maxTestAcc = testAcc
                    self.saveParams(currDir)

            print("Test Loss: " + str(testLoss) +
                  " Test Accuracy: " + str(testAcc), file=sys.stdout)
            sys.stdout.flush()

        print("\nMaximum Test accuracy at compressed" +
              " model size(including early stopping): " +
              str(maxTestAcc) + " at Epoch: " +
              str(maxTestAccEpoch + 1) + "\nFinal Test" +
              " Accuracy: " + str(testAcc), file=sys.stdout)
        print("\n\nNon-Zeros: " + str(self.getModelSize()[0]) +
              " Model Size: " + str(float(self.getModelSize()[1]) / 1024.0) +
              " KB hasSparse: " + str(self.getModelSize()[2]) + "\n",
              file=sys.stdout)

        resultFile.write("MaxTestAcc: " + str(maxTestAcc) +
                         " at Epoch(totalEpochs): " +
                         str(maxTestAccEpoch + 1) +
                         "(" + str(totalEpochs) + ")" + " ModelSize: " +
                         str(float(self.getModelSize()[1]) / 1024.0) +
                         " KB hasSparse: " + str(self.getModelSize()[2]) +
                         " Param Directory: " +
                         str(os.path.abspath(currDir)) + "\n")

        print("The Model Directory: " + currDir + "\n")

        resultFile.close()
        sys.stdout.flush()
        if sys.stdout is not sys.stdout:
            sys.stdout.close()

# Main script to create and train the FastGRNN model
if __name__ == "__main__":
    # Define model parameters
    rnn_name = "FastGRNN"
    input_dim = 10
    num_layers = 2
    hidden_units_list = [20, 20]
    wRank_list = [10, 10]
    uRank_list = [10, 10]
    wSparsity_list = [0.5, 0.5]
    uSparsity_list = [0.5, 0.5]
    gate_nonlinearity = torch.sigmoid
    update_nonlinearity = torch.tanh
    num_classes = 5
    linear = True
    batch_first = False
    apply_softmax = True

    # Create the model
    RNNClassifierModel = get_model_class()
    model = RNNClassifierModel(rnn_name, input_dim, num_layers, hidden_units_list,
                               wRank_list, uRank_list, wSparsity_list, uSparsity_list,
                               gate_nonlinearity, update_nonlinearity, num_classes,
                               linear, batch_first, apply_softmax)

    # Define training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()
    accuracy = lambda logits, labels: (logits.argmax(dim=1) == labels).float().mean()
    batchSize = 32
    totalEpochs = 10
    decayStep = 5
    decayRate = 0.1
    dataDir = "./data"
    currDir = "./model"

    # Generate dummy data for training and testing
    Xtrain = torch.randn(100, input_dim)
    Ytrain = torch.randint(0, num_classes, (100,))
    Xtest = torch.randn(20, input_dim)
    Ytest = torch.randint(0, num_classes, (20,))

    # Create the trainer and train the model
    trainer = FastTrainer(model, optimizer, loss, accuracy, device)
    trainer.train(batchSize, totalEpochs, Xtrain, Xtest, Ytrain, Ytest, decayStep, decayRate, dataDir, currDir)