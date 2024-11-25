import argparse
import os
import wave
from array import array
from collections import Counter
from queue import Queue
from sys import byteorder
from threading import Thread
import json

import numpy as np
import pyaudio
import torch
from torch.nn.functional import softmax
from mfccProcessor import MFCCProcessor
from trainingConfig import TrainingConfig
from model import get_model_class

# Audio Recording Parameters
FORMAT = pyaudio.paInt16
RATE = 16000
stride = int(50 * (RATE / 1000))
CHUNK_SIZE = stride

# Model and Feature Extraction Parameters
NUM_FILTERS = 32
MAXLEN = 16000
WINLEN = 0.025
WINSTEP = 0.010

NUM_WINDOWS = 10
MAJORITY = 5
QUEUE = Queue(maxsize=100000)


class RecordingThread(Thread):
    """
    Thread to capture audio data from the microphone.
    """

    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=1, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK_SIZE)

        global QUEUE
        while True:
            snd_data = array('h', stream.read(CHUNK_SIZE, exception_on_overflow=False))
            if byteorder == 'big':
                snd_data.byteswap()
            QUEUE.put(snd_data)

        stream.stop_stream()
        stream.close()
        p.terminate()


class PredictionThread(Thread):
    """
    Thread to process audio data and make predictions.
    """

    def __init__(self, model, mean, std, device, class_labels):
        super(PredictionThread, self).__init__()
        self.model = model
        self.mean = mean
        self.std = std
        self.device = device
        self.class_labels = class_labels
        self.mfcc_processor = MFCCProcessor(
            sample_rate=RATE, n_mfcc=NUM_FILTERS, winlen=WINLEN, winstep=WINSTEP, feature_type='delta'
        )
        self.buffer = array('h')
        self.votes = []
        self.previous_prediction = None

    def run(self):
        global QUEUE
        while True:
            data = QUEUE.get()
            QUEUE.task_done()
            self.buffer.extend(data)

            if len(self.buffer) < MAXLEN:
                continue

            self.buffer = self.buffer[CHUNK_SIZE:]

            # Extract features
            features = self.mfcc_processor.compute_features(np.array(self.buffer, dtype=np.float32))
            features = (features - self.mean) / self.std
            features = torch.FloatTensor(features).unsqueeze(0).to(self.device)

            # Perform inference
            with torch.no_grad():  # Add no_grad for inference
                logits = self.model(features)
                probabilities = softmax(logits, dim=1)
                predicted_index = torch.argmax(probabilities, dim=1).item()
                predicted_word = self.class_labels[predicted_index]

            # Majority voting
            if len(self.votes) == NUM_WINDOWS:
                self.votes.pop(0)
            self.votes.append(predicted_word)

            # Check for majority prediction
            if len(self.votes) >= MAJORITY:
                majority_word, frequency = Counter(self.votes).most_common(1)[0]
                if majority_word != self.previous_prediction and frequency >= MAJORITY:
                    print(f"Detected keyword: {majority_word}")
                    self.previous_prediction = majority_word


def initialize_batchnorm_model(config, input_dim, num_classes):
    """
    Initialize the RNNClassifierModel with BatchNorm for inference.
    """
    rnn_name = "FastGRNNBatchNorm"
    ModelClass = get_model_class()

    hidden_units_list = [config['hidden_units1'], config['hidden_units2'], config['hidden_units3']]
    wRank_list = [config['wRank1'], config['wRank2'], config['wRank3']]
    uRank_list = [config['uRank1'], config['uRank2'], config['uRank3']]
    wSparsity_list = [config['wSparsity']] * len(hidden_units_list)
    uSparsity_list = [config['uSparsity']] * len(hidden_units_list)

    model = ModelClass(
        rnn_name=rnn_name,
        input_dim=input_dim,
        num_layers=config['num_layers'],
        hidden_units_list=hidden_units_list,
        wRank_list=wRank_list,
        uRank_list=uRank_list,
        wSparsity_list=wSparsity_list,
        uSparsity_list=uSparsity_list,
        gate_nonlinearity=config['gate_nonlinearity'],
        update_nonlinearity=config['update_nonlinearity'],
        num_classes=num_classes,
        linear=True,
        batch_first=True,
        apply_softmax=True
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keyword Spotting with BatchNorm")
    parser.add_argument("--config_path", help="Path to config file", type=str, required=True)
    parser.add_argument("--model_path", help="Path to trained model", type=str, required=True)
    parser.add_argument("--mean_path", help="Path to train dataset mean", type=str, required=True)
    parser.add_argument("--std_path", help="Path to train dataset std", type=str, required=True)

    args = parser.parse_args()

    # Load config
    config = TrainingConfig()
    config.load(args.config_path)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint and model info
    checkpoint = torch.load(args.model_path, map_location=device)
    mean = checkpoint['mean']
    std = checkpoint['std']
    label_encoder = checkpoint['label_encoder']
    class_labels = label_encoder.classes_.tolist()

    # Initialize and load model
    model = initialize_batchnorm_model(config, NUM_FILTERS, len(class_labels))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Start threads for real-time audio processing
    pred_thread = PredictionThread(model, mean, std, device, class_labels)
    rec_thread = RecordingThread()

    try:
        pred_thread.start()
        rec_thread.start()

        pred_thread.join()
        rec_thread.join()
    except KeyboardInterrupt:
        print("\nStopping keyword detection...")
        exit(0)