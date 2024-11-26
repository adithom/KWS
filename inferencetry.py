#python.\inferencetry.py - -config_path.\config.json - -model_path.\training\FastGRNNBatchNorm_KeywordSpotter.pt - -mean_path.\training\mean.npy - -std_path.\training\std.npy

import argparse
from array import array
from collections import Counter
from queue import Queue, Empty
from sys import byteorder
from threading import Thread

import numpy as np
import pyaudio
import torch
from torch.nn.functional import softmax
#from data_pipeline.bogusmfcc import MFCCProcessor
import sys
import os
import threading
import traceback

stop_event = threading.Event()


# Add the root directory to sys.path
root_path = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
sys.path.append(root_path)  # Add root directory to sys.path
sys.path.append(os.path.join(root_path, ".."))  # Include parent directory if trainingConfig is there

from trainingConfig import TrainingConfig
from model import get_model_class

import librosa
import numpy as np


class MFCCProcessor:
    def __init__(self, n_mfcc=32, width=9, sample_rate=16000, winlen=0.025, winstep=0.010, feature_type='delta'):
        """
        Initialize the MFCCProcessor with parameters.
        """
        self.n_mfcc = n_mfcc
        self.width = width
        self.sample_rate = sample_rate
        self.winlen = winlen
        self.winstep = winstep
        self.feature_type = feature_type
        self.n_fft = int(winlen * sample_rate)
        self.hop_length = int(winstep * sample_rate)

    def compute_features(self, y):
        """
        Compute MFCC features from raw audio data.

        Parameters:
        - y: numpy array, raw audio data.

        Returns:
        - features: numpy array, computed features.
        """
        # Ensure audio data is in the correct format
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        if np.max(np.abs(y)) > 1.0:
            y = y / np.max(np.abs(y))  # Normalize to [-1, 1]

        # Compute MFCC features
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Compute delta and delta-delta features if required
        if self.feature_type == 'delta':
            delta_mfcc = librosa.feature.delta(mfcc)
            #delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            features = np.concatenate((mfcc, delta_mfcc), axis=0).T
        else:
            features = mfcc.T  # Transpose to have time steps in the first dimension

        return features


# Audio Recording Parameters
FORMAT = pyaudio.paInt16
RATE = 16000
stride = int(50 * (RATE / 1000))
CHUNK_SIZE = stride

# Model and Feature Extraction Parameters
MAXLEN = 16000  # 1 second of audio at 16kHz
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
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT, channels=1, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK_SIZE)

            global QUEUE
            while not stop_event.is_set():
                snd_data = array('h', stream.read(CHUNK_SIZE, exception_on_overflow=False))
                if byteorder == 'big':
                    snd_data.byteswap()
                QUEUE.put(snd_data)
        except Exception as e:
            print(f"Error in RecordingThread: {e}")
            stop_event.set()
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()



class PredictionThread(Thread):
    """
    Thread to process audio data and make predictions.
    """

    def __init__(self, model, mean, std, device, class_labels, n_mfcc, feature_type, num_filters):
        super(PredictionThread, self).__init__()
        self.model = model
        self.device = device
        self.class_labels = class_labels
        self.n_mfcc = n_mfcc
        self.feature_type = feature_type
        self.num_filters = num_filters
        self.mfcc_processor = MFCCProcessor(
            n_mfcc=self.n_mfcc,
            sample_rate=RATE,
            winlen=WINLEN,
            winstep=WINSTEP,
            feature_type=self.feature_type
        )
        self.buffer = array('h')
        self.votes = []
        self.previous_prediction = None

        # Convert mean and std to torch tensors and move to device
        self.mean_tensor = torch.from_numpy(mean).float().to(self.device)
        self.std_tensor = torch.from_numpy(std).float().to(self.device)

    def run(self):
        try:
            global QUEUE
            while not stop_event.is_set():
                try:
                    data = QUEUE.get(timeout=1)  # Timeout to allow checking stop_event
                except Empty:
                    continue  # Check stop_event again
                self.buffer.extend(data)

                if len(self.buffer) < MAXLEN:
                    continue

                # Maintain a sliding window of 1 second
                self.buffer = self.buffer[-MAXLEN:]

                # Convert buffer to numpy array and normalize
                raw_audio = np.array(self.buffer, dtype=np.float32)
                max_val = np.max(np.abs(raw_audio))
                if max_val > 0:
                    raw_audio = raw_audio / max_val  # Normalize to [-1, 1]
                else:
                    raw_audio = raw_audio  # Silent audio

                # Extract features
                features = self.mfcc_processor.compute_features(raw_audio)

                # Pad or truncate to 99 time steps
                desired_time_steps = 99
                current_time_steps = features.shape[0]
                if current_time_steps < desired_time_steps:
                    pad_width = desired_time_steps - current_time_steps
                    features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
                elif current_time_steps > desired_time_steps:
                    features = features[:desired_time_steps, :]

                # Convert features to torch tensor and move to device
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)  # Shape: (1, 99, num_filters)

                # Normalize features using mean and std tensors
                features_tensor = (features_tensor - self.mean_tensor) / self.std_tensor

                # Perform inference
                with torch.no_grad():
                    print("Features_tensor shape:", features_tensor.shape)  # Expected: (1, 99, num_filters))
                    logits = self.model(features_tensor)
                    print("Logits shape:", logits.shape)  # Expected: (99, num_classes)
                    # logits shape: (99, num_classes)

                    # Option 1: Use the logits from the last time step
                    last_logits = logits[-1, :]  # Shape: (num_classes)
                    print("Last logits shape:", last_logits.shape)  # Expected: (num_classes)

                    # Option 2: Average the logits over all time steps
                    # mean_logits = logits.mean(dim=1)  # Shape: (1, num_classes)

                    # Choose one option based on your model and preference
                    # For this example, we'll use the last time step
                    probabilities = softmax(last_logits, dim=0)
                    predicted_index = torch.argmax(probabilities).item()
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
        except Exception as e:
            print(f"Error in PredictionThread: {e}")
            traceback.print_exc()
            stop_event.set()



def initialize_batchnorm_model(config, input_dim, num_classes):
    """
    Initialize the RNNClassifierModel with BatchNorm for inference.
    """
    rnn_name = "FastGRNNBatchNorm"
    ModelClass = get_model_class()

    hidden_units_list = [
        config.model.hidden_units1,
        config.model.hidden_units2,
        config.model.hidden_units3
    ]
    wRank_list = [
        config.model.wRank1,
        config.model.wRank2,
        config.model.wRank3
    ]
    uRank_list = [
        config.model.uRank1,
        config.model.uRank2,
        config.model.uRank3
    ]
    wSparsity_list = [config.model.wSparsity] * len(hidden_units_list)
    uSparsity_list = [config.model.uSparsity] * len(hidden_units_list)

    model = ModelClass(
        rnn_name=rnn_name,
        input_dim=input_dim,
        num_layers=config.model.num_layers,
        hidden_units_list=hidden_units_list,
        wRank_list=wRank_list,
        uRank_list=uRank_list,
        wSparsity_list=wSparsity_list,
        uSparsity_list=uSparsity_list,
        gate_nonlinearity=config.model.gate_nonlinearity,
        update_nonlinearity=config.model.update_nonlinearity,
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
    mean = np.load(args.mean_path)
    std = np.load(args.std_path)
    mean = np.squeeze(mean)
    std = np.squeeze(std)

    # Retrieve MFCC parameters from config or checkpoint
    n_mfcc = checkpoint.get('n_mfcc', 32)  # Default to 32 if not found
    feature_type = checkpoint.get('feature_type', 'delta')  # Default to 'delta' if not found

    # Determine the feature dimension
    if feature_type == 'delta':
        num_filters = n_mfcc * 2
    else:
        num_filters = n_mfcc

    # Update NUM_FILTERS based on actual feature dimension
    NUM_FILTERS = num_filters

    print("Mean shape:", mean.shape)
    print("Std shape:", std.shape)
    print("MFCC n_mfcc:", n_mfcc)
    print("Feature type:", feature_type)
    print("Feature dimension (NUM_FILTERS):", NUM_FILTERS)

    label_encoder = checkpoint['label_encoder']
    class_labels = label_encoder.classes_.tolist()

    # Initialize and load model
    model = initialize_batchnorm_model(config, NUM_FILTERS, len(class_labels))
    print(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Start threads for real-time audio processing
    pred_thread = PredictionThread(model, mean, std, device, class_labels, n_mfcc, feature_type, NUM_FILTERS)
    rec_thread = RecordingThread()

    try:
        pred_thread.start()
        rec_thread.start()

        while not stop_event.is_set():
            # Main thread can perform other tasks or simply wait
            pred_thread.join(timeout=1)
            rec_thread.join(timeout=1)
    except KeyboardInterrupt:
        print("\nStopping keyword detection...")
        stop_event.set()
    finally:
        pred_thread.join()
        rec_thread.join()
        print("Shutdown complete.")
