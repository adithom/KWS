#MFCCProcessor
import librosa
import numpy as np

class MFCCProcessor:
    def __init__(self, n_mfcc=32, width=9, sample_rate = 16000, winlen = 0.025, winstep = 0.010, feature_type='mfcc'):
        """
        Initialize the MFCCProcessor with parameters.

        Parameters:
        - n_mfcc: int, Number of MFCC coefficients to compute.
        - width: int, Window width used for delta and delta-delta calculations.
        """
        self.n_mfcc = n_mfcc
        self.width = width
        self.sample_rate = sample_rate
        self.winlen = winlen
        self.winstep = winstep
        self.feature_type = feature_type
        self.n_fft = int(winlen * sample_rate)  # Number of samples per window
        self.hop_length = int(winstep * sample_rate)

    def compute_features(self, audio_file):
        """Compute features based on the specified type."""
        if self.feature_type == 'mfcc':
            return self.compute_mfcc(audio_file)
        elif self.feature_type == 'delta':
            return self.compute_delta(audio_file)
        elif self.feature_type == 'delta_delta':
            return self.compute_delta_delta(audio_file)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")

    def load_audio(self, audio_file):
        """Load an audio file."""
        try:
            signal, sr = librosa.load(audio_file, sr=self.sample_rate)
            return signal, sr
        except Exception as e:
            print(f"Error loading audio file {audio_file}: {e}")
            return None, None

    def compute_mfcc(self, audio_file):
        """Compute the MFCCs of an audio file."""
        signal, sr = self.load_audio(audio_file)
        if signal is None:
            return None
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        return mfcc

    def compute_delta(self, audio_file):
        """Compute the delta of the MFCCs."""
        mfcc = self.compute_mfcc(audio_file)
        if mfcc is None:
            return None
        delta_mfcc = librosa.feature.delta(mfcc, width=self.width)
        mfcc_features = np.concatenate((mfcc, delta_mfcc), axis=0)
        return mfcc_features

    def compute_delta_delta(self, audio_file):
        """Compute the delta-delta (second-order delta) of the MFCCs."""
        mfcc= self.compute_mfcc(audio_file)
        if mfcc is None:
            return None
        delta_mfcc = librosa.feature.delta(mfcc, width=self.width)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2, width=self.width)
        mfcc_features = np.concatenate((mfcc, delta_mfcc, delta2_mfcc), axis=0)
        return mfcc_features