#MFCCProcessor
import librosa
import numpy as np

class MFCCProcessor:
    def __init__(self, n_mfcc=13, width=9):
        """
        Initialize the MFCCProcessor with parameters.

        Parameters:
        - n_mfcc: int, Number of MFCC coefficients to compute.
        - width: int, Window width used for delta and delta-delta calculations.
        """
        self.n_mfcc = n_mfcc
        self.width = width

    @staticmethod
    def load_audio(audio_file):
        """Load an audio file."""
        try:
            signal, sr = librosa.load(audio_file)
            return signal, sr
        except Exception as e:
            print(f"Error loading audio file {audio_file}: {e}")
            return None, None

    def compute_mfcc(self, audio_file):
        """Compute the MFCCs of an audio file."""
        signal, sr = self.load_audio(audio_file)
        if signal is None:
            return None, None
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=self.n_mfcc)  # Compute MFCCs
        return mfcc, sr

    def compute_delta(self, audio_file):
        """Compute the delta of the MFCCs."""
        mfcc, sr = self.compute_mfcc(audio_file)
        if mfcc is None:
            return None, None
        delta_mfcc = librosa.feature.delta(mfcc, width=self.width)
        mfcc_features = np.concatenate((mfcc, delta_mfcc), axis=0)
        return mfcc_features, sr

    def compute_delta_delta(self, audio_file):
        """Compute the delta-delta (second-order delta) of the MFCCs."""
        mfcc, sr = self.compute_mfcc(audio_file)
        if mfcc is None:
            return None, None
        delta_mfcc = librosa.feature.delta(mfcc, width=self.width)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2, width=self.width)
        mfcc_features = np.concatenate((mfcc, delta_mfcc, delta2_mfcc), axis=0)
        return mfcc_features, sr