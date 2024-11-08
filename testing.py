import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from mfccProcessor import MFCCProcessor
from preprocessing import GoogleSpeechDataset

processor = MFCCProcessor()
dataset = GoogleSpeechDataset('google_speech/dog', processor, training=True)  # First run with training=True

# Store normalization parameters
mean = dataset.mean
std = dataset.std



# Print dataset length
print(f'Total samples: {len(dataset)}')

# Access the first sample
feature, label = dataset[0]
print(f'Feature shape: {feature.shape}, Label: {label}')

# Use the processor to compute MFCC features directly for this one-second file
# audio = 'google_speech/yes/0a7c2a8d_nohash_0.wav'
# features = processor.compute_delta_delta(audio)
#
# # Calculate the shape of the features
# print(f"Shape of MFCC features for a one-second audio file: {features.shape}")
#
# # Print the expected number of frames for a one-second file based on hop length and n_fft
# n_fft = processor.n_fft
# hop_length = processor.hop_length
# sample_rate = processor.sample_rate
#
# # Expected number of frames
# expected_num_frames = 1 + (sample_rate - n_fft) // hop_length
# print(f"Expected number of frames for a one-second audio file: {expected_num_frames}")
#
# # Check the feature length matches the expected number of frames
# print("Feature shape:", features.shape)  # Should be (n_mfcc, num_frames)
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(features, x_axis='time', sr=processor.sample_rate,hop_length=processor.hop_length, cmap='viridis', vmin=-50, vmax=50)
# plt.colorbar(format='%+2.0f dB')
# plt.title("MFCC")
# plt.xlabel("Time (s)")
# plt.ylabel("MFCC Coefficients")
# plt.tight_layout()
# plt.show()

'''
Returns:
Total samples: 1592
Feature shape: torch.Size([13, 44]), Label: 0

Process finished with exit code 0

Total samples: 2128
Feature shape: torch.Size([26, 44]), Label: 0

Process finished with exit code 0
'''