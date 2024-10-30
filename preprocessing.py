import os
import torch
from sympy.stats.sampling.sample_numpy import numpy
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mfccProcessor import MFCCProcessor


class GoogleSpeechDataset(Dataset):
    def __init__(self, root_dir, processor, max_len=44, exclude_files=None):
        """
        Initialize the dataset.

        Parameters:
        - root_dir: str, Path to the dataset root directory.
        - processor: MFCCProcessor, The processor to extract MFCC features.
        - exclude_files: set, Files to exclude (e.g., validation or test files).
        """
        self.root_dir = str(root_dir)
        self.processor = processor
        self.exclude_files = exclude_files or set()
        self.label_encoder = LabelEncoder()
        self.max_len = max_len
        self.data, self.labels = self.load_dataset()

    def pad_or_truncate(self, mfcc):
        """Pad or truncate the MFCC feature to a fixed length."""
        # If the MFCC is shorter than max_len, pad with zeros
        if mfcc.shape[1] < self.max_len:
            pad_width = self.max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')

        # If the MFCC is longer than max_len, truncate it
        elif mfcc.shape[1] > self.max_len:
            mfcc = mfcc[:, :self.max_len]

        return mfcc

    def load_dataset(self):
        """Scan the entire directory, load all .wav files, and extract labels."""
        features, labels = [], []

        # Scan the entire directory for .wav files
        for root, _, files in os.walk(self.root_dir):
            for file_name in files:
                if file_name.endswith('.wav'):
                    # Construct the relative file path
                    relative_path = os.path.relpath(os.path.join(root, file_name), self.root_dir)

                    # Skip if the file is in the excluded list (validation or test set)
                    if relative_path not in self.exclude_files:
                        # Extract the label from the directory name
                        label = relative_path.split(os.sep)[0]

                        # Compute MFCC features
                        mfcc = self.processor.compute_mfcc(os.path.join(self.root_dir, relative_path))
                        if mfcc is not None:
                            mfcc = self.pad_or_truncate(mfcc)
                            features.append(mfcc)
                            labels.append(label)

        # Encode string labels to integers
        encoded_labels = self.label_encoder.fit_transform(labels)
        features_array = np.array(features)

        # Convert features and labels to PyTorch tensors
        features_tensor = torch.tensor(features_array, dtype=torch.float32)
        labels_tensor = torch.tensor(encoded_labels, dtype=torch.long)

        return features_tensor, labels_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve the feature and label for the given index."""
        # Use clone().detach() to safely return a copy without gradients
        feature = self.data[idx].clone().detach().float()
        label = self.labels[idx].clone().detach().long()
        return feature, label

def load_file_list(file_list_path):
    """Load a list of files from a text file into a set."""
    with open(file_list_path, 'r') as f:
        return set(line.strip() for line in f)


def create_dataloaders(root_dir, batch_size=64):
    """Create DataLoaders for training, validation, and testing datasets."""
    processor = MFCCProcessor()  # Initialize the MFCC Processor

    # Load validation and test files into sets for exclusion
    validation_files = load_file_list('validation_list.txt')
    testing_files = load_file_list('testing_list.txt')

    # Combine validation and test files to exclude them from training
    exclude_files = validation_files | testing_files

    # Create datasets
    train_dataset = GoogleSpeechDataset(root_dir, processor, exclude_files=exclude_files)
    val_dataset = GoogleSpeechDataset(root_dir, processor, validation_files)
    test_dataset = GoogleSpeechDataset(root_dir, processor, testing_files)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
