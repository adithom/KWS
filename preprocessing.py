import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from mfccprocessor import MFCCProcessor

class GoogleSpeechDataset(Dataset):
    def __init__(self, root_dir, processor, file_list=None):
        """
        Initialize the dataset.

        Parameters:
        - root_dir: str, Path to the dataset root directory.
        - processor: MFCCProcessor, The processor to extract MFCC features.
        - file_list: str or list, Path to a text file with audio file paths or a list of audio file paths.
        """
        self.root_dir = root_dir
        self.processor = processor
        self.label_encoder = LabelEncoder()
        self.data, self.labels = self.load_dataset(file_list)

    def load_dataset(self, file_list):
        """Load audio files and labels from the directory or provided file list."""
        features, labels = [], []

        # If file_list is a string, assume it's a path to a text file with audio paths
        if isinstance(file_list, str):
            with open(file_list, 'r') as f:
                audio_files = [line.strip() for line in f]
        elif isinstance(file_list, list):
            audio_files = file_list  # Directly use the list if provided
        else:
            audio_files = []  # Empty list if no file list provided

        # Load audio and extract features
        for audio_path in audio_files:
            full_path = os.path.join(self.root_dir, audio_path)
            label = audio_path.split('/')[0]  # Extract label from directory name
            mfcc = self.processor.compute_mfcc(full_path)
            if mfcc is not None:
                features.append(mfcc)
                labels.append(label)

        # Encode string labels into integers
        encoded_labels = self.label_encoder.fit_transform(labels)
        return np.array(features), np.array(encoded_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

def load_file_list(file_list_path):
    """Load file paths from a text file."""
    with open(file_list_path, 'r') as f:
        return set(line.strip() for line in f)

def get_training_file_list(root_dir, val_files, test_files):
    """Get a list of training files, excluding validation and test files."""
    training_files = []
    for label in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label)
        if os.path.isdir(label_path):
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label, file_name)  # Relative path
                if file_path not in val_files and file_path not in test_files:
                    training_files.append(file_path)
    return training_files

def create_dataloaders(root_dir, batch_size=32):
    """Create DataLoaders for training, validation, and test datasets."""
    processor = MFCCProcessor()  # Initialize the MFCC Processor

    # Load validation and testing file lists
    validation_files = load_file_list('validation_list.txt')
    testing_files = load_file_list('testing_list.txt')

    # Get the training files, excluding validation and testing files
    train_files = get_training_file_list(root_dir, validation_files, testing_files)

    # Create datasets
    train_dataset = GoogleSpeechDataset(root_dir, processor, train_files)
    val_dataset = GoogleSpeechDataset(root_dir, processor, 'validation_list.txt')
    test_dataset = GoogleSpeechDataset(root_dir, processor, 'testing_list.txt')

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
