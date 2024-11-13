import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mfcc_processor import MFCCProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GoogleSpeechDataset(Dataset):
    def __init__(self, root_dir, processor, max_len=98, include_files=None, exclude_files=None, training=False, feature_type = 'mfcc', mean=None, std=None):
        """
        Initialize the dataset.

        Parameters:
        - root_dir: str, Path to the dataset root directory.
        - processor: MFCCProcessor, The processor to extract MFCC features.
        - exclude_files: set, Files to exclude (e.g., validation or test files).
        - max_len: int, Maximum length of the MFCC features. Decided based on expected_num_frames = 1 + (sample_rate - n_fft) // hop_length
        """
        self.root_dir = str(root_dir)
        self.processor = processor
        self.include_files = include_files or set()
        self.exclude_files = exclude_files or set()
        #todo: integrate exclude files from create_dataloaders
        self.label_encoder = LabelEncoder()
        self.max_len = max_len
        self.training = training
        self.feature_type = feature_type
        
        # Normalization parameters
        self.mean = mean
        self.std = std
        
        self.data, self.labels = self.load_dataset()
        
    def compute_normalization_stats(self, features_array):
        """Compute mean and standard deviation for normalization."""
        # Compute along all axes except the batch dimension
        self.mean = np.mean(features_array, axis=(0, 2), keepdims=True)
        self.std = np.std(features_array, axis=(0, 2), keepdims=True)
        # Avoid division by zero
        self.std = np.where(self.std == 0, 1e-6, self.std)
        return self.mean, self.std
        
    def normalize_features(self, features_array):
        """Normalize features using mean and standard deviation."""
        if self.mean is None or self.std is None:
            if self.training:
                self.compute_normalization_stats(features_array)
            else:
                raise ValueError("Normalization parameters not set for non-training dataset")
        
        # Apply normalization
        normalized_features = (features_array - self.mean) / self.std
        return normalized_features

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
            subdirectory = os.path.relpath(root, self.root_dir)
            logger.info(f"Processing subdirectory: {subdirectory}")
            for file_name in files:
                if file_name.endswith('.wav'):
                    # Construct the relative file path
                    relative_path = os.path.relpath(os.path.join(root, file_name), self.root_dir)

                    # Skip if the file is in the excluded list (validation or test set)
                    if (self.include_files and relative_path not in self.include_files) or (self.exclude_files and relative_path in self.exclude_files):
                        continue
                    # Extract the label from the directory name
                    label = relative_path.split(os.sep)[0]

                    # Compute MFCC features
                    mfcc = self.processor.compute_features(os.path.join(self.root_dir, relative_path),self.feature_type)
                    if mfcc is not None:
                        mfcc = self.pad_or_truncate(mfcc)
                        features.append(mfcc)
                        labels.append(label)

        if not features:
            raise ValueError("No features were extracted. Please check dataset paths and file filtering criteria.")

        # Encode string labels to integers
        encoded_labels = self.label_encoder.fit_transform(labels)
        features_array = np.stack(features)
        #todo: stack or array
        if self.training and self.mean is None and self.std is None:
            # Compute mean and std if training and not provided
            self.mean, self.std = self.compute_normalization_stats(features_array)
        normalized_features = self.normalize_features(features_array)

        # Convert features and labels to PyTorch tensors
        features_tensor = torch.tensor(normalized_features, dtype=torch.float32)
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


def create_dataloaders(root_dir, batch_size=128, feature_type='mfcc', num_workers=4, pin_memory=False):
    """Create normalized DataLoaders for training, validation, and testing datasets."""
    processor = MFCCProcessor()

    try:
        validation_files = load_file_list(os.path.join(root_dir, 'validation_list.txt'))
    except FileNotFoundError:
        print("Warning: validation_list.txt not found. Proceeding without validation files.")
        validation_files = set()

    try:
        testing_files = load_file_list(os.path.join(root_dir, 'testing_list.txt'))
    except FileNotFoundError:
        print("Warning: testing_list.txt not found. Proceeding without testing files.")
        testing_files = set()
    exclude_files = validation_files | testing_files

    # Create training dataset first to compute normalization parameters
    train_dataset = GoogleSpeechDataset(root_dir, processor, exclude_files=exclude_files, training=True, feature_type=feature_type)

    if train_dataset.mean is None or train_dataset.std is None:
        raise ValueError(
            "Failed to compute mean and std for training dataset. Check the data and feature extraction pipeline.")

    print(f"Training mean: {train_dataset.mean}, Training std: {train_dataset.std}")

    # Apply training set's normalization parameters to val and test sets
    val_dataset = GoogleSpeechDataset(root_dir, processor, include_files=validation_files, training=False, feature_type=feature_type,mean=train_dataset.mean, std=train_dataset.std)
    test_dataset = GoogleSpeechDataset(root_dir, processor, include_files=testing_files, training=False, feature_type=feature_type, mean=train_dataset.mean, std=train_dataset.std)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
