import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mfccProcessor import MFCCProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#todo: Test time augmentation: time shifting, time stretching, pitch augmentation

class GoogleSpeechDataset(Dataset):
    def __init__(self, root_dir, processor, max_len=99, training=False, feature_type = 'mfcc', mean=None, std=None):
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
        # if include_files:
        #     self.include_files = set()
        #     for f in include_files:
        #         self.include_files.add(os.path.normpath(f))
        #         self.include_files.add(f.replace('\\', '/'))
        #         self.include_files.add(f.replace('/', '\\'))
        # else:
        #     self.include_files = set()
        #
        # if exclude_files:
        #     self.exclude_files = set()
        #     for f in exclude_files:
        #         self.exclude_files.add(os.path.normpath(f))
        #         self.exclude_files.add(f.replace('\\', '/'))
        #         self.exclude_files.add(f.replace('/', '\\'))
        # else:
        #     self.exclude_files = set()
        self.label_encoder = LabelEncoder()
        self.max_len = max_len
        self.training = training
        self.feature_type = feature_type

        self.mean = mean
        self.std = std

        logger.info(f"Dataset initialization - Mode: {'Training' if training else 'Validation/Test'}")
        logger.info(f"Root directory: {self.root_dir}")
        # logger.info(f"Number of include files: {len(self.include_files)}")
        # logger.info(f"Number of exclude files: {len(self.exclude_files)}")
        
        self.data, self.labels = self.load_dataset()
        
    def compute_normalization_stats(self, features_array):
        """Compute mean and standard deviation for normalization."""
        # Compute along all axes except the batch dimension
        self.mean = np.mean(features_array, axis=(0, 2), keepdims=True)
        self.std = np.std(features_array, axis=(0, 2), keepdims=True)
        self.std = np.where(self.std == 0, 1e-6, self.std)
        return self.mean, self.std
        
    def normalize_features(self, features_array):
        """Normalize features using mean and standard deviation."""
        if self.mean is None or self.std is None:
            if self.training:
                self.compute_normalization_stats(features_array)
            else:
                raise ValueError("Normalization parameters not set for non-training dataset")

        normalized_features = (features_array - self.mean) / self.std
        return normalized_features

    def pad_or_truncate(self, mfcc):
        """Pad or truncate the MFCC feature to a fixed length."""
        if mfcc.shape[1] < self.max_len:
            pad_width = self.max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')

        elif mfcc.shape[1] > self.max_len:
            mfcc = mfcc[:, :self.max_len]

        return mfcc

#     def load_dataset(self):
#         """Scan the entire directory, load all .wav files, and extract labels."""
#         features, labels = [], []
#         processed_files = 0
#         skipped_files = 0
#
#         # Scan the entire directory for .wav files
#         for root, _, files in os.walk(self.root_dir):
#             subdirectory = os.path.relpath(root, self.root_dir)
#             if subdirectory in ('.', './idea'):
#                 continue
#             wav_files = [f for f in files if f.endswith('.wav')]
#             logger.info(f"Processing subdirectory: {subdirectory} - Found {len(wav_files)} WAV files")
#             for file_name in wav_files:
#                 relative_path = os.path.normpath(os.path.relpath(os.path.join(root, file_name), self.root_dir))
#
#                 logger.debug(f"Processing file: {relative_path}")
#
#                 # Extract the label from the directory name
#                 label = os.path.basename(os.path.dirname(relative_path))
#
#                 # Compute features
#                 full_path = os.path.join(self.root_dir, relative_path)
#                 mfcc = self.processor.compute_features(full_path)
#
#                 if mfcc is not None:
#                     mfcc = self.pad_or_truncate(mfcc)
#                     features.append(mfcc)
#                     labels.append(label)
#                     processed_files += 1
#                 else:
#                     logger.warning(f"Failed to extract features from {relative_path}")
#
#             logger.info(f"Dataset loading complete - Processed {processed_files} files, Skipped {skipped_files} files")
#             logger.info(f"Final dataset size: {len(features)} samples")
#
#         if not features:
#             logger.error("No features were extracted. Debug information:")
#             logger.error(f"Training mode: {self.training}")
#             logger.error(f"Include files count: {len(self.include_files)}")
#             logger.error(f"Exclude files count: {len(self.exclude_files)}")
#             raise ValueError("No features were extracted. Please check dataset paths and file filtering criteria.")
#
#         # Encode string labels to integers
#         encoded_labels = self.label_encoder.fit_transform(labels)
#         features_array = np.stack(features)
#         #todo: stack or array
#         if self.training and self.mean is None and self.std is None:
#             self.mean, self.std = self.compute_normalization_stats(features_array)
#         normalized_features = self.normalize_features(features_array)
#
#         features_tensor = torch.tensor(normalized_features, dtype=torch.float32)
#         labels_tensor = torch.tensor(encoded_labels, dtype=torch.long)
#
#         return features_tensor, labels_tensor
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         """Retrieve the feature and label for the given index.
#         This is PyTorch requirement for Dataset class."""
#         feature = self.data[idx].clone().detach().float()
#         label = self.labels[idx].clone().detach().long()
#         return feature, label
#
#
# def load_file_list(file_list_path):
#     """Load a list of files from a text file into a set, handling different path formats."""
#     files = set()
#     valid_subdirs = {'yes','no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop','go', 'zero', 'one'}
#
#     with open(file_list_path, 'r') as f:
#         for line in f:
#             path = os.path.normpath(line.strip())
#             parts = path.split(os.sep)
#
#             if parts[0] not in valid_subdirs:
#                 continue
#
#             files.add(path)
#             files.add(path.replace('\\', '/'))
#             files.add(path.replace('/', '\\'))
#
#     logger.info(f"Loaded {len(files)} valid files from {file_list_path}")
#     return files
#
#
# def create_dataloaders(root_dir, batch_size=128, feature_type='mfcc', num_workers=4, pin_memory=False):
#     """Create normalized DataLoaders for training, validation, and testing datasets."""
#     processor = MFCCProcessor(feature_type=feature_type)
#
#     try:
#         validation_files = load_file_list(os.path.join(root_dir, 'validation_list.txt'))
#     except FileNotFoundError:
#         print("Warning: validation_list.txt not found. Proceeding without validation files.")
#         validation_files = set()
#
#     try:
#         testing_files = load_file_list(os.path.join(root_dir, 'testing_list.txt'))
#     except FileNotFoundError:
#         print("Warning: testing_list.txt not found. Proceeding without testing files.")
#         testing_files = set()
#     exclude_files = validation_files | testing_files
#     logger.info(f"Number of exclude_files: {len(exclude_files)}")
#     logger.info(f"Number of include_files: {len(validation_files) + len(testing_files)}")
#
#     # Create training dataset first to compute normalization parameters
#     train_dataset = GoogleSpeechDataset(root_dir, processor, exclude_files=exclude_files, training=True, feature_type=feature_type)
#
#     if train_dataset.mean is None or train_dataset.std is None:
#         raise ValueError(
#             "Failed to compute mean and std for training dataset. Check the data and feature extraction pipeline.")
#
#     #print(f"Training mean: {train_dataset.mean}, Training std: {train_dataset.std}")
#
#     val_dataset = GoogleSpeechDataset(root_dir, processor, include_files=validation_files, training=False, feature_type=feature_type,mean=train_dataset.mean, std=train_dataset.std)
#     test_dataset = GoogleSpeechDataset(root_dir, processor, include_files=testing_files, training=False, feature_type=feature_type, mean=train_dataset.mean, std=train_dataset.std)
#
#     # Create DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
#
#     return train_loader, val_loader, test_loader

    # def load_dataset(self):
    #     """Load all .wav files and extract labels."""
    #     features, labels = [], []
    #
    #     subdirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
    #     logger.info(f"Found {len(subdirs)} subdirectories (classes) in {self.root_dir}")
    #
    #     for subdir in subdirs:
    #         subdir_path = os.path.join(self.root_dir, subdir)
    #         wav_files = [f for f in os.listdir(subdir_path) if f.endswith('.wav')]
    #         logger.info(f"Processing directory: {subdir} with {len(wav_files)} files")
    #
    #         for wav_file in wav_files:
    #             file_path = os.path.join(subdir_path, wav_file)
    #             mfcc = self.processor.compute_features(file_path, feature_type=self.feature_type)
    #
    #             if mfcc is not None:
    #                 mfcc = self.pad_or_truncate(mfcc)
    #                 features.append(mfcc)
    #                 labels.append(subdir)
    #             else:
    #                 logger.warning(f"Failed to process {file_path}")
    #
    #     if not features:
    #         raise ValueError(f"No features were extracted from {self.root_dir}")
    #
    #     # Encode string labels to integers
    #     encoded_labels = self.label_encoder.fit_transform(labels)
    #     features_array = np.stack(features)
    #
    #     if self.training and self.mean is None and self.std is None:
    #         self.mean, self.std = self.compute_normalization_stats(features_array)
    #
    #     normalized_features = self.normalize_features(features_array)
    #     features_tensor = torch.tensor(normalized_features, dtype=torch.float32)
    #     labels_tensor = torch.tensor(encoded_labels, dtype=torch.long)
    #
    #     return features_tensor, labels_tensor

    def load_dataset(self):
        """Scan the entire directory, load all .wav files, and extract labels."""
        features, labels = [], []
        processed_files = 0
        skipped_files = 0

        # Scan the entire directory for .wav files
        for root, _, files in os.walk(self.root_dir):
            subdirectory = os.path.relpath(root, self.root_dir)
            if subdirectory in ('.', './idea'):
                continue
            wav_files = [f for f in files if f.endswith('.wav')]
            logger.info(f"Processing subdirectory: {subdirectory} - Found {len(wav_files)} WAV files")
            for file_name in wav_files:
                relative_path = os.path.normpath(os.path.relpath(os.path.join(root, file_name), self.root_dir))

                logger.debug(f"Processing file: {relative_path}")

                # Extract the label from the directory name
                label = os.path.basename(os.path.dirname(relative_path))

                # Compute features
                full_path = os.path.join(self.root_dir, relative_path)
                mfcc = self.processor.compute_features(full_path)

                if mfcc is not None:
                    mfcc = self.pad_or_truncate(mfcc)
                    features.append(mfcc)
                    labels.append(label)
                    processed_files += 1
                else:
                    logger.warning(f"Failed to extract features from {relative_path}")

            logger.info(f"Dataset loading complete - Processed {processed_files} files, Skipped {skipped_files} files")
            logger.info(f"Final dataset size: {len(features)} samples")

        if not features:
            logger.error("No features were extracted. Debug information:")
            logger.error(f"Training mode: {self.training}")
            logger.error(f"Include files count: {len(self.include_files)}")
            logger.error(f"Exclude files count: {len(self.exclude_files)}")
            raise ValueError("No features were extracted. Please check dataset paths and file filtering criteria.")

        # Encode string labels to integers
        encoded_labels = self.label_encoder.fit_transform(labels)
        features_array = np.stack(features)
        #todo: stack or array
        if self.training and self.mean is None and self.std is None:
            self.mean, self.std = self.compute_normalization_stats(features_array)
        normalized_features = self.normalize_features(features_array)

        features_tensor = torch.tensor(normalized_features, dtype=torch.float32)
        labels_tensor = torch.tensor(encoded_labels, dtype=torch.long)

        return features_tensor, labels_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.data[idx].clone().detach().float()
        label = self.labels[idx].clone().detach().long()
        return feature, label

def create_dataloaders(data_dir, batch_size=128, feature_type='mfcc', num_workers=4, pin_memory=False):
    """
    Create DataLoaders for train, validation, and test datasets based on directory structure.
    """
    processor = MFCCProcessor(feature_type=feature_type)

    # Training dataset and DataLoader
    train_dataset = GoogleSpeechDataset(
        root_dir=os.path.join(data_dir, 'train'),
        processor=processor,
        training=True,
        feature_type=feature_type
    )

    # Validation dataset and DataLoader
    val_dataset = GoogleSpeechDataset(
        root_dir=os.path.join(data_dir, 'val'),
        processor=processor,
        training=False,
        feature_type=feature_type,
        mean=train_dataset.mean,
        std=train_dataset.std
    )

    # Test dataset and DataLoader
    test_dataset = GoogleSpeechDataset(
        root_dir=os.path.join(data_dir, 'test'),
        processor=processor,
        training=False,
        feature_type=feature_type,
        mean=train_dataset.mean,
        std=train_dataset.std
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    data_directory = "google_12_augmented"
    train_loader, val_loader, test_loader = create_dataloaders(data_directory, batch_size=64)

    for batch in train_loader:
        features, labels = batch
        print(f"Feature shape: {features.shape}, Labels: {labels}")
        break
