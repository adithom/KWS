import os
import shutil
import random
import librosa
import soundfile as sf
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MISSING_FILES = []

# Augmentation functions
def augment_audio(file_path, output_dir, num_copies=2, noise_files=None):
    """
    Augments an audio file by applying time stretching, pitch shift, and background noise.
    Saves the augmented files to the output directory.

    Args:
        file_path (str): Path to the source audio file.
        output_dir (str): Directory to save the augmented audio files.
        num_copies (int): Number of augmented copies to generate.
        noise_files (list): List of noise files for adding background noise.
    """
    y, sr = librosa.load(file_path, sr=None)

    for i in range(num_copies):
        augmented_y = y

        # Random time stretching
        if random.choice([True, False]):
            rate = random.uniform(0.9, 1.1)  # Slight time stretch
            augmented_y = librosa.effects.time_stretch(y = augmented_y, rate = rate)

        # Random pitch shift
        if random.choice([True, False]):
            n_steps = random.randint(-2, 2)  # Pitch shift between -2 and +2 semitones
            augmented_y = librosa.effects.pitch_shift(y = augmented_y, sr = sr, n_steps=n_steps)

        # Add background noise
        if noise_files and random.choice([True, False]):
            noise_file = np.random.choice(noise_files)
            try:
                noise, _ = librosa.load(noise_file, sr=sr)
                if len(noise) < len(augmented_y):
                    noise = np.pad(noise, (0, len(augmented_y) - len(noise)), mode='wrap')
                augmented_y += 0.1 * noise[:len(augmented_y)]
            except FileNotFoundError:
                MISSING_FILES.append(noise_file)
                continue
            except Exception as e:
                print(f"Error processing noise file {noise_file}: {e}")
                MISSING_FILES.append(noise_file)
                continue

        # Save augmented file
        output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_aug{i}.wav")
        sf.write(output_file, augmented_y, sr)
        print(f"Generated augmented file: {output_file}")

# Create augmented dataset
def create_augmented_dataset(source_dir, target_dir, augmentation_fraction=0.25, val_test_split=0.15):
    """
    Creates an augmented dataset with 25% augmentation in each directory.
    15% of both augmented and unaugmented files are added to validation and test sets.

    Args:
        source_dir (str): Path to the source directory.
        target_dir (str): Path to the target directory.
        augmentation_fraction (float): Fraction of augmented files per directory.
        val_test_split (float): Fraction of files for validation/test datasets.
        num_copies (int): Number of augmented copies for each file.
    """
    os.makedirs(target_dir, exist_ok=True)
    noise_files = []

    # Prepare the noise files
    noise_dir = os.path.join(source_dir, "noise")
    if os.path.exists(noise_dir):
        noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.wav')]

    # Iterate over each subdirectory
    for directory in os.listdir(source_dir):
        src_dir_path = os.path.join(source_dir, directory)
        if not os.path.isdir(src_dir_path) or directory.startswith("_"):
            continue

        dest_train_dir = os.path.join(target_dir, "train", directory)
        dest_val_dir = os.path.join(target_dir, "val", directory)
        dest_test_dir = os.path.join(target_dir, "test", directory)

        os.makedirs(dest_train_dir, exist_ok=True)
        os.makedirs(dest_val_dir, exist_ok=True)
        os.makedirs(dest_test_dir, exist_ok=True)

        # Get all wav files in the directory
        files = [os.path.join(src_dir_path, f) for f in os.listdir(src_dir_path) if f.endswith('.wav')]
        logger.info(f"Processing directory: {directory} with {len(files)} files.")

        # Select files for augmentation
        num_augmented = int(len(files) * augmentation_fraction)
        selected_files = random.sample(files, num_augmented)

        # Augment files and save to the train directory
        for file in selected_files:
            augment_audio(file, dest_train_dir, noise_files=noise_files)

        # Combine augmented and unaugmented files
        all_files = files + [os.path.join(dest_train_dir, f) for f in os.listdir(dest_train_dir) if f.endswith('.wav')]

        # Split files for validation and test
        num_val_test = int(len(all_files) * val_test_split)
        val_files = random.sample(all_files, num_val_test)
        test_files = random.sample([f for f in all_files if f not in val_files], num_val_test)

        # Move validation and test files
        for file in val_files:
            shutil.copy(file, os.path.join(dest_val_dir, os.path.basename(file)))
        for file in test_files:
            shutil.copy(file, os.path.join(dest_test_dir, os.path.basename(file)))

        # Copy remaining files to the train directory
        remaining_files = [f for f in all_files if f not in val_files and f not in test_files]
        for file in remaining_files:
            if not os.path.exists(os.path.join(dest_train_dir, os.path.basename(file))):
                shutil.copy(file, dest_train_dir)

        logger.info(f"Processed {directory}: {len(val_files)} val, {len(test_files)} test, {len(remaining_files)} train.")

if __name__ == "__main__":
    source_directory = "google_12"
    target_directory = "google_12_augmented"
    create_augmented_dataset(source_directory, target_directory, val_test_split=0.15)
