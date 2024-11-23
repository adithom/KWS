import os
import shutil
import random
import librosa
import soundfile as sf

#todo: remove one and zero and add noise and background noise

def augment_audio(file_path, output_dir, num_copies=10):
    """
    Augments an audio file by applying slight variations and saves them to the output directory.

    Args:
        file_path (str): Path to the source audio file.
        output_dir (str): Directory to save the augmented audio files.
        num_copies (int): Number of augmented copies to generate.
    """
    y, sr = librosa.load(file_path, sr=None)
    for i in range(num_copies):
        augmented_y = y
        if random.choice([True, False]):  
            rate = random.uniform(0.9, 1.1)  
            augmented_y = librosa.effects.time_stretch(augmented_y, rate)

        if random.choice([True, False]):  
            n_steps = random.randint(-2, 2)  
            augmented_y = librosa.effects.pitch_shift(augmented_y, sr, n_steps)

        output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_aug{i}.wav")
        sf.write(output_file, augmented_y, sr)
        print(f"Generated augmented file: {output_file}")

def copy_google_speech_data(source_dir, target_dir, sample_size=100):
    """
    Copies specified subdirectories and files from source_dir to target_dir.

    Args:
        source_dir (str): Path to the source directory (google_speech).
        target_dir (str): Path to the target directory (google_12).
        sample_size (int): Number of random noise samples to collect.
    """
    directories_to_copy = [
        'yes','no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop','go', 'zero', 'one'
    ]

    files_to_copy = ['validation_list.txt', 'testing_list.txt']

    os.makedirs(target_dir, exist_ok=True)

    # Copy specified directories
    for directory in directories_to_copy:
        src_dir_path = os.path.join(source_dir, directory)
        dest_dir_path = os.path.join(target_dir, directory)
        if os.path.exists(src_dir_path):
            shutil.copytree(src_dir_path, dest_dir_path, dirs_exist_ok=True)
            print(f"Copied directory: {directory}")
        else:
            print(f"Directory not found, skipping: {directory}")

    # Copy specified files
    for file_name in files_to_copy:
        src_file_path = os.path.join(source_dir, file_name)
        dest_file_path = os.path.join(target_dir, file_name)
        if os.path.exists(src_file_path):
            shutil.copy2(src_file_path, dest_file_path)
            print(f"Copied file: {file_name}")
        else:
            print(f"File not found, skipping: {file_name}")

    print(f"All specified files and directories have been copied to {target_dir}")
    
    noise_dir = os.path.join(target_dir, "noise")
    os.makedirs(noise_dir, exist_ok=True)
    
    remaining_directories = [
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d)) and d not in directories_to_copy and not d.startswith('_')
    ]

    noise_files = []
    for directory in remaining_directories:
        src_dir_path = os.path.join(source_dir, directory)
        if os.path.exists(src_dir_path):
            files = [os.path.join(src_dir_path, f) for f in os.listdir(src_dir_path) if f.endswith('.wav')]
            noise_files.extend(files)

    sampled_files = random.sample(noise_files, min(sample_size, len(noise_files)))
    for file_path in sampled_files:
        shutil.copy2(file_path, noise_dir)
        print(f"Copied to noise: {os.path.basename(file_path)}")
    
    background_noise_dir = os.path.join(source_dir, "_background_noise_")
    if os.path.exists(background_noise_dir):
        bg_files = [os.path.join(background_noise_dir, f) for f in os.listdir(background_noise_dir) if f.endswith('.wav')]
        for bg_file in bg_files:
            # Generate augmented copies
            augment_audio(bg_file, noise_dir, num_copies=10)
    
    print(f"{len(sampled_files)} files have been assigned to the noise class.")

if __name__ == "__main__":
    source_directory = "google_speech"
    target_directory = "google_12"
    copy_google_speech_data(source_directory, target_directory, sample_size=3000)
