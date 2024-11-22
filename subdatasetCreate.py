import os
import shutil

#todo: remove one and zero and add noise and background noise

def copy_google_speech_data(source_dir, target_dir):
    """
    Copies specified subdirectories and files from source_dir to target_dir.

    Args:
        source_dir (str): Path to the source directory (google_speech).
        target_dir (str): Path to the target directory (google_12).
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


# Example usage
if __name__ == "__main__":
    source_directory = "google_speech"
    target_directory = "google_12"
    copy_google_speech_data(source_directory, target_directory)
