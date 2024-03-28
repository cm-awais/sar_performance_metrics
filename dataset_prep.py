import os
import shutil
import random
import zipfile
from zipfile  import ZipFile 
import rarfile


def extract_fusar(fusar_archive_path):
   with ZipFile(fusar_archive_path, 'r') as zObject: 
    zObject.extractall() 

def process_nested_archive(nested_archive, destination_folder):
    """
    Extracts data from a nested archive.

    Args:
        nested_archive: The opened nested zip/rar archive.
        destination_folder: Destination folder for extracted data.
    """

    for nested_member in nested_archive.namelist():
        if nested_member.startswith('Patch_RGB/'):
            nested_archive.extract(nested_member, destination_folder)

def extract_data(main_archive_path, destination_folder="open_sar_ship_dataset_patch"):
    """
    Extracts data from nested zip/rar files within a main archive.

    Args:
        main_archive_path: Path to the main zip/rar file.
        destination_folder: Destination folder for extracted data (defaults to "open_sar_ship_dataset_patch").
    """

    os.makedirs(destination_folder, exist_ok=True)  # Create destination folder if needed

    with zipfile.ZipFile(main_archive_path, 'r') as main_archive:
        for member in main_archive.namelist():
            if member.endswith('.zip') or member.endswith('.rar'):
                if member.endswith('.zip'):
                    with zipfile.ZipFile(main_archive.open(member), 'r') as nested_archive:
                        process_nested_archive(nested_archive, destination_folder)
                else:  # member ends with '.rar'
                    with rarfile.RarFile(main_archive.open(member), 'r') as nested_archive:
                        process_nested_archive(nested_archive, destination_folder)
            elif member.startswith('Patch_RGB/'):  # Extract directly from main archive
                main_archive.extract(member, destination_folder)

def load_data_and_stats(folder_path):
  """
  Loads data from a folder and shows statistics by class names.

  Args:
    folder_path: Path to the folder containing data files.

  Returns:
    A dictionary with statistics for each class, including:
      - count: Number of files belonging to the class.
      - unique_filenames: List of unique filenames for the class.
  """

  class_stats = {}

  for filename in os.listdir(folder_path):
    class_name = filename.split("_")[0]  # Assuming first word before _ is class
    if class_name not in class_stats:
      class_stats[class_name] = {
          "count": 0,
          "unique_filenames": []
      }

    class_stats[class_name]["count"] += 1
    class_stats[class_name]["unique_filenames"].append(filename)

  return class_stats

def processing_open_sarship(src_folder, dest_folder):
  """
  Extracting only specific classes with png extension
  """
  choosen = ["Cargo", "Tanker", "Fishing"]

  src_folder += "/Patch_RGB"
  stats = load_data_and_stats(src_folder)

  for key, value in stats.items():
    if key in choosen:
      # Create class folder if it doesn't exist
      class_folder = os.path.join(dest_folder, key)
      os.makedirs(class_folder, exist_ok=True)

      for filename in value['unique_filenames']:
        if filename.endswith("vv.png"):
          # Copy the file to the class folder
          source_path = os.path.join(src_folder, filename)
          destination_path = os.path.join(class_folder, filename)
          shutil.copy2(source_path, destination_path)

def split_folder(data_dir, new_dir, test_size=0.2, shuffle=True):
  """
  Splits a folder containing subfolders for classes into train and test sets.

  Args:
    data_dir: Path to the folder containing class subfolders.
    test_size: Proportion of data to be used for the test set (default: 0.2).
    shuffle: Whether to shuffle the files before splitting (default: True).
  """
  # Create train and test directories
  train_dir = os.path.join(new_dir, "train")
  test_dir = os.path.join(new_dir, "test")
  for class_dir in os.listdir(data_dir):
    if class_dir in [".", ".."]:
      continue
    class_path = os.path.join(data_dir, class_dir)
    
    os.makedirs(os.path.join(train_dir, class_dir), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_dir), exist_ok=True)

    # Get all files in the class directory
    files = os.listdir(class_path)
    if shuffle:
      random.shuffle(files)

    # Split files into train and test sets
    split_index = int(len(files) * test_size)
    train_files, test_files = files[:split_index], files[split_index:]

    # Move files to train and test directories
    for filename in train_files:
      src = os.path.join(class_path, filename)
      dst = os.path.join(train_dir, class_dir, filename)
      shutil.copy2(src, dst)
    for filename in test_files:
      src = os.path.join(class_path, filename)
      dst = os.path.join(test_dir, class_dir, filename)
      shutil.copy2(src, dst)

def merge_folders(source_dir1, source_dir2, destination_dir):
  """
  Merges the contents of two folders into a new folder.

  Args:
      source_dir1: Path to the first source folder.
      source_dir2: Path to the second source folder.
      destination_dir: Path to the destination folder.
  """
  # Create the destination directory if it doesn't exist
  os.makedirs(destination_dir, exist_ok=True)

  # Merge files from both source folders
  for source_dir in [source_dir1, source_dir2]:
    for root, _, files in os.walk(source_dir):
      for file in files:
        source_file = os.path.join(root, file)
        destination_file = os.path.join(destination_dir, os.path.relpath(source_file, source_dir))
        # Use shutil.copy2() to preserve file metadata (e.g., creation time, permissions)
        shutil.copy2(source_file, destination_file)

  print(f"Successfully merged folders into {destination_dir}")

def main():

  openSarship_archive_path = 'OpenSARShip_2.zip' # https://opensar.sjtu.edu.cn/openSAR/OpenSARShip_2.zip
  fusar_archive_path = "fusar.zip" # https://emwlab.fudan.edu.cn/67/05/c20899a222981/page.htm
  opensarship_extract_path = "open_sar_ship_extract"
  fusar_path = "FUSAR_Ship1.0"
  open_sar_path = "opensar_data"
  mix_path = "mix"

  split_fusar = "fusar_split"
  split_open_sar_path = "opensar_split"
  split_mix_path = "mix_split"



  sub_folders = ["Cargo", "Fishing", "Tanker"]

  extract_data(openSarship_archive_path, destination_folder=opensarship_extract_path)
  extract_fusar(fusar_archive_path)
  processing_open_sarship(src_folder=opensarship_extract_path, dest_folder=open_sar_path)

  for folder in sub_folders:
      # Example usage:
      folder1 = fusar_path + folder
      folder2 = open_sar_path +folder
      destination_folder = mix_path +folder
      merge_folders(folder1, folder2, destination_folder)


  # Example usage
  split_folder(fusar_path, split_fusar)  # Splits with default parameters (test_size=0.2, shuffle=True)
  split_folder(open_sar_path, split_open_sar_path)  # Splits with default parameters (test_size=0.2, shuffle=True)
  split_folder(mix_path, split_mix_path)  # Splits with default parameters (test_size=0.2, shuffle=True)

if __name__ == "__main__":
   main()