import pandas as pd 
import numpy as np 
import os, glob
import argparse
import importlib
from tqdm import tqdm 

def main():
    # Get the current working directory
    current_dir = os.getcwd()

    # Paths for training and testing data
    tr_path = os.path.join(current_dir, "train")
    total_data = []

    # Iterate over the sorted list of files in the 'train' directory
    for file_name in tqdm(sorted(glob.glob(tr_path + "/*"))):
        print(f"Processing file: {file_name}")

        # Read the CSV file with proper handling for encoding and bad lines
        try:
            data = pd.read_csv(file_name, sep='\t', encoding='ISO-8859-1')
            print(f"File {file_name} read successfully.")

        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            continue

        # Add the file name as a new column to the data
        data["file_name"] = file_name.split(os.sep)[-1].split('.')[0]
        total_data.append(data)

    # Concatenate all the data into a single DataFrame
    if total_data:
        total_data = pd.concat(total_data, ignore_index=True)
        print("All data concatenated successfully.")

        # Save the combined DataFrame as a Parquet file
        os.makedirs(tr_path, exist_ok=True)
        total_data.to_parquet(os.path.join(tr_path, "train.parquet"), index=False)
        print("Training data saved successfully.")

    # Process the test data
    te_path = os.path.join(current_dir, "test")
    os.makedirs(te_path, exist_ok=True)

    try:
        test_data = pd.read_csv(os.path.join(te_path, "000000000000.csv"), sep='\t', encoding='ISO-8859-1')
        test_data.to_parquet(os.path.join(te_path, "test.parquet"), index=False)
        print("Test data saved successfully.")
    except Exception as e:
        print(f"Error reading test data: {e}")

    print("Data preprocessing completed successfully.")

def parse_arguments():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Recsys challenge 2023")
    #parser.add_argument('--config', required=True, help="Config filename")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()