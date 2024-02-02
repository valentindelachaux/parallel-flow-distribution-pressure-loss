import os 
import numpy as np
import pandas as pd

import shutil

def create_folder(folder_path):
    """
    Creates a folder at the specified path.

    :param folder_path: Path where the folder will be created.
    """
    try:
        # Create the folder if it does not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder created at {folder_path}")
        else:
            print(f"Folder already exists at {folder_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def delete_folder(folder_path):
    """
    Deletes the folder at the specified path.

    :param folder_path: Path of the folder to be deleted.
    """
    try:
        # Delete the folder if it exists
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Folder deleted at {folder_path}")
        else:
            print(f"No folder found at {folder_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Function to replace placeholders
def replace_placeholders(line, df, row_index):
    if 'BC_ID' in line:
        line = line.replace('BC_ID', df.at[row_index, 'BC_ID'])
    if 'REPORT_NAME' in line:
        line = line.replace('REPORT_NAME', df.at[row_index, 'REPORT_NAME'])
    return line

# Reading the input file and writing to the output file
def process_file(input_file, output_file, df, row_index):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        for line in lines[:]:
            modified_line = replace_placeholders(line, df, row_index)
            file.write(modified_line)

def concatenate_txt_files(folder_path, output_file):
    """
    Concatenates all .txt files in the specified folder and writes the result to the output file.
    
    :param folder_path: Path to the folder containing .txt files.
    :param output_file: Path to the output file where concatenated content will be saved.
    """
    all_texts = ["/file/set-tui-version \"22.2\""]

    # List all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            
            # Read each file and append its content
            with open(file_path, 'r') as file:
                all_texts.append(file.read())

    # Concatenate all texts
    concatenated_text = "\n".join(all_texts)

    # Write the concatenated text to the output file
    with open(output_file, 'w') as file:
        file.write(concatenated_text)

import pandas as pd

def create_dataframe_from_complex_text_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    column_names = lines[0].strip().split('\t')
    data = {col: [] for col in column_names}

    for line in lines[1:]:
        values = line.strip().split('\t')
        
        # Ensure that the number of values matches the number of columns
        if len(values) == len(column_names):
            for col, value in zip(column_names, values):
                data[col].append(value)
        else:
            print(f"Warning: Line skipped due to mismatched number of columns: {line}")

    df = pd.DataFrame(data)
    return df

