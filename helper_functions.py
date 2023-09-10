"""The purpose of this script is to store the helper functions like loading and manipulating datasets for the machine
learning aspect in assignment 2 of SCIE3100.
"""
#######################################################################################################################
# Setting up the environment #
#######################################################################################################################
import pandas as pd


def checkFileReading():
    """Tries to read a file we expect to be in our current working directory"""
    fileHandle = open('designNotes.txt', 'r')
    for line in fileHandle:
        print(line)


def load_data(file_path):
    # Load the data using Pandas (adjust the delimiter if needed)
    data = pd.read_csv(file_path, index_col=0)

    # Extract features (all columns except the label column)
    X = data.drop(columns=['Label'])  # Replace 'Label' with the actual label column name

    # Extract labels (assuming the label column name is 'Label')
    y = data['Label']

    return X, y
