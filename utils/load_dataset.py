import os

import pandas as pd
import tensorflow as tf


# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {"sentence": []}
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            # tf.gfile.GFile.read reads file contents into a string
            # strings with file contents are stored into “sentence” array in data dict
            data["sentence"].append(f.read())

    return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    # Load review text files into Pandas data frames with positive and negative texts
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))

    # Add another column to the dataframes polarity
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0

    # Concatenate the datasets with positive and negative texts
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)
