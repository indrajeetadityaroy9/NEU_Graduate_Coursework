import pandas as pd
import numpy as np


# Function to load data and select features from a CSV file
def load_and_select_features(filename, features):
    df = pd.read_csv(filename, delimiter=";")
    x = df[list(features)].values
    y = df['quality'].values
    return x, y


# Function to compute and display errors for a given feature set
def compute_and_display_errors(y, y_pred, feature_set_name):
    errors = np.sum(y != y_pred)
    error_prob = errors / len(y)
    print(f"Errors for {feature_set_name}: {errors}")
    print(f"Error probability for {feature_set_name}: {error_prob}")


# Function to load training and test data from files
def load_data(y_test_file, x_test_file, y_train_file, x_train_file):
    y_test = pd.read_csv(y_test_file, header=None).values.ravel()
    x_test = pd.read_csv(x_test_file, delim_whitespace=True, header=None).values
    y_train = pd.read_csv(y_train_file, header=None).values.ravel()
    x_train = pd.read_csv(x_train_file, delim_whitespace=True, header=None).values
    x = np.vstack((x_train, x_test))
    y = np.hstack((y_train, y_test))
    return x, y
