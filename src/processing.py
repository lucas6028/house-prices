"""
Module: processing
This module provides functions to preprocess the house prices dataset for a Kaggle competition. 
It includes loading the data, cleaning it, encoding features, and saving the processed data.
Functions:
    - clean_data: Cleans the dataset by handling missing values, outliers, etc.
    - encode_features: Encodes categorical features into numerical values.
Usage:
    The script reads the training and test datasets, applies preprocessing steps, 
    and saves the processed data to CSV files.
"""
import pandas as pd
from data_cleaning import clean_data
from encoding import encode_features

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Display the first few rows of each DataFrame to verify
print(train.head())
print(test.head())

# Apply preprocessing steps
train = clean_data(train)
test = clean_data(test)

train, test = encode_features(train, test)

# Save processed data
train.to_csv('data/train_processed.csv', index=False)
test.to_csv('data/test_processed.csv', index=False)
