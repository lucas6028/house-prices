# preprocess.py
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
