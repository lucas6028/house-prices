"""
Module for cleaning data.
"""

import pandas as pd
import numpy as np

def clean_data(df):
    """
    Cleans the input DataFrame by handling missing values, converting data types,
    removing duplicates, and removing outliers.
    
    Args:
        df (pd.DataFrame): The input DataFrame to be cleaned.
    
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df = df.copy()
    df = fill_missing_values(df)
    df = convert_data_types(df)
    df = remove_duplicates(df)
    df = remove_outliers(df)
    assert df.isnull().sum().sum() == 0
    save_cleaned_data(df)
    return df

def fill_missing_values(df):
    """
    Fill missing values in the DataFrame with appropriate values.

    Parameters:
    df (pandas.DataFrame): The input DataFrame with missing values.

    Returns:
    pandas.DataFrame: The DataFrame with missing values filled.

    The function performs the following operations:
    - Drops columns that are not needed: 'PoolQC', 'MiscFeature', 'Alley', 'Fence'.
    - Fills missing values in 'MasVnrType' and 'FireplaceQu' with their respective mode.
    - Fills missing values in 'LotFrontage' with the median value grouped by 'Neighborhood'.
    - Fills missing values in 'SaleType' with 'Oth'.
    - Fills missing values in garage-related columns 
        with 'None' for categorical and 0 for numerical columns.
    - Fills missing values in 'GarageYrBlt' with 1801.
    - Fills missing values in basement-related columns 
        with 'None' for categorical and 0 for numerical columns.
    - Fills missing values in 'MasVnrArea' with 0.
    - Fills missing values in 'Electrical' with its mode.
    - Fills missing values in 'MSZoning' with its mode.
    - Fills missing values in 'Functional' with its mode.
    - Fills missing values in 'Utilities' with its mode.
    - Fills missing values in 'KitchenQual' with its mode.
    - Fills missing values in 'Exterior2nd' and 'Exterior1st' with their respective mode.
    """
    none_fill_columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
    df = df.drop(columns=none_fill_columns)
    df['MasVnrType'] = df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
    df['FireplaceQu'] = df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    df['SaleType'] = df['SaleType'].fillna('Oth')
    garage_cols = ['GarageCond', 'GarageType', 'GarageFinish', 'GarageQual']
    garage_num_cols = ['GarageCars', 'GarageArea']
    df[garage_cols] = df[garage_cols].fillna('None')
    df[garage_num_cols] = df[garage_num_cols].fillna(0)
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(1801)
    bsmt_cols = [
        'BsmtFinType2', 
        'BsmtExposure', 
        'BsmtQual', 
        'BsmtCond', 
        'BsmtFinType1'
    ]
    bsmt_num_cols = [
        'BsmtFinSF1', 
        'BsmtFinSF2',
        'BsmtUnfSF',
        'TotalBsmtSF', 
        'BsmtFullBath', 
        'BsmtHalfBath'
    ]
    df[bsmt_cols] = df[bsmt_cols].fillna('None')
    df[bsmt_num_cols] = df[bsmt_num_cols].fillna(0)
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    df['SaleType'] = df['SaleType'].fillna('Oth')
    df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
    df['Functional'] = df['Functional'].fillna(df['Functional'].mode()[0])
    df['Utilities'] = df['Utilities'].fillna(df['Utilities'].mode()[0])
    df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
    df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
    df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
    return df

def convert_data_types(df):
    """
    Convert data types of specific columns in the DataFrame.

    This function converts the data types of various columns 
    in the input DataFrame to appropriate types.
    Numeric columns are converted to numeric types, 
    date columns are converted to datetime, and some columns
    are converted to string types.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data to be converted.

    Returns:
    pandas.DataFrame: The DataFrame with converted data types.

    Columns converted:
    - 'LotFrontage': to numeric
    - 'MasVnrArea': to numeric
    - 'BsmtFinSF1': to numeric
    - 'BsmtFinSF2': to numeric
    - 'BsmtUnfSF': to numeric
    - 'TotalBsmtSF': to numeric
    - 'BsmtFullBath': to numeric
    - 'BsmtHalfBath': to numeric
    - 'GarageCars': to numeric
    - 'GarageArea': to numeric
    - 'OverallQual': to numeric
    - 'OverallCond': to numeric
    - 'YearBuilt': to datetime (format: '%Y')
    - 'YearRemodAdd': to datetime (format: '%Y')
    - 'GarageYrBlt': to datetime (format: '%Y')
    - 'YrSold': to datetime (format: '%Y')
    - 'MSSubClass': to string
    """
    df['LotFrontage'] = pd.to_numeric(df['LotFrontage'], errors='coerce')
    df['MasVnrArea'] = pd.to_numeric(df['MasVnrArea'], errors='coerce')
    df['BsmtFinSF1'] = pd.to_numeric(df['BsmtFinSF1'], errors='coerce')
    df['BsmtFinSF2'] = pd.to_numeric(df['BsmtFinSF2'], errors='coerce')
    df['BsmtUnfSF'] = pd.to_numeric(df['BsmtUnfSF'], errors='coerce')
    df['TotalBsmtSF'] = pd.to_numeric(df['TotalBsmtSF'], errors='coerce')
    df['BsmtFullBath'] = pd.to_numeric(df['BsmtFullBath'], errors='coerce')
    df['BsmtHalfBath'] = pd.to_numeric(df['BsmtHalfBath'], errors='coerce')
    df['GarageCars'] = pd.to_numeric(df['GarageCars'], errors='coerce')
    df['GarageArea'] = pd.to_numeric(df['GarageArea'], errors='coerce')
    df['OverallQual'] = pd.to_numeric(df['OverallQual'], errors='coerce')
    df['OverallCond'] = pd.to_numeric(df['OverallCond'], errors='coerce')
    df['YearBuilt'] = pd.to_datetime(df['YearBuilt'], format='%Y', errors='coerce')
    df['YearRemodAdd'] = pd.to_datetime(df['YearRemodAdd'], format='%Y', errors='coerce')
    df['GarageYrBlt'] = pd.to_datetime(df['GarageYrBlt'], format='%Y', errors='coerce')
    df['YrSold'] = pd.to_datetime(df['YrSold'], format='%Y', errors='coerce')
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    return df

def remove_duplicates(df):
    """
    Remove duplicate rows from a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame from which to remove duplicate rows.

    Returns:
    pandas.DataFrame: A DataFrame with duplicate rows removed.
    """
    return df.drop_duplicates()

def remove_outliers(df):
    """
    Remove outliers from the DataFrame based on predefined thresholds for various columns.

    This function performs the following operations:
    - Applies a log transformation to the 'SalePrice' column if it exists.
    - Removes rows where 'LotFrontage' is 300 or more.
    - Removes rows where 'LotArea' is 100,000 or more.
    - Removes rows where 'TotalBsmtSF' is 3,000 or more.
    - Removes rows where '1stFlrSF' is 3,000 or more.
    - Removes rows where '2ndFlrSF' is 1,500 or more.
    - Removes rows where 'GrLivArea' is 4,000 or more.
    - Removes rows where 'GarageArea' is 1,200 or more.
    - Removes rows where 'WoodDeckSF' is 600 or more.
    - Removes rows where 'OpenPorchSF' is 300 or more.
    - Removes rows where 'MiscVal' is 5,000 or more.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing house price data.

    Returns:
    pandas.DataFrame: The DataFrame with outliers removed.
    """
    if 'SalePrice' in df.columns:
        df['SalePrice_log'] = np.log(df['SalePrice'])  # Log transformation
    df = df[df['LotFrontage'] < 300]
    df = df[df['LotArea'] < 100000]
    df = df[df['TotalBsmtSF'] < 3000]
    df = df[df['1stFlrSF'] < 3000]
    df = df[df['2ndFlrSF'] < 1500]
    df = df[df['GrLivArea'] < 4000]
    df = df[df['GarageArea'] < 1200]
    df = df[df['WoodDeckSF'] < 600]
    df = df[df['OpenPorchSF'] < 300]
    df = df[df['MiscVal'] < 5000]
    return df

def save_cleaned_data(df):
    """
    Save the cleaned DataFrame to a CSV file.

    This function checks if the 'SalePrice_log' column exists in the DataFrame.
    If it does, the DataFrame is saved to 'data/train_cleaned.csv'.
    Otherwise, it is saved to 'data/test_cleaned.csv'.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be saved.

    Returns:
    None
    """
    if 'SalePrice_log' in df.columns:
        df.to_csv('data/train_cleaned.csv', index=False)
    else:
        df.to_csv('data/test_cleaned.csv', index=False)
