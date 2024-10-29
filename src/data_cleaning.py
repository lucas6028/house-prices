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
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Fill missing values
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

    # Remove duplicates
    df = df.drop_duplicates()

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

    # Remove outliers
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

    # Final consistency and missing checking
    assert df.isnull().sum().sum() == 0

    # Save cleaned data
    if 'SalePrice_log' in df.columns:
        df.to_csv('data/train_cleaned.csv', index=False)
    else:
        df.to_csv('data/test_cleaned.csv', index=False)

    return df
