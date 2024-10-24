import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

def clean_data(df):
    # Check data shape and types
    print(f"Data shape: {df.shape}")
    print(df.dtypes.value_counts())

    # Summary statistics
    print(df.describe())

    # Missing values per column
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (missing / len(df)) * 100
    missing_data = pd.DataFrame({'Missing Values': missing, 'Percentage': missing_percent})
    print(missing_data[missing_data['Missing Values'] > 0])

    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Fill missing values
    none_fill_columns = ['PoolQC', 'MiscFeature', 'Alley', 'MasVnrType', 'FireplaceQu', 'Fence', 'KitchenQual', 'MSZoning', 'Utilities', 'Functional']
    df[none_fill_columns] = df[none_fill_columns].fillna('None')

    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    df['SaleType'] = df['SaleType'].fillna('Oth')

    garage_cols = ['GarageCond', 'GarageType', 'GarageFinish', 'GarageQual']
    df[garage_cols] = df[garage_cols].fillna('None')
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

    bsmt_cols = ['BsmtFinType2', 'BsmtExposure', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
    df[bsmt_cols] = df[bsmt_cols].fillna('None')
    df['BsmtFullBath'] = df['BsmtFullBath'].fillna(0)
    df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(0)

    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

    # Handling Outliers and Duplicates
    sns.boxplot(x=df['GrLivArea'])
    plt.show()

    df['zscore'] = zscore(df['GrLivArea'])
    df = df[(df['zscore'] > -3) & (df['zscore'] < 3)]
    df = df[df['GrLivArea'] < 4500]

    duplicates = df.duplicated()
    print(f"Number of duplicates: {duplicates.sum()}")
    df = df.drop_duplicates()

    # Convert data types
    df['LotFrontage'] = pd.to_numeric(df['LotFrontage'], errors='coerce')
    df['MasVnrArea'] = pd.to_numeric(df['MasVnrArea'], errors='coerce').astype(int)

    date_cols = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold', 'MoSold']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format='%Y', errors='coerce')

    df['MSSubClass'] = df['MSSubClass'].astype(str)
    df['OverallQual'] = df['OverallQual'].astype(str)
    df['OverallCond'] = df['OverallCond'].astype(str)

    return df