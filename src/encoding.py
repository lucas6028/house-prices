from one_hot_encoding import one_hot_encode_sklearn
from target_encoding import target_encoder

def quality_mapping(train, test):
    """
    Maps quality and condition features to numerical values.
    
    Args:
        train (pd.DataFrame): Training dataset.
        test (pd.DataFrame): Testing dataset.
    
    Returns:
        pd.DataFrame, pd.DataFrame: Transformed training and testing datasets.
    """
    quality_mapping = {
        'Ex': 5,  # Excellent
        'Gd': 4,  # Good
        'TA': 3,  # Typical/Average
        'Fa': 2,  # Fair
        'Po': 1,  # Poor
        'NA': 0,  # None
        'None': 0  # None
    }

    finished_mapping = {
        'GLQ': 6,  # Good Living Quarters
        'ALQ': 5,  # Average Living Quarters
        'BLQ': 4,  # Below Average Living Quarters
        'Rec': 3,  # Average Rec Room
        'LwQ': 2,  # Low Quality
        'Unf': 1,  # Unfinished
        'NA': 0,  # None
        'None': 0  # None
    }

    for feature in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
                    'HeatingQC', 'KitchenQual', 'FireplaceQu', 
                    'GarageQual', 'GarageCond']:
        train[feature] = train[feature].map(quality_mapping)
        test[feature] = test[feature].map(quality_mapping)

    for feature in ['BsmtFinType1', 'BsmtFinType2']:
        train[feature] = train[feature].map(finished_mapping)
        test[feature] = test[feature].map(finished_mapping)

    return train, test

def target_encoding(train, test):
    """
    Applies target encoding to categorical features.
    
    Args:
        train (pd.DataFrame): Training dataset.
        test (pd.DataFrame): Testing dataset.
    
    Returns:
        pd.DataFrame, pd.DataFrame: Transformed training and testing datasets.
    """
    categorical_features = [
        'MSSubClass', 'MSZoning', 'Neighborhood', 'Condition1', 
        'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 
        'Exterior1st', 'Exterior2nd', 'Foundation', 'SaleType', 
        'SaleCondition', 'LotConfig', 'LandSlope', 'Functional', 
        'MasVnrType', 'GarageType'
    ]

    encoded_train = target_encoder(
        train,
        features=categorical_features,
        target='SalePrice'
    )

    global_mean = train['SalePrice'].mean()
    encoded_test = test.copy()

    for feature in categorical_features:
        means = train.groupby(feature)['SalePrice'].mean()
        encoded_test[f'{feature}_target_enc'] = test[feature].map(means).fillna(global_mean)

    return encoded_train, encoded_test

def onehot_encoding(train, test):
    """
    Applies one-hot encoding to categorical features.
    
    Args:
        train (pd.DataFrame): Training dataset.
        test (pd.DataFrame): Testing dataset.
    
    Returns:
        pd.DataFrame, pd.DataFrame: Transformed training and testing datasets.
    """
    onehot_features = [
        'CentralAir', 'PavedDrive', 'LotShape', 'LotConfig', 
        'Utilities', 'LandContour', 'BsmtExposure', 'Electrical', 
        'Heating'
    ]

    encoded_train = one_hot_encode_sklearn(train, onehot_features)
    encoded_test = one_hot_encode_sklearn(test, onehot_features)

    return encoded_train, encoded_test

def encode_features(train, test):
    """
    Encodes features using quality mapping, target encoding, and one-hot encoding.
    
    Args:
        train (pd.DataFrame): Training dataset.
        test (pd.DataFrame): Testing dataset.
    
    Returns:
        pd.DataFrame, pd.DataFrame: Transformed training and testing datasets.
    """
    train, test = quality_mapping(train, test)
    train, test = target_encoding(train, test)
    train, test = onehot_encoding(train, test)

    return train, test
    