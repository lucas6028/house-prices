from one_hot_encoding import one_hot_encode_sklearn
from target_encoding import target_encoder

def quality_mapping(train, test):
    quality_mapping = {
        'Ex': 5,  # Excellent
        'Gd': 4,  # Good
        'TA': 3,  # Typical/Average
        'Fa': 2,  # Fair
        'Po': 1,   # Poor
        'NA': 0,   # None
        'None': 0   # None
    }

    finished_mapping = {
        'GLQ': 6,  # Good Living Quarters
        'ALQ': 5,  # Average Living Quarters
        'BLQ': 4,  # Below Average Living Quarters
        'Rec': 3,  # Average Rec Room
        'LwQ': 2,  # Low Quality
        'Unf': 1,  # Unfinshed
        'NA': 0,   # None
        'None': 0   # None
    }

    train['ExterQual'] = train['ExterQual'].map(quality_mapping)
    test['ExterQual'] = test['ExterQual'].map(quality_mapping)
    train['ExterCond'] = train['ExterCond'].map(quality_mapping)
    test['ExterCond'] = test['ExterCond'].map(quality_mapping)
    train['BsmtQual'] = train['BsmtQual'].map(quality_mapping)
    test['BsmtQual'] = test['BsmtQual'].map(quality_mapping)
    train['BsmtCond'] = train['BsmtCond'].map(quality_mapping)
    test['BsmtCond'] = test['BsmtCond'].map(quality_mapping)
    train['HeatingQC'] = train['HeatingQC'].map(quality_mapping)
    test['HeatingQC'] = test['HeatingQC'].map(quality_mapping)
    train['KitchenQual'] = train['KitchenQual'].map(quality_mapping)
    test['KitchenQual'] = test['KitchenQual'].map(quality_mapping)
    train['FireplaceQu'] = train['FireplaceQu'].map(quality_mapping)
    test['FireplaceQu'] = test['FireplaceQu'].map(quality_mapping)
    train['GarageQual'] = train['GarageQual'].map(quality_mapping)
    test['GarageQual'] = test['GarageQual'].map(quality_mapping)
    train['GarageCond'] = train['GarageCond'].map(quality_mapping)
    test['GarageCond'] = test['GarageCond'].map(quality_mapping)

    train['BsmtFinType1'] = train['BsmtFinType1'].map(finished_mapping)
    test['BsmtFinType1'] = test['BsmtFinType1'].map(finished_mapping)
    train['BsmtFinType2'] = train['BsmtFinType2'].map(finished_mapping)
    test['BsmtFinType2'] = test['BsmtFinType2'].map(finished_mapping)

    return train, test

def target_encoding(train, test):
    # List of features that benefit from target encoding
    categorical_features = [
        'MSSubClass',      # Type of dwelling
        'MSZoning',        # General zoning classification
        'Neighborhood',    # Physical locations
        'Condition1',      # Proximity to various conditions
        'Condition2',      # Proximity to various conditions (if more than one)
        'BldgType',        # Type of dwelling
        'HouseStyle',      # Style of dwelling
        'RoofStyle',       # Type of roof
        'Exterior1st',     # Exterior covering on house
        'Exterior2nd',     # Exterior covering on house (if more than one)
        'Foundation',      # Type of foundation
        'SaleType',        # Type of sale
        'SaleCondition',   # Condition of sale
        'LotConfig',       # Lot configuration
        'LandSlope',       # Slope of property
        'Functional',      # Home functionality
        'MasVnrType',      # Masonry veneer type
        'GarageType',      # Garage location
    ]

    # Apply target encoding
    encoded_train = target_encoder(
        train,
        features=categorical_features,
        target='SalePrice'
    )

    # For test set, use means from entire training set
    global_mean = train['SalePrice'].mean()
    encoded_test = test.copy()

    for feature in categorical_features:
        means = train.groupby(feature)['SalePrice'].mean()
        encoded_test[f'{feature}_target_enc'] = test[feature].map(means).fillna(global_mean)

    train = encoded_train
    test = encoded_test
    
    return train, test

def onehot_encoding(train, test):
    # Features that benefit from one-hot encoding
    onehot_features = [
        'CentralAir',    # Y/N for central air
        'PavedDrive',    # Y/P/N for paved driveway
        'LotShape',      # Regular, Irregular, etc.
        'LotConfig',     # Inside, Corner, etc.
        'Utilities',     # All public, NoSeWa, etc.
        'LandContour',   # Level, Low, etc.
        'BsmtExposure',  # None, Partial, etc.
        'Electrical',    # SBrkr, FuseA, etc.
        'Heating'        # GasA, GasW, etc.
    ]

    encoded_train = one_hot_encode_sklearn(train, onehot_features)
    encoded_test = one_hot_encode_sklearn(test, onehot_features)
    train = encoded_train
    test = encoded_test

    # Function to handle missing values in categorical features
    def handle_missing_categories(df, features):
        """
        Fill missing values in categorical features with 'Missing'
        """
        df_cleaned = df.copy()
        for feature in features:
            df_cleaned[feature] = df_cleaned[feature].fillna('Missing')
        return df_cleaned
    
    return train, test

def encode_features(train, test):
    # Apply quality mapping
    train, test = quality_mapping(train, test)

    # Apply target encoding
    train, test = target_encoding(train, test)

    # Apply one-hot encoding
    train, test = onehot_encoding(train, test)

    return train, test
