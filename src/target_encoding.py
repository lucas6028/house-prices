"""
Module for performing target encoding with k-fold cross-validation.
"""

import numpy as np
from sklearn.model_selection import KFold

def target_encoder(df, features, target, n_splits=5, alpha=5):
    """
    Performs target encoding with k-fold cross-validation to prevent leakage.

    Parameters:
    -----------
    df : DataFrame
        Input dataset.
    features : list
        List of categorical features to encode.
    target : str
        Name of the target variable.
    n_splits : int
        Number of folds for cross-validation.
    alpha : float
        Smoothing parameter.

    Returns:
    --------
    DataFrame with encoded features.
    """
    df_encoded = df.copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for feature in features:
        # Calculate global mean
        global_mean = df[target].mean()

        # Create a new column for the encoding
        df_encoded[f'{feature}_target_enc'] = np.nan

        # Perform k-fold target encoding
        for train_idx, val_idx in kf.split(df):
            # Get the train and validation sets
            train = df.iloc[train_idx]
            val = df.iloc[val_idx]

            # Calculate means for each category in training set
            means = train.groupby(feature)[target].agg(['mean', 'count'])

            # Apply smoothing
            smoothed_means = (
                means['mean'] * means['count'] + global_mean * alpha
            ) / (means['count'] + alpha)

            # Map the means to validation set
            df_encoded.iloc[val_idx, df_encoded.columns.get_loc(f'{feature}_target_enc')] = \
                val[feature].map(smoothed_means).fillna(global_mean)

    return df_encoded