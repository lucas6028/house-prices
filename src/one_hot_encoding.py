import pandas as pd

from sklearn.preprocessing import OneHotEncoder

def onehot_encoder(df, features, handle_unknown='ignore'):
    """
    Performs one-hot encoding for specified categorical features

    Parameters:
    -----------
    df : DataFrame
        Input dataset
    features : list
        List of categorical features to encode
    handle_unknown : str
        Strategy for handling unknown categories in test set

    Returns:
    --------
    DataFrame with encoded features, encoder object
    """
    # Initialize encoder with updated parameters
    encoder = OneHotEncoder(sparse_output=False, handle_unknown=handle_unknown)

    # Fit and transform the features
    encoded_array = encoder.fit_transform(df[features])

    # Get feature names
    feature_names = []
    for i, feature in enumerate(features):
        categories = encoder.categories_[i]
        for category in categories:
            feature_names.append(f"{feature}_{category}")

    # Create DataFrame with encoded features
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=feature_names,
        index=df.index
    )

    # Return encoded features and the encoder for later use
    return encoded_df, encoder