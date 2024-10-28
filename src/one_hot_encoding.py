import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def one_hot_encode_sklearn(df, features):
    """
    Function to apply one-hot encoding using sklearn's OneHotEncoder.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    features (list): A list of column names in the DataFrame to apply one-hot encoding.
    
    Returns:
    pd.DataFrame: A DataFrame with the specified features one-hot encoded.
    """
    # Initialize OneHotEncoder
    ohe = OneHotEncoder(sparse_output=False, drop=None)  # Use drop='first' to drop the first category if needed
    
    # Fit and transform only the specified features
    df_to_encode = df[features]
    
    # Perform one-hot encoding and return it as a DataFrame
    ohe_encoded = ohe.fit_transform(df_to_encode)
    
    # Create a DataFrame with encoded feature names
    encoded_columns = ohe.get_feature_names_out(features)
    df_encoded = pd.DataFrame(ohe_encoded, columns=encoded_columns, index=df.index)
    
    # Drop original features and concatenate the encoded features
    df_final = pd.concat([df.drop(features, axis=1), df_encoded], axis=1)
    
    return df_final