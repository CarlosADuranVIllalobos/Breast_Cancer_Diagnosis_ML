import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Clean the dataset by handling missing values and removing unnecessary columns."""
    # Drop columns that are not needed for the analysis
    df = df.drop(columns=['UDI', 'Product ID'])
    
    # Handle missing values (if any)
    df = df.dropna()
    
    return df

def feature_scaling(X_train, X_test):
    """Scale the features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def preprocess_data(filepath):
    """Load, clean, and preprocess the data."""
    # Load data
    df = load_data(filepath)
    
    # Clean data
    df = clean_data(df)
    
    # Separate features and target variable
    X = df.drop(columns=['Machine failure'])
    y = df['Machine failure']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    X_train_scaled, X_test_scaled = feature_scaling(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Filepath to the dataset
    filepath = '../data/ai4i2020.csv'
    
    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(filepath)
    
    # Save preprocessed data for later use
    np.savez('../data/preprocessed_data.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    
    print("Data preprocessing completed and saved to '../data/preprocessed_data.npz'")
