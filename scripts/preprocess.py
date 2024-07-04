"""
preprocess.py
Author: Carlos A. Duran Villalobos

This script preprocesses the Wisconsin Breast Cancer dataset by cleaning the data,
encoding categorical variables, scaling features, and splitting the data into training
and testing sets. The preprocessed data is then saved for later use.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    """Clean the dataset by handling missing values and removing unnecessary columns.
    
    Args:
        df (DataFrame): The raw dataframe to be cleaned.
    
    Returns:
        DataFrame: The cleaned dataframe.
    """
    # Drop columns that are not needed for the analysis
    df = df.drop(columns=['id'])
    df = df.drop(columns=['Unnamed: 32'])
    
    # Handle missing values (if any)
    df = df.dropna()
    
    return df

def encode_categorical(df):
    """Encode non-numeric columns.
    
    Args:
        df (DataFrame): The dataframe with categorical columns to be encoded.
    
    Returns:
        tuple: DataFrame with encoded columns and dictionary of label encoders.
    """
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders

def feature_scaling(X_train, X_test):
    """Scale the features using StandardScaler.
    
    Args:
        X_train (ndarray): Training feature matrix.
        X_test (ndarray): Testing feature matrix.
    
    Returns:
        tuple: Scaled training and testing feature matrices.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def preprocess_data(filepath):
    """Load, clean, and preprocess the data.
    
    Args:
        filepath (str): Path to the dataset CSV file.
    
    Returns:
        tuple: Scaled training and testing feature matrices, training and testing labels, and feature names.
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Clean data
    df = clean_data(df)
    
    # Encode non-numeric columns
    df, label_encoders = encode_categorical(df)
    
    # Separate features and target variable
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    X_train_scaled, X_test_scaled = feature_scaling(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

if __name__ == "__main__":
    # Filepath to the dataset
   filepath = '../data/breast_cancer_wisconsin.csv'
   
   # Preprocess the data
   X_train, X_test, y_train, y_test, feature_names = preprocess_data(filepath)
   
   # Save preprocessed data for later use
   np.savez('../data/preprocessed_data.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, feature_names=feature_names)
   
   print("Data preprocessing completed and saved to '../data/preprocessed_data.npz'")