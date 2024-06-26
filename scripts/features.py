import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocess the data: clean, split, and scale."""
    # Drop columns that are not needed for the analysis
    df = df.drop(columns=['UDI', 'Product ID'])
    
    # Handle missing values (if any)
    df = df.dropna()
    
    # Separate features and target variable
    X = df.drop(columns=['Machine failure'])
    y = df['Machine failure']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def select_features(X_train, y_train, X_test, n_components=10):
    """Perform feature selection using PLS regression."""
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)
    
    model = SelectFromModel(pls, prefit=True)
    X_train_selected = model.transform(X_train)
    X_test_selected = model.transform(X_test)
    
    return X_train_selected, X_test_selected, model.get_support()

def plot_feature_importance(pls, feature_names):
    """Plot the importance of selected features."""
    plt.figure(figsize=(10, 8))
    sns.barplot(x=pls.coef_[:, 0], y=feature_names)
    plt.title('Feature Importance based on PLS Regression')
    plt.xlabel('PLS Coefficient')
    plt.ylabel('Feature')
    plt.show()

if __name__ == "__main__":
    # Filepath to the dataset
    filepath = '../data/ai4i2020.csv'
    
    # Load and preprocess data
    data = load_data(filepath)
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(data)
    
    # Select features
    X_train_selected, X_test_selected, selected_features_mask = select_features(X_train, y_train, X_test)
    selected_features = feature_names[selected_features_mask]
    
    # Save the selected features data
    np.savez('../data/selected_features_data.npz', X_train=X_train_selected, X_test=X_test_selected, y_train=y_train, y_test=y_test)
    
    # Plot feature importance
    pls = PLSRegression(n_components=10)
    pls.fit(X_train, y_train)
    plot_feature_importance(pls, selected_features)
    
    print("Feature selection completed and data saved to '../data/selected_features_data.npz'")
