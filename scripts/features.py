"""
features.py
Author: Carlos A. Duran Villalobos

This script performs feature selection on the preprocessed Wisconsin Breast Cancer dataset
using PLS regression and ANOVA F-test methods. It includes functions to determine the optimal
number of PLS components, select features, and plot the importance of the selected features.
The selected features data is saved for later use.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import seaborn as sns

def load_preprocessed_data(filepath):
    """Load preprocessed data from a .npz file.
    
    Args:
        filepath (str): Path to the preprocessed data file.
    
    Returns:
        tuple: Training and testing feature matrices, training and testing labels, and feature names.
    """
    data = np.load(filepath, allow_pickle=True)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']
    return X_train, X_test, y_train, y_test, feature_names

def optimal_n_components(X, y, max_components=10, tolerance=0.01):
    """Determine the optimal number of components for PLS regression using cross-validation.
    
    Args:
        X (ndarray): Feature matrix.
        y (ndarray): Target vector.
        max_components (int): Maximum number of components to consider.
        tolerance (float): Tolerance for stopping criterion based on MSE.
    
    Returns:
        int: Optimal number of PLS components.
    """
    mse = []
    for i in range(1, max_components + 1):
        pls = PLSRegression(n_components=i)
        scores = cross_val_score(pls, X, y, cv=KFold(5), scoring='neg_mean_squared_error')
        mse.append(-scores.mean())
    
    # Determine the optimal number of components
    optimal_components = 1
    for i in range(1, len(mse)):
        if (mse[i - 1] - mse[i]) / mse[i - 1] < tolerance:
            optimal_components = i
            break
    else:
        optimal_components = np.argmin(mse) + 1
    
    return optimal_components

def select_features(X_train, y_train, X_test):
    """Perform feature selection using PLS regression with optimal number of components.
    
    Args:
        X_train (ndarray): Training feature matrix.
        y_train (ndarray): Training labels.
        X_test (ndarray): Testing feature matrix.
    
    Returns:
        tuple: Selected training and testing feature matrices, mask of selected features, and fitted PLS model.
    """
    # Determine the optimal number of components
    optimal_components = optimal_n_components(X_train, y_train)
    print(f"Optimal number of components: {optimal_components}")
    
    # Fit PLS with optimal number of components
    pls = PLSRegression(n_components=optimal_components)
    pls.fit(X_train, y_train)
    
    # Perform feature selection
    model = SelectFromModel(pls, prefit=True)
    X_train_selected = model.transform(X_train)
    X_test_selected = model.transform(X_test)
    
    return X_train_selected, X_test_selected, model.get_support(), pls

def plot_feature_importance(pls, selected_features_mask, feature_names):
    """Plot the importance of selected features.
    
    Args:
        pls (PLSRegression): Fitted PLS regression model.
        selected_features_mask (ndarray): Mask of selected features.
        feature_names (Index): Names of all features.
    """
    # Extract coefficients for the selected features
    coef = pls.coef_[0, :]
    selected_coef = coef[selected_features_mask]
    
    selected_features = feature_names[selected_features_mask]
    print(f"Number of coefficients: {len(selected_coef)}")
    print(f"Number of selected features: {len(selected_features)}")
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=selected_coef, y=selected_features)
    plt.title('Feature Importance based on PLS Regression')
    plt.xlabel('PLS Coefficient')
    plt.ylabel('Feature')
    plt.show()

def select_features_anova(X_train, y_train, X_test, k=30):
    """Perform feature selection using ANOVA F-test.
    
    Args:
        X_train (ndarray): Training feature matrix.
        y_train (ndarray): Training labels.
        X_test (ndarray): Testing feature matrix.
        k (int): Number of top features to select.
    
    Returns:
        tuple: Selected training and testing feature matrices, mask of selected features, and fitted ANOVA selector.
    """
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    return X_train_selected, X_test_selected, selector.get_support(), selector

def plot_feature_importance_anova(selector, selected_features_mask, feature_names):
    """Plot the importance of selected features based on ANOVA F-test.
    
    Args:
        selector (SelectKBest): Fitted ANOVA selector.
        selected_features (ndarray): Mask of selected features.
        feature_names (Index): Names of all features.
    """
    selected_features = feature_names[selected_features_mask]
    scores = selector.scores_[selector.get_support()]
    print(f"Number of scores: {len(scores)}")
    print(f"Number of selected features: {len(selected_features)}")
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=scores, y=selected_features)
    plt.title('Feature Importance based on ANOVA F-test')
    plt.xlabel('ANOVA F-score')
    plt.ylabel('Feature')
    plt.show()

if __name__ == "__main__":
    # Filepath to the dataset
    filepath = '../data/preprocessed_data.npz'
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names = load_preprocessed_data(filepath)
    
    # Select features
    X_train_selected, X_test_selected, selected_features_mask, selector = select_features_anova(X_train, y_train, X_test, k=20)
    
    # Save the selected features data
    np.savez('../data/selected_features_data.npz', X_train=X_train_selected, X_test=X_test_selected, y_train=y_train, y_test=y_test , selected_features=feature_names[selected_features_mask])
    
    # Plot feature importance
    plot_feature_importance_anova(selector, selected_features_mask, feature_names)
    
    print("Feature selection completed and data saved to '../data/selected_features_data.npz'")
