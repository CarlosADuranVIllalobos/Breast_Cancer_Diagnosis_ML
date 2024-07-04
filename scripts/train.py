"""
train.py
Author: Carlos A. Duran Villalobos

This script trains, tunes, and evaluates machine learning models for the Wisconsin Breast Cancer dataset.
It includes functions to load preprocessed data, add noise, tune hyperparameters, train models, 
evaluate models, and save the trained models and evaluation metrics. The models include 
Random Forest, Gradient Boosting, PLS Regression, and Neural Network.
"""

import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, make_scorer
from imblearn.over_sampling import SMOTE
import joblib

def load_features_data(filepath):
    """Load preprocessed data from a .npz file.
    
    Args:
        filepath (str): Path to the preprocessed data file.
    
    Returns:
        tuple: Training and testing feature matrices, training and testing labels, and selected features.
    """
    data = np.load(filepath, allow_pickle=True)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    selected_features = data['selected_features']
    return X_train, X_test, y_train, y_test, selected_features

def add_noise_to_data(X, y, noise_level=0.00, label_noise=0.00):
    """Add noise to the data.
    
    Args:
        X (ndarray): Feature matrix.
        y (ndarray): Target vector.
        noise_level (float): Standard deviation of Gaussian noise added to features.
        label_noise (float): Proportion of labels to flip.
    
    Returns:
        tuple: Noisy feature matrix and target vector.
    """
    np.random.seed(66)
    
    # Add Gaussian noise to features
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise
    
    # Introduce label noise
    n_samples = len(y)
    n_noisy_labels = int(label_noise * n_samples)
    noisy_indices = np.random.choice(n_samples, n_noisy_labels, replace=False)
    y_noisy = y.copy()
    y_noisy[noisy_indices] = 1 - y_noisy[noisy_indices]  # Flip the labels
    
    return X_noisy, y_noisy

def binary_recall(y_true, y_pred, threshold=0.5):
    """Convert predictions to binary and calculate recall.
    
    Args:
        y_true (ndarray): True labels.
        y_pred (ndarray): Predicted labels or probabilities.
        threshold (float): Threshold for converting probabilities to binary labels.
    
    Returns:
        float: Recall score.
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    return recall_score(y_true, y_pred_binary)

binary_scorer = make_scorer(binary_recall)

def tune_model(model, param_grid, X_train, y_train):
    """Tune hyperparameters using GridSearchCV with custom scoring.
    
    Args:
        model: Machine learning model to tune.
        param_grid (dict): Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
        X_train (ndarray): Training feature matrix.
        y_train (ndarray): Training labels.
    
    Returns:
        model: Best estimator found by GridSearchCV.
    """
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=binary_scorer, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_model(model, X_train, y_train):
    """Train the machine learning model.
    
    Args:
        model: Machine learning model to train.
        X_train (ndarray): Training feature matrix.
        y_train (ndarray): Training labels.
    
    Returns:
        model: Trained model.
    """
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name, is_regression=False, threshold=0.5):
    """Evaluate the trained model and save predictions in a global dictionary.
    
    Args:
        model: Trained machine learning model.
        X_test (ndarray): Testing feature matrix.
        y_test (ndarray): Testing labels.
        model_name (str): Name of the model.
        is_regression (bool): Whether the model is a regression model.
        threshold (float): Threshold for converting probabilities to binary labels.
    
    Returns:
        dict: Dictionary with evaluation metrics.
    """
    if is_regression:
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred >= threshold).astype(int)  # Convert continuous to binary
        y_pred_proba = y_pred  # For regression, use the continuous predictions as probabilities
    else:
        y_pred = model.predict(X_test)
        y_pred_binary = y_pred
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.decision_function(X_test) if hasattr(model, "decision_function") else y_pred_binary
    
    # Ensure y_test is binary
    y_test_binary = (y_test >= threshold).astype(int)
  
    metrics = {
        'accuracy': accuracy_score(y_test_binary, y_pred_binary),
        'precision': precision_score(y_test_binary, y_pred_binary),
        'recall': recall_score(y_test_binary, y_pred_binary),
        'f1_score': f1_score(y_test_binary, y_pred_binary),
        'roc_auc': roc_auc_score(y_test_binary, y_pred_proba),  # Use probabilities for ROC AUC
        'confusion_matrix': confusion_matrix(y_test_binary, y_pred_binary)
    }
    
    return metrics

def save_model(model, filepath):
    """Save the trained model to a file.
    
    Args:
        model: Trained machine learning model.
        filepath (str): Path to the file where the model will be saved.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Create the directory if it doesn't exist
    joblib.dump(model, filepath)

def get_models_and_params(X_train):
    """Define models and hyperparameter grids.
    
    Args:
        X_train (ndarray): Training feature matrix.
    
    Returns:
        tuple: Dictionary of models and dictionary of parameter grids.
    """
    n_components_upper_bound = min(X_train.shape[0], X_train.shape[1])
    
    models = {
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'PLS Regression': PLSRegression(),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', random_state=42)
    }
    
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', {0: 1, 1: 3}, {0: 1, 1: 5}]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
        },
        'PLS Regression': {
            'n_components': [i for i in range(2, n_components_upper_bound + 1)]
        },
        'Neural Network': {
            'hidden_layer_sizes': [(10,), (20,), (50,), (10,10), (20,20)],
            'max_iter': [500, 1000],  # Increased max_iter to allow more time for convergence
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01, 0.1]
        }
    }
    return models, param_grids

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train, tune, and evaluate models.
    
    Args:
        X_train (ndarray): Training feature matrix.
        X_test (ndarray): Testing feature matrix.
        y_train (ndarray): Training labels.
        y_test (ndarray): Testing labels.
    
    Returns:
        dict: Dictionary with evaluation metrics for each model.
    """
    models, param_grids = get_models_and_params(X_train)
    results = {}
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Add noise to the data
    X_train_balanced, y_train_balanced = add_noise_to_data(X_train_balanced, y_train_balanced)
    X_test, y_test = add_noise_to_data(X_test, y_test)
    
    # Check distribution of target variable
    print("y_train distribution after SMOTE and noise:", np.bincount(y_train_balanced))
    print("y_test distribution:", np.bincount(y_test))
    
    for name, model in models.items():
        print(f"Tuning {name}...")
        tuned_model = tune_model(model, param_grids[name], X_train_balanced, y_train_balanced)
        print(f"Training {name}...")
        trained_model = train_model(tuned_model, X_train_balanced, y_train_balanced)
        print(f"Evaluating {name}...")
        is_regression = True if name == 'PLS Regression' else False
        metrics = evaluate_model(trained_model, X_test, y_test, name, is_regression=is_regression)
        results[name] = metrics
        save_model(trained_model, f'../models/{name.lower().replace(" ", "_")}_model.pkl')
    return results

if __name__ == "__main__":
    # Load preprocessed data
    X_train, X_test, y_train, y_test, selected_features = load_features_data('../data/selected_features_data.npz')
    
    # Train, tune, and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Save evaluation metrics
    results_df = pd.DataFrame(results).T
    os.makedirs('../results', exist_ok=True)  # Create the directory if it doesn't exist
    results_df.to_csv('../results/evaluation_metrics.csv', index=True)
    
    print("Model training, tuning, and evaluation completed. Results saved.")