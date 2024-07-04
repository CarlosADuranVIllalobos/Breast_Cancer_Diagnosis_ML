"""
evaluate.py
Author: Carlos A. Duran Villalobos

This script evaluates the performance of machine learning models on the Wisconsin Breast Cancer dataset.
It includes functions to load preprocessed data and trained models, plot confusion matrices, 
plot ROC curves, and parse evaluation metrics. The script generates visualizations of the models' 
performance and saves the plots to the results directory.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay

def load_features_data(filepath):
    """Load preprocessed data from a .npz file.
    
    Args:
        filepath (str): Path to the preprocessed data file.
    
    Returns:
        tuple: Testing feature matrix and labels.
    """
    data = np.load(filepath)
    X_test = data['X_test']
    y_test = data['y_test']
    return X_test, y_test

def load_model(filepath):
    """Load a trained model from a file.
    
    Args:
        filepath (str): Path to the trained model file.
    
    Returns:
        model: Trained model.
    """
    from joblib import load
    return load(filepath)

def plot_confusion_matrices(metrics):
    """Plot all confusion matrices in one figure.
    
    Args:
        metrics (DataFrame): DataFrame containing evaluation metrics, including confusion matrices.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, (model_name, cm) in enumerate(metrics['confusion_matrix'].items()):
        # Swap the rows and columns
        cm_swapped = cm[::-1, ::-1]
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_swapped, display_labels=['Malignant', 'Benign'])
        disp.plot(ax=axes[i], cmap='Blues', values_format='d', text_kw={'fontsize': 16}, colorbar=False)
        axes[i].set_title(f'{model_name} Confusion Matrix', fontsize=20)
        axes[i].set_xlabel('Predicted', fontsize=18)
        axes[i].set_ylabel('Actual', fontsize=18)
        
        # Move x-axis labels above the figures
        axes[i].xaxis.set_label_position('top')
        axes[i].xaxis.tick_top()
    
    # Adjust the layout to make room for the dividing lines
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.2)
    
    # Add red lines to divide the quadrants
    line_color = 'red'
    line_style = '-'
    line_width = 2

    # Horizontal line in the middle
    fig.lines.extend([plt.Line2D([0.05, 0.95], [0.52, 0.52], color=line_color, transform=fig.transFigure, figure=fig, linestyle=line_style, linewidth=line_width)])

    # Vertical line in the middle
    fig.lines.extend([plt.Line2D([0.48, 0.48], [0.05, 0.95], color=line_color, transform=fig.transFigure, figure=fig, linestyle=line_style, linewidth=line_width)])

    plt.savefig('../results/confusion_matrices.png')
    plt.show()

def plot_single_confusion_matrix(model_name, cm):
    """Plot a single model's confusion matrix.
    
    Args:
        model_name (str): Name of the model.
        cm (ndarray): Confusion matrix.
    """
    cm_swapped = cm[::-1, ::-1]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_swapped, display_labels=['Malignant', 'Benign'])
    disp.plot(cmap='Blues', values_format='d', text_kw={'fontsize': 16}, colorbar=False)
    plt.title(f'{model_name} Confusion Matrix', fontsize=20)
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('Actual', fontsize=18)
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.tick_top()
    plt.show()

def plot_roc_curves(models, X_test, y_test, metrics):
    """Plot ROC curves for multiple models.
    
    Args:
        models (dict): Dictionary of trained models.
        X_test (ndarray): Testing feature matrix.
        y_test (ndarray): Testing labels.
        metrics (DataFrame): DataFrame containing evaluation metrics, including AUC values.
    """
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:  # For models that do not support predict_proba (like PLSRegression)
            y_pred = model.predict(X_test)
            y_pred_proba = y_pred  # For regression, use the continuous predictions as probabilities
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_value = metrics.loc[name, 'roc_auc']
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_value:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='best')
    plt.savefig('../results/roc_curves.png')
    plt.show()

def parse_confusion_matrix(cm_str):
    """Parse confusion matrix string into a numpy array.
    
    Args:
        cm_str (str): Confusion matrix string.
    
    Returns:
        ndarray: Parsed confusion matrix.
    """
    cm_str = cm_str.replace('[', '').replace(']', '').replace('\n', '')
    cm_list = list(map(int, cm_str.split()))
    cm_array = np.array(cm_list).reshape(2, 2)
    return cm_array

def load_metrics(filepath):
    """Load evaluation metrics from a CSV file.
    
    Args:
        filepath (str): Path to the evaluation metrics CSV file.
    
    Returns:
        DataFrame: DataFrame containing evaluation metrics.
    """
    metrics = pd.read_csv(filepath, index_col=0)
    metrics['confusion_matrix'] = metrics['confusion_matrix'].apply(parse_confusion_matrix)
    return metrics

if __name__ == "__main__":
    # Load preprocessed data
    X_test, y_test = load_features_data('../data/selected_features_data.npz')

    # Load models
    model_names = ['random_forest_model.pkl', 'gradient_boosting_model.pkl', 'pls_regression_model.pkl', 'neural_network_model.pkl']
    models = {
        'Random Forest': load_model('../models/random_forest_model.pkl'),
        'Gradient Boosting': load_model('../models/gradient_boosting_model.pkl'),
        'PLS Regression': load_model('../models/pls_regression_model.pkl'),
        'Neural Network': load_model('../models/neural_network_model.pkl')
    }

    # Load evaluation metrics
    metrics = load_metrics('../results/evaluation_metrics.csv')
    
    # Plot confusion matrices
    plot_confusion_matrices(metrics)
    
    # Plot ROC curves
    plot_roc_curves(models, X_test, y_test, metrics)
    
    # Plot a single confusion matrix for demonstration
    plot_single_confusion_matrix('Random Forest', metrics.loc['Random Forest', 'confusion_matrix'])
    
    print("Model evaluation plots completed.")
