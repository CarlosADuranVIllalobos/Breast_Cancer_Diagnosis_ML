import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

def load_preprocessed_data(filepath):
    """Load preprocessed data from a .npz file."""
    data = np.load(filepath)
    X_test = data['X_test']
    y_test = data['y_test']
    return X_test, y_test

def load_model(filepath):
    """Load a trained model from a file."""
    return joblib.load(filepath)

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return metrics

def plot_confusion_matrix(cm, title='Confusion Matrix'):
    """Plot confusion matrix."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Failure', 'Failure'], yticklabels=['No Failure', 'Failure'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

def plot_roc_curves(models, X_test, y_test):
    """Plot ROC curves for multiple models."""
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='best')
    plt.show()

def evaluate_and_plot(models, X_test, y_test):
    """Evaluate models, plot confusion matrices and ROC curves."""
    results = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        plot_confusion_matrix(metrics['confusion_matrix'], title=f'{name} Confusion Matrix')
    plot_roc_curves(models, X_test, y_test)
    return results

if __name__ == "__main__":
    # Load preprocessed data
    X_test, y_test = load_preprocessed_data('../data/selected_features_data.npz')

    # Load models
    model_names = ['random_forest_model.pkl', 'gradient_boosting_model.pkl', 'pls_regression_model.pkl', 'neural_network_model.pkl']
    models = {name.split('_model.pkl')[0].replace('_', ' ').title(): load_model(f'../models/{name}') for name in model_names}
    
    # Evaluate models and plot results
    results = evaluate_and_plot(models, X_test, y_test)
    
    # Save evaluation metrics
    results_df = pd.DataFrame(results).T
    results_df.to_csv('../results/evaluation_metrics.csv', index=True)
    print("Model evaluation completed. Results saved.")
