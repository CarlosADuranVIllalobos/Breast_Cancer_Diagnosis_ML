import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

def load_preprocessed_data(filepath):
    """Load preprocessed data from a .npz file."""
    data = np.load(filepath)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    return X_train, X_test, y_train, y_test

def tune_model(model, param_grid, X_train, y_train):
    """Tune hyperparameters using GridSearchCV."""
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_model(model, X_train, y_train):
    """Train the machine learning model."""
    model.fit(X_train, y_train)
    return model

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

def save_model(model, filepath):
    """Save the trained model to a file."""
    joblib.dump(model, filepath)

def get_models_and_params():
    """Define models and hyperparameter grids."""
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'PLS Regression': PLSRegression(),
        'Neural Network': MLPClassifier(random_state=42)
    }
    
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'PLS Regression': {
            'n_components': [5, 10, 15]
        },
        'Neural Network': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'max_iter': [200, 300, 500],
            'alpha': [0.0001, 0.001, 0.01]
        }
    }
    return models, param_grids

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train, tune, and evaluate models."""
    models, param_grids = get_models_and_params()
    results = {}
    for name, model in models.items():
        print(f"Tuning {name}...")
        tuned_model = tune_model(model, param_grids[name], X_train, y_train)
        print(f"Training {name}...")
        trained_model = train_model(tuned_model, X_train, y_train)
        metrics = evaluate_model(trained_model, X_test, y_test)
        results[name] = metrics
        save_model(trained_model, f'../models/{name.lower().replace(" ", "_")}_model.pkl')
    return results

if __name__ == "__main__":
    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_preprocessed_data('../data/selected_features_data.npz')
    
    # Train, tune, and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Save evaluation metrics
    results_df = pd.DataFrame(results).T
    results_df.to_csv('../results/evaluation_metrics.csv', index=True)
    print("Model training, tuning, and evaluation completed. Results saved.")
