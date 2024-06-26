# Failure_prediction_ML
![License](https://img.shields.io/badge/license-MIT-brightgreen.svg)
![Python](https://img.shields.io/badge/python-3.7%2B-brightgreen.svg)

This repository demonstrates how to use machine learning models to predict equipment failures in a manufacturing process in Python. It includes data preprocessing, feature engineering, model training, and evaluation.

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The AI4I 2020 Predictive Maintenance Dataset is sourced from the UCI Machine Learning Repository. It contains 10,000 data points with 14 features, including:

- `UID`: Unique identifier
- `productID`: Product quality variant (L, M, H)
- `air temperature [K]`: Air temperature in Kelvin
- `process temperature [K]`: Process temperature in Kelvin
- `rotational speed [rpm]`: Rotational speed in revolutions per minute
- `torque [Nm]`: Torque in Newton-meters
- `tool wear [min]`: Tool wear in minutes
- `machine failure`: Binary label indicating machine failure

Additional failure modes include tool wear failure (TWF), heat dissipation failure (HDF), power failure (PWF), overstrain failure (OSF), and random failures (RNF).

### Citation of the Dataset

AI4I 2020 Predictive Maintenance Dataset. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C.

## Project Structure
Failure_prediction_ML/

```plaintext
Failure_prediction_ML/
├── data/
│   ├── ai4i2020.csv            # Raw dataset
├── notebooks/
│   ├── data_preprocessing.ipynb  # Data preprocessing and EDA
│   ├── feature_selection.ipynb # Feature engineering
│   ├── model_training.ipynb      # Model training and tuning
│   ├── model_evaluation.ipynb    # Model evaluation
├── scripts/
│   ├── preprocess.py            # Data preprocessing script
│   ├── train.py                 # Model training script
│   ├── evaluate.py              # Model evaluation script
├── models/
│   ├── model.pkl                # Trained model
├── results/
│   ├── evaluation_metrics.csv   # Evaluation metrics
│   ├── confusion_matrix.png     # Confusion matrix
├── README.md                    # Project README
```

## Usage

1. **Data Preprocessing**:
   - Execute the `data_preprocessing.ipynb` notebook to clean and preprocess the data.

2. **Feature Engineering**:
   - Run the `feature_engineering.ipynb` notebook to create new features and select the most important ones.

3. **Model Training**:
   - Use the `model_training.ipynb` notebook to train various machine learning models and tune hyperparameters.

4. **Model Evaluation**:
   - Evaluate the performance of the trained models using the `model_evaluation.ipynb` notebook.

To run the scripts from the command line:
python scripts/preprocess.py
python scripts/train.py
python scripts/evaluate.py
## Modeling

The project explores various machine learning models, including:

- Random Forest (RF)
- Gradient Boosting (GB)
- Partial Least Squares (PLS)
- Gaussian Process Regression (GPR)
- Neural Networks (NN)

The models are evaluated using cross-validation and tuned for optimal performance.

## Evaluation

Model performance is assessed using metrics such as:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Visualizations include confusion matrices and ROC curves.

## Contributing

Contributions are welcome! Please create an issue or submit a pull request for any feature requests or improvements.

## License

This project is licensed under the MIT License.

If you use this repository in your research, please cite it as shown in the right sidebar.
