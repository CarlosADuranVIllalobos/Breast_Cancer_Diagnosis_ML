This repository demonstrates how to use machine learning models to predict breast cancer diagnosis using the Breast Cancer Wisconsin (Diagnostic) Dataset in Python. It includes data preprocessing, feature engineering, model training, and evaluation.

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The Breast Cancer Wisconsin (Diagnostic) Dataset is sourced from the UCI Machine Learning Repository. It contains 569 data points with 30 features, including:

- `ID`: Unique identifier
- `Diagnosis`: Binary label indicating diagnosis (M = malignant, B = benign)
- `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, etc.: Various measurements of cell nuclei characteristics

### Citation of the Dataset

Wolberg,William, Mangasarian,Olvi, Street,Nick, and Street,W.. (1995). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.

## Project Structure
Breast_Cancer_Diagnosis_ML/

```plaintext
Breast_Cancer_Diagnosis_ML/
├── data/
│   ├── breast_cancer_wisconsin.csv  # Raw dataset
├── notebooks/
│   ├── data_preprocessing.ipynb     # Data preprocessing and EDA
│   ├── feature_engineering.ipynb    # Feature engineering
│   ├── model_training.ipynb         # Model training and tuning
│   ├── model_evaluation.ipynb       # Model evaluation
├── scripts/
│   ├── preprocess.py                # Data preprocessing script
│   ├── features.py                  # Feature selection script
│   ├── train.py                     # Model training script
│   ├── evaluate.py                  # Model evaluation script
├── models/
│   ├── model.pkl                    # Trained model
├── results/
│   ├── evaluation_metrics.csv       # Evaluation metrics
│   ├── confusion_matrix.png         # Confusion matrix
├── README.md                        # Project README
```

## Usage

1. **Data Preprocessing**:
   - Execute the `data_preprocessing.ipynb` notebook to clean and preprocess the data.
     [Data Preprocessing Notebook](notebooks/data_preprocessing.ipynb)

2. **Feature Engineering**:
   - Run the `feature_engineering.ipynb` notebook to create new features and select the most important ones.

3. **Model Training**:
   - Use the `model_training.ipynb` notebook to train various machine learning models and tune hyperparameters.

4. **Model Evaluation**:
   - Evaluate the performance of the trained models using the `model_evaluation.ipynb` notebook.


## Modeling

The project explores various machine learning models, including:

- Random Forest (RF)
- Gradient Boosting (GB)
- Partial Least Squares (PLS)
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
