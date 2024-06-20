# Failure_prediction_ML
This repository demonstrates how to use machine learning models to predict equipment failures in a manufacturing process in Python. It includes data preprocessing, feature engineering, model training, and evaluation.

## Dataset
The AI4I 2020 Predictive Maintenance Dataset is sourced from the UCI Machine Learning Repository. It contains 10,000 data points with 14 features, including:

UID: Unique identifier
productID: Product quality variant (L, M, H)
air temperature [K]: Air temperature in Kelvin
process temperature [K]: Process temperature in Kelvin
rotational speed [rpm]: Rotational speed in revolutions per minute
torque [Nm]: Torque in Newton-meters
tool wear [min]: Tool wear in minutes
machine failure: Binary label indicating machine failure
Additional failure modes include tool wear failure (TWF), heat dissipation failure (HDF), power failure (PWF), overstrain failure (OSF), and random failures (RNF).
