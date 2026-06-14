# -------------------------------------------------------
# Week 10 - Activity 1.1 - Prediction time series dataset with regression
# Author : Benjelyn Reves Patiag
# Date   : 14 June 2026
# Description:
# Use the sample dataset to implement both Linear Regression and Polynomial Regression models. Share your code and short description about the error metrics (e.g., MAE, MSE, and RMSE) for each model.
# -------------------------------------------------------


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# STEP 1: Load Dataset
# ============================================================
df = pd.read_csv("salary-dataset (1).csv")

# ============================================================
# STEP 2: Data Cleaning and Preprocessing
# ============================================================

# Remove unnecessary unnamed columns if present
df = df.drop(columns=[col for col in df.columns if "Unnamed" in col])

# Remove duplicate records
df = df.drop_duplicates()

# Convert columns to numeric format
df["YearsExperience"] = pd.to_numeric(df["YearsExperience"], errors="coerce")
df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")

# Remove missing values
df = df.dropna()

# ============================================================
# STEP 3: Feature Selection
# ============================================================

# Independent Variable (X)
X = df[["YearsExperience"]]

# Dependent Variable (y)
y = df["Salary"]

# ============================================================
# STEP 4: Train-Test Split
# ============================================================

# 80% training data, 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ============================================================
# STEP 5: Linear Regression Model
# ============================================================

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict Salary using Linear Regression
linear_pred = linear_model.predict(X_test)

# ============================================================
# STEP 6: Polynomial Regression Model
# ============================================================

poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("linear", LinearRegression())
])

poly_model.fit(X_train, y_train)

# Predict Salary using Polynomial Regression
poly_pred = poly_model.predict(X_test)

# ============================================================
# STEP 7: Model Evaluation Function
# ============================================================

def evaluate_model(name, y_true, y_pred):

    # MAE (Mean Absolute Error)
    # Measures the average prediction error.
    # Lower MAE means predictions are closer to actual values.
    mae = mean_absolute_error(y_true, y_pred)

    # MSE (Mean Squared Error)
    # Squares each error before averaging.
    # Large prediction errors receive a higher penalty.
    # Lower MSE indicates better model performance.
    mse = mean_squared_error(y_true, y_pred)

    # RMSE (Root Mean Squared Error)
    # Square root of MSE.
    # Easier to interpret because it is in the same unit as Salary.
    # Lower RMSE means more accurate predictions.
    rmse = np.sqrt(mse)

    # R² Score
    # Measures how well the model explains the variation in Salary.
    # Values closer to 1 indicate better performance.
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

# ============================================================
# STEP 8: Evaluate Models
# ============================================================

evaluate_model("Linear Regression", y_test, linear_pred)
evaluate_model("Polynomial Regression", y_test, poly_pred)

# ============================================================
# INTERPRETATION OF ERROR METRICS
# ============================================================

# MAE:
# Average prediction error.
# Lower MAE = Better model.

# MSE:
# Gives greater penalty to large prediction errors.
# Lower MSE = Better model.

# RMSE:
# Indicates average prediction error in Salary units.
# Lower RMSE = Better model.

# Compare the MAE, MSE, and RMSE values of both models.
# The model with the lowest values generally provides
# more accurate predictions on the test dataset.