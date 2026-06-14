# -------------------------------------------------------
# Week 10 A1.2 - Salary Prediction using Linear Regression and Polynomial Regression
# Author : Benjelyn Reves Patiag
# Date   : 20 May 2026
# Description:
# 1. Loads and cleans the salary dataset.
# 2. Trains Linear Regression and Polynomial Regression models.
# 3. Evaluates both models using MAE, MSE, RMSE, and R².
# 4. Predicts salary for 14, 14.5, and 15 years of experience.
# 5. Saves output CSV files and visualization inside the output folder.
# -------------------------------------------------------

"""

This script:
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# STEP 1: File Paths and Output Folder
# ============================================================

DATA_FILE = "salary-dataset (1).csv"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# STEP 2: Load Dataset
# ============================================================

# Load the salary dataset from CSV file.
df = pd.read_csv(DATA_FILE)


# ============================================================
# STEP 3: Data Cleaning and Preprocessing
# ============================================================

# Remove unnecessary index columns such as "Unnamed: 0" if they exist.
df = df.drop(columns=[col for col in df.columns if "Unnamed" in col], errors="ignore")

# Remove duplicate rows to avoid repeated records affecting model training.
df = df.drop_duplicates()

# Convert important columns into numeric values.
# If invalid values exist, they become NaN and will be removed later.
df["YearsExperience"] = pd.to_numeric(df["YearsExperience"], errors="coerce")
df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")

# Remove missing values after conversion.
df = df.dropna(subset=["YearsExperience", "Salary"])

# Sort the data by YearsExperience to make visualization cleaner.
df = df.sort_values("YearsExperience").reset_index(drop=True)

# Save cleaned dataset for checking and reproducibility.
df.to_csv(OUTPUT_DIR / "cleaned_salary_dataset.csv", index=False)


# ============================================================
# STEP 4: Feature Selection
# ============================================================

# X is the independent variable or input feature.
# In this case, it is years of experience.
X = df[["YearsExperience"]]

# y is the dependent variable or target output.
# In this case, it is salary.
y = df["Salary"]


# ============================================================
# STEP 5: Train-Test Split
# ============================================================

# 80% of data is used for training and 20% is used for testing.
# random_state makes the result reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# STEP 6: Train Linear Regression Model
# ============================================================

# Linear Regression learns a straight-line relationship between experience and salary.
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict salary on the test dataset using Linear Regression.
linear_test_pred = linear_model.predict(X_test)


# ============================================================
# STEP 7: Train Polynomial Regression Model
# ============================================================

# Polynomial Regression adds curved features such as X².
# degree=2 means the model can learn a simple curve.
# Pipeline keeps preprocessing and regression together.
poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("linear", LinearRegression())
])

poly_model.fit(X_train, y_train)

# Predict salary on the test dataset using Polynomial Regression.
poly_test_pred = poly_model.predict(X_test)


# ============================================================
# STEP 8: Model Evaluation Function
# ============================================================

def evaluate_model(model_name, y_true, y_pred):
    """Calculate common regression error metrics."""

    # MAE means Mean Absolute Error.
    # It shows the average absolute difference between actual salary and predicted salary.
    # Lower MAE means the model prediction is closer to the actual salary.
    mae = mean_absolute_error(y_true, y_pred)

    # MSE means Mean Squared Error.
    # It squares the prediction errors before averaging them.
    # Large mistakes get bigger penalty, so lower MSE is better.
    mse = mean_squared_error(y_true, y_pred)

    # RMSE means Root Mean Squared Error.
    # It is the square root of MSE.
    # It is easier to understand because it is in the same unit as Salary.
    # Lower RMSE means better prediction accuracy.
    rmse = np.sqrt(mse)

    # R² score explains how much variation in Salary is explained by the model.
    # Values closer to 1 mean better model fit.
    r2 = r2_score(y_true, y_pred)

    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2_Score": r2
    }


# Evaluate Linear Regression.
# For this model, MAE/RMSE show the average salary error from the straight-line prediction.
linear_metrics = evaluate_model("Linear Regression", y_test, linear_test_pred)

# Evaluate Polynomial Regression.
# For this model, MAE/RMSE show the average salary error from the curved-line prediction.
poly_metrics = evaluate_model("Polynomial Regression", y_test, poly_test_pred)

# Save metrics for both models.
metrics_df = pd.DataFrame([linear_metrics, poly_metrics])
metrics_df.to_csv(OUTPUT_DIR / "model_error_metrics.csv", index=False)


# ============================================================
# STEP 9: Predict Salary for 14, 14.5, and 15 Years Experience
# ============================================================

future_experience = pd.DataFrame({"YearsExperience": [14, 14.5, 15]})

# Predict using Linear Regression model.
future_experience["Linear_Regression_Predicted_Salary"] = linear_model.predict(future_experience[["YearsExperience"]])

# Predict using Polynomial Regression model.
future_experience["Polynomial_Regression_Predicted_Salary"] = poly_model.predict(future_experience[["YearsExperience"]])

# Round salary prediction for cleaner output.
future_experience["Linear_Regression_Predicted_Salary"] = future_experience["Linear_Regression_Predicted_Salary"].round(2)
future_experience["Polynomial_Regression_Predicted_Salary"] = future_experience["Polynomial_Regression_Predicted_Salary"].round(2)

# Save future salary predictions.
future_experience.to_csv(OUTPUT_DIR / "salary_predictions_14_14_5_15_years.csv", index=False)


# ============================================================
# STEP 10: Best Visualization
# ============================================================

# Create a smooth range of experience values for plotting both model lines.
experience_grid = pd.DataFrame({
    "YearsExperience": np.linspace(df["YearsExperience"].min(), 15, 300)
})

# Predict salary across the smooth range.
experience_grid["Linear Regression"] = linear_model.predict(experience_grid[["YearsExperience"]])
experience_grid["Polynomial Regression"] = poly_model.predict(experience_grid[["YearsExperience"]])

# Reshape data for easier line plotting.
plot_lines = experience_grid.melt(
    id_vars="YearsExperience",
    value_vars=["Linear Regression", "Polynomial Regression"],
    var_name="Model",
    value_name="Predicted Salary"
)

# Reshape future predictions for easier scatter plotting.
future_plot = future_experience.rename(columns={
    "Linear_Regression_Predicted_Salary": "Linear Regression",
    "Polynomial_Regression_Predicted_Salary": "Polynomial Regression"
}).melt(
    id_vars="YearsExperience",
    value_vars=["Linear Regression", "Polynomial Regression"],
    var_name="Model",
    value_name="Predicted Salary"
)

# Set readable chart size.
plt.figure(figsize=(12, 7))

# Seaborn scatterplot shows the original actual salary data points.
sns.scatterplot(
    data=df,
    x="YearsExperience",
    y="Salary",
    s=80,
    label="Actual Salary Data"
)

# Line plot shows Linear and Polynomial model prediction curves.
sns.lineplot(
    data=plot_lines,
    x="YearsExperience",
    y="Predicted Salary",
    hue="Model",
    linewidth=2.5
)

# Scatterplot highlights the predicted salaries for 14, 14.5, and 15 years.
sns.scatterplot(
    data=future_plot,
    x="YearsExperience",
    y="Predicted Salary",
    hue="Model",
    style="Model",
    s=180,
    legend=False
)

# Add labels beside the future prediction points.
for _, row in future_plot.iterrows():
    plt.text(
        row["YearsExperience"] + 0.05,
        row["Predicted Salary"],
        f'{row["YearsExperience"]} yrs\n{row["Predicted Salary"]:,.0f}',
        fontsize=9
    )

plt.title("Salary Prediction: Linear Regression vs Polynomial Regression")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save visualization in the output folder.
plt.savefig(OUTPUT_DIR / "salary_regression_prediction_visualization.png", dpi=300)
plt.close()


# ============================================================
# STEP 11: Print Results
# ============================================================

print("\nModel Error Metrics")
print(metrics_df.round(2).to_string(index=False))

print("\nSalary Predictions for 14, 14.5, and 15 Years Experience")
print(future_experience.to_string(index=False))

print("\nFiles saved in output folder:")
print("- cleaned_salary_dataset.csv")
print("- model_error_metrics.csv")
print("- salary_predictions_14_14_5_15_years.csv")
print("- salary_regression_prediction_visualization.png")
