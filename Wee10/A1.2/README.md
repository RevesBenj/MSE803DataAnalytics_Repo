# Salary Prediction: Linear and Polynomial Regression

## Project Overview

This script uses the sample salary dataset to train and compare two regression models:

1. Linear Regression
2. Polynomial Regression with degree 2

The goal is to predict salary based on years of experience. The script also predicts the salary for people with 14, 14.5, and 15 years of experience.

## Dataset

Input file:

```text
salary-dataset (1).csv
```

Main columns used:

| Column | Description |
|---|---|
| YearsExperience | Input feature used for prediction |
| Salary | Target value to predict |

## Data Cleaning and Preprocessing

The script performs these cleaning steps:

- Removes unnecessary `Unnamed` columns.
- Removes duplicate rows.
- Converts `YearsExperience` and `Salary` to numeric values.
- Removes missing values.
- Sorts the data by years of experience.
- Saves the cleaned dataset in the `output` folder.

## Models Used

### Linear Regression

Linear Regression learns a straight-line relationship between years of experience and salary.

### Polynomial Regression

Polynomial Regression uses `PolynomialFeatures(degree=2)` to create curved features such as `YearsExperience²`. This allows the model to learn a curved relationship.

## Error Metrics

The script calculates these metrics for each model:

| Metric | Meaning |
|---|---|
| MAE | Mean Absolute Error. It shows the average absolute prediction error. Lower is better. |
| MSE | Mean Squared Error. It gives bigger penalty to large errors. Lower is better. |
| RMSE | Root Mean Squared Error. It is easier to understand because it is in salary unit. Lower is better. |
| R² Score | Shows how much salary variation is explained by the model. Closer to 1 is better. |

## Prediction Output

The script predicts salary for:

- 14 years of experience
- 14.5 years of experience
- 15 years of experience

Predictions are saved in:

```text
output/salary_predictions_14_14_5_15_years.csv
```

## Visualization

The visualization shows:

- Actual salary data points
- Linear Regression prediction line
- Polynomial Regression prediction curve
- Highlighted predictions for 14, 14.5, and 15 years of experience

Saved file:

```text
output/salary_regression_prediction_visualization.png
```

## Output Files

After running the script, the following files are created:

```text
output/cleaned_salary_dataset.csv
output/model_error_metrics.csv
output/salary_predictions_14_14_5_15_years.csv
output/salary_regression_prediction_visualization.png
```

## How to Run

Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Run the script:

```bash
python salary_regression_models.py
```

## Short Conclusion

Both models perform well on the salary dataset. Polynomial Regression gives slightly lower MAE, MSE, and RMSE, so it performs slightly better for this dataset. However, Linear Regression is still easier to explain because it uses a simple straight-line relationship.
