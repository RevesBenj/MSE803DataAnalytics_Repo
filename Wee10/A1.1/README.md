# Salary Prediction using Linear and Polynomial Regression

## Overview
This project compare:
- Linear Regression
- Polynomial Regression (Degree 2)

Goal is predict Salary based on Years of Experience.

## Dataset
File:
- salary-dataset (1).csv

Columns:
- YearsExperience
- Salary

## Steps
1. Load dataset
2. Clean data
   - Remove unnamed columns
   - Remove duplicates
   - Convert data to numeric
   - Remove missing values
3. Split data
   - 80% Training
   - 20% Testing
4. Train Linear Regression model
5. Train Polynomial Regression model
6. Evaluate models

## Error Metrics
### MAE (Mean Absolute Error)
Average prediction error.
Lower value = better model.

### MSE (Mean Squared Error)
Gives higher penalty to large errors.
Lower value = better model.

### RMSE (Root Mean Squared Error)
Prediction error in salary units.
Lower value = better model.

### R² Score
Shows how well model explains salary variation.
Closer to 1 = better model.

## Libraries Used
- pandas
- numpy
- scikit-learn

## Run
```bash
python Analysis1.py
```

## Output
The script prints:
- MAE
- MSE
- RMSE
- R² Score

for both Linear Regression and Polynomial Regression models.

Actual Reuslt: 
![alt text](image.png)

## Author
Benjelyn Reves Patiag