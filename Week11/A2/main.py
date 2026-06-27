"""
Week 11 - Activity 2: New Zealand wellbeing statistics: 2014–18 (time series) predication project
Author: Benjelyn Reves Patiag
Date: 27 June 2026

Description:
Develop a forecasting model to predict the next year of time-series data related to wellbeing in New Zealand. 
Apply and compare multiple forecasting techniques, including Linear Regression, XGBoost, LSTM, ANN, and ARIMA. 
Evaluate the performance of each model using appropriate metrics and identify the best-performing approach. 
Share your final results as a GitHub link, including your source code, presentation slides, 
and any supporting documentation. Dataset link: Wellbeing time series explorer | Stats NZ

"""

from src.app import NZWellbeingForecastingApp

if __name__ == "__main__":
    app = NZWellbeingForecastingApp(
        data_path="data/raw/wellbeing-statistics-2014-18-time-series.xlsx",
        output_dir="output",
        models_dir="models",
        next_year=None,
        selected_series=None
    )
    app.run()
