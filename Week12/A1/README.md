# Hotel Booking Data Analysis Project

This README documents the hotel booking analysis project.

## Overview

Complete workflow: loading, preprocessing, EDA, feature engineering, ML
comparison.

## Dataset

hotel_bookings.csv

## Pipeline

1.  Load dataset
2.  Inspect data
3.  Clean missing values
4.  Remove duplicates/outliers
5.  Feature engineering
6.  EDA summaries
7.  Preprocess (Scaling + One-Hot Encoding)
8.  Train Logistic Regression, Decision Tree and Random Forest
9.  Compare Accuracy, Precision, Recall, F1 and ROC-AUC

## Cleaning

-   Remove duplicates
-   Fill missing values
-   Remove invalid guest records
-   Remove negative/extreme ADR
-   Convert dates

## Features

-   total_guests
-   total_stay_nights
-   has_agent
-   has_company
-   room_changed
-   is_peak_month

## Outputs

-   cleaned_hotel_bookings.csv
-   hotel_summary.csv
-   monthly_summary.csv
-   room_type_summary.csv
-   model_comparison.csv
-   preprocessing_report.txt

## Expected Result

Random Forest is typically the best-performing model (F1≈0.67,
ROC-AUC≈0.89).
