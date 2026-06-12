# MSE803 Assessment 1
# Avon River Water Quality & Fish Population Analysis

**Author:** Benjelyn Reves Patiag
**Student ID:** 270770489
**Course:** MSE803 – Data Analytics
**Institution:** Yoobee College of Creative Innovation
**Date:** 20 May 2026
**GitHub Repository:** [RevesBenj/MSE803DataAnalytics_Repo](https://github.com/RevesBenj/MSE803DataAnalytics_Repo/tree/main/Assessment1AvonRiver)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Background](#background)
3. [Problem Statement](#problem-statement)
4. [Objectives](#objectives)
5. [Expected Outcomes](#expected-outcomes)
6. [Dataset Description](#dataset-description)
7. [Data Preparation & Cleaning](#data-preparation--cleaning)
8. [Exploratory Data Analysis](#exploratory-data-analysis)
9. [Correlation Analysis (Technique 1)](#correlation-analysis-technique-1)
10. [Multiple Linear Regression (Technique 2)](#multiple-linear-regression-technique-2)
11. [Environmental Challenges Identified](#environmental-challenges-identified)
12. [Tool Justification](#tool-justification)
13. [Data Visualisations](#data-visualisations)
14. [Findings](#findings)
15. [Recommendations](#recommendations)
16. [System Architecture](#system-architecture)
17. [Technology Stack](#technology-stack)
18. [Project Structure](#project-structure)
19. [Installation](#installation)
20. [Usage Guide](#usage-guide)
21. [Data Analysis Workflow](#data-analysis-workflow)
22. [Known Limitations](#known-limitations)
23. [Future Enhancements](#future-enhancements)
24. [License](#license)
25. [Author / Credits](#author--credits)

---

## Project Overview

This project is for **MSE803 Assessment 1**. The goal is to use Python data analysis and statistics
to investigate the **Avon River** (Ōtākaro) in Christchurch, New Zealand.

2 things:

1. **Water quality**: temperature, pH, dissolved oxygen
2. **Fish population**: species, count, average size

The project uses an object-oriented Python pipeline with **9 classes** covering data loading,
cleaning, exploratory analysis, statistical modelling, machine learning, visualisation, and
automated report export.

---

## Background

This project analyses data collected from three monitoring sites (AV-1, AV-2, AV-3) over
October to December 2023 to understand the current health of the river.

---

## Problem Statement

2 environmental challenges are identified from the data:

**Challenge 1: Poor water quality may reduce fish health and population**

Dissolved oxygen (DO), temperature, and pH are key water quality parameters. When these
go outside the comfortable range for fish, fish health and count can decrease.

**Challenge 2: Some sites or species show poor fish population indicators**

Not all sites and species are equal. Some monitoring sites show lower fish count and
smaller fish size. Some species like Shortfin Eel have very low count at all sites.

These two challenges are the focus of this investigation.

---

## Objectives

1. Load and clean the Avon River Excel dataset
2. Explore and summarise data using descriptive statistics
3. Apply **Pearson correlation analysis** to find relationships between water quality and fish
4. Apply **multiple linear regression** to predict fish count from water quality parameters
5. Apply **advanced machine learning** (K-Means, PCA, six classifiers, time series forecasting)
6. Generate 17 professional visualisation charts
7. Create five evidence-based environmental recommendations

---

## Expected Outcomes

After running the code, the user will have:

- Clean dataset (Excel and CSV format)
- Correlation matrix with interpretation
- Regression model with R², MAE, RMSE
- 17 PNG chart figures (12 core + 5 advanced ML)
- Site and species summary reports
- Advanced ML tables (14 files) covering clustering, classification, and time series
- Five recommendations for environmental action
- Assessment mapping document linking all outputs to tasks

---

## Dataset Description

**File:** `data/Data_Set_Assignmnet_1-V0.1_20426.xlsx`

The Excel file has **one sheet** with **71 rows** and **11 columns** (including a separator column).
Row 0 is a large merged header row. Row 1 contains the actual column names.
The code reads using `header=1` to skip the top merged row correctly.

The data is arranged as two side-by-side tables:

| Section | Column | Description |
|---------|--------|-------------|
| Water Quality | Site ID | Monitoring site (AV-1, AV-2, AV-3) |
| Water Quality | Date | Date of water quality measurement |
| Water Quality | Temperature (°C) | Water temperature in Celsius |
| Water Quality | pH | Acidity or alkalinity of water |
| Water Quality | Dissolved Oxygen (mg/L) | Oxygen dissolved in water |
| Fish Population | Site ID | Same monitoring site |
| Fish Population | Date | Date of fish survey |
| Fish Population | Species | Fish species name |
| Fish Population | Count | Number of fish observed |
| Fish Population | Avg. Size (cm) | Average fish length in centimetres |

**Sites:**
- **AV-1**: Upstream site (19 records)
- **AV-2**: Middle site (25 records)
- **AV-3**: Downstream site (26 records)

**Species found:**
- Brown Trout (native/introduced)
- Inanga (whitebait, native, threatened)
- Longfin Eel (native, protected)
- Shortfin Eel (native)

**Time period:** October – December 2023 (spring to early summer)

---

## Data Preparation & Cleaning


Data cleaning is performed by the **DataCleaner** class in `main.py`.

* Removed blank separator column (`Unnamed: 5`).
* Renamed columns to simple names (Site_ID, Date, Temperature_C, pH, Dissolved_Oxygen, Species, Fish_Count, Avg_Size_cm).
* Converted `Date` and `Date_FP` to datetime format.
* Replaced **"No fish observed"** with **0** before numeric conversion.
* Converted numeric fields using `pd.to_numeric()`.
* Filled missing values:

  * Species → `"Unknown"`
  * Numeric values → column median.
* Removed exact duplicate records.
* Applied plausibility checks:

  * Temperature (0–40°C)
  * pH (0–14)
  * Dissolved Oxygen (0–20 mg/L)
  * Fish Count (0–500)
  * Average Size (0–200 cm)
* Detected and removed extreme outliers using the **IQR method**.
* Created new features:

  * Fish_Health_Index (FHI)
  * Water_Quality_Index (WQI)
  * Oxygen_Status
  * pH_Status
  * Temperature_Status
  * Site_Health_Category
  * Month
  * Season

**Final cleaned dataset:** 70 records × 18 columns.

---

## Exploratory Data Analysis

All EDA is done by the `ExploratoryDataAnalyzer` class.

### Descriptive Statistics

| Variable | Mean | Std Dev | Min | Max |
|----------|------|---------|-----|-----|
| Temperature (°C) | 16.39 | 2.12 | 13.2 | 21.0 |
| pH | 7.44 | 0.21 | 7.04 | 7.88 |
| Dissolved Oxygen (mg/L) | 7.87 | 0.74 | 6.5 | 9.2 |
| Fish Count | 13.1 | 7.70 | 3 | 30 |
| Avg Size (cm) | 23.08 | 15.01 | 6.9 | 44.9 |

Skewness and kurtosis are also computed for all numeric columns and saved to `descriptive_statistics.csv`.

### Site Comparison

| Site | Temp (°C) | pH | DO (mg/L) | Mean Fish Count | WQI |
|------|-----------|----|-----------|-----------------|-----|
| AV-1 | 14.8 | 7.32 | 7.87 | 11.2 | 0.749 |
| AV-2 | 16.2 | 7.54 | 8.06 | 12.8 | 0.760 |
| AV-3 | 17.7 | 7.44 | 7.69 | 14.8 | 0.678 |

AV-3 (downstream) has the highest temperature and lowest water quality index.

### Species Summary

| Species | Total Fish | Mean Count | Mean Size (cm) |
|---------|-----------|-----------|----------------|
| Inanga | 472 | 19.7 | 7.6 |
| Brown Trout | 234 | 12.3 | 17.9 |
| Longfin Eel | 120 | 8.0 | 42.7 |
| Shortfin Eel | 65 | 5.9 | 40.3 |

Inanga is the most abundant species. Shortfin Eel has the lowest count.

### Monthly Trend
Mean DO, mean fish count, and mean WQI are aggregated by month (Oct, Nov, Dec)
to show seasonal changes across the monitoring period.

---

## Correlation Analysis (Technique 1)

### Why Correlation Analysis is Suitable

Pearson Correlation Analysis is used to measure the relationship between water quality and fish population variables.

* Suitable for continuous numeric data.
* Measures strength and direction of relationships.
* Produces correlation coefficient (r) between -1 and +1.
* Widely used in environmental and data analytics studies.

### Method

Pearson correlation coefficient was calculated using `df.corr(method="pearson")`.

Correlation strength:

* Strong: |r| ≥ 0.70
* Moderate: |r| ≥ 0.40
* Weak: |r| < 0.40

P-values were calculated using `scipy.stats.pearsonr()`.

### Key Findings

| Variable Pair             | r Value | Interpretation     |
| ------------------------- | ------- | ------------------ |
| DO vs Fish Count          | +0.03   | Weak positive      |
| DO vs WQI                 | +0.95   | Strong positive    |
| Temperature vs DO         | -0.00   | Very weak negative |
| Temperature vs Fish Count | +0.11   | Weak positive      |
| pH vs Fish Count          | +0.13   | Weak positive      |
| WQI vs FHI                | -0.14   | Weak negative      |
| Fish Count vs Avg Size    | -0.67   | Moderate negative  |

The most notable result is the moderate negative correlation between Fish Count and Average Fish Size (r = -0.67). Sites with many small fish species, such as Inanga, tend to have lower average size, while sites with fewer large fish, such as eels, show higher average size.

### Important Limitation

Correlation does not prove causation. A relationship between two variables does not mean one directly causes the other. Other factors such as habitat condition, pollution, season, and river flow may also influence the results.


---

## Multiple Linear Regression (Technique 2)

### Why Regression Analysis is Suitable

Multiple Linear Regression is used to examine how water quality variables affect fish count.

* Predict fish count from environmental data.
* Uses multiple predictors: Temperature, pH, and Dissolved Oxygen (DO).
* Measures contribution of each variable using coefficients.
* Uses R² to evaluate model performance.
* Common technique in environmental and ecological data analysis.

### Method

* Model: Ordinary Least Squares (OLS) Multiple Linear Regression.
* Target Variable: Fish_Count.
* Predictors: Temperature_C, pH, Dissolved_Oxygen.
* Evaluation: Full cleaned dataset (explanatory analysis).
* OLS (`statsmodels`): provides coefficients, p-values, and confidence intervals.
* Scikit-Learn (`LinearRegression`): provides R², MAE, and RMSE.

### Results

| Metric | Value  |
| ------ | ------ |
| R²     | ~0.025 |
| MAE    | ~7.3   |
| RMSE   | ~8.2   |

OLS results show all predictors are not statistically significant (p > 0.05), and the overall model is not significant.

### Interpretation

The model has very low explanatory power (R² ≈ 0.025). Temperature, pH, and DO alone cannot reliably explain fish count variation.

Possible factors not included in the model:

* Species type and behaviour.
* Habitat condition.
* Seasonal migration patterns.
* Other environmental and pollution factors.

### Limitations

* Small dataset (n = 70).
* Only 3 monitoring sites (AV-1, AV-2, AV-3).
* Only 3 months of data (Oct–Dec 2023).
* Species composition varies between sites.
* Important variables such as turbidity, nutrients, and flow rate are not available.


---

## Environmental Challenges Identified

### Challenge 1: Poor Water Quality May Affect Fish Health

**Evidence:**

* AV-3 has the lowest Water Quality Index (WQI = 0.678).
* AV-3 has the highest average temperature (17.7°C).
* Dissolved Oxygen (DO) shows a declining trend in December.
* DO and WQI have a strong positive correlation, although WQI already includes DO in its calculation.

**Environmental Impact:**

Lower DO and higher temperature can stress freshwater fish. Some AV-3 readings are close to the 6 mg/L DO threshold considered important for healthy fish populations.

### Challenge 2: Weak Fish Population Indicators

**Evidence:**

* Shortfin Eel has the lowest total population (65 fish).
* Longfin Eel also has low population counts.
* One site records the lowest Fish Health Index (FHI).
* Only four fish species were observed across three monitoring sites.

**Environmental Impact:**

Low eel populations may indicate conservation concern. Longfin Eel is a taonga species in New Zealand and continued decline may affect both biodiversity and cultural values.


---

## Tool Justification

### Tool 1: Python

Python was used for data cleaning, analysis, visualisation, and machine learning.

* **Pandas** – data cleaning and summaries
* **NumPy** – calculations
* **Matplotlib & Seaborn** – charts and visualisations
* **Scikit-Learn** – regression, classification, K-Means, PCA
* **Statsmodels** – OLS regression
* **SciPy** – Pearson correlation analysis

### Tool 2: Power BI

Power BI is suitable for non-technical users because:

* Easy dashboard creation
* Supports filters and KPI cards
* Simple reporting and sharing
* Can directly import `cleaned_avon_river_data.xlsx`


---

## Data Visualisations

All figures are saved in `outputs/figures/`. Charts include titles, labels, legends, and consistent site colours:

* AV-1 = #2d6a4f
* AV-2 = #457b9d
* AV-3 = #e76f51

### Core Analysis Figures (12)

| Figure | Type          | Purpose                                    |
| ------ | ------------- | ------------------------------------------ |
| Fig 01 | Histogram     | Distribution of key variables              |
| Fig 02 | Boxplot       | Spread and outlier detection               |
| Fig 03 | Heatmap       | Pearson correlation matrix                 |
| Fig 04 | Scatter       | DO vs Fish Count                           |
| Fig 05 | Scatter       | Temperature vs DO                          |
| Fig 06 | Scatter       | pH vs Fish Count                           |
| Fig 07 | Regression    | Actual vs Predicted and Residual Plot      |
| Fig 08 | Pair Plot     | Relationships between numeric variables    |
| Fig 09 | Bar Chart     | Site comparison (WQI, FHI, DO, Fish Count) |
| Fig 10 | Bar + Boxplot | Species distribution and fish size         |
| Fig 11 | Line Chart    | Monthly DO, Fish Count, and WQI trends     |
| Fig 12 | Dashboard     | WQI & FHI by site, WQI vs FHI, DO trend    |

### Advanced ML Figures (5)

| Figure | Type         | Purpose                                    |
| ------ | ------------ | ------------------------------------------ |
| Fig 13 | Line Chart   | K-Means Elbow Method                       |
| Fig 14 | Line Chart   | K-Means Silhouette Score                   |
| Fig 15 | Scatter Plot | PCA visualisation of K-Means clusters      |
| Fig 16 | Bar Chart    | Macro F1-score comparison of 6 classifiers |
| Fig 17 | Line Chart   | Monthly trend and rolling forecast         |

### Output Location

```text
outputs/figures/
```

### GitHub Repository

https://github.com/RevesBenj/MSE803DataAnalytics_Repo/tree/main/Assessment1AvonRiver

---

## Findings

* Water quality is generally good across all sites (WQI = 0.68–0.76), but AV-3 has the lowest WQI.
* Inanga is the most abundant species, while Longfin Eel and Shortfin Eel have low population counts.
* Fish Count and Average Size show a moderate negative correlation (r = -0.67), mainly due to species differences.
* Water quality variables alone are weak predictors of Fish Count (R² ≈ 0.025). The OLS model is not significant.
* Dissolved Oxygen shows a declining trend in December, consistent with warmer summer conditions.
* AV-3 is the most stressed site, with the highest temperature and lowest WQI.
* K-Means clustering identified hidden patterns in the dataset, supported by PCA visualisation.
* Six classification models were tested to predict Water_Quality_Category. The target label was created from WQI quantiles and is not an official ecological classification.

---

## Recommendations

### R1: Improve Water Quality at AV-3

**Action:** Increase DO monitoring, reduce runoff, and expand riparian planting.

**Evidence:** AV-3 has the lowest WQI and highest temperature.

**Benefit:** Supports better river health and long-term monitoring.

### R2: Increase Monitoring and Collect More Variables

**Action:** Monitor more frequently and collect turbidity, nutrients, flow rate, rainfall, and habitat data.

**Evidence:** Regression performance is very low (R² ≈ 0.025).

**Benefit:** Improves future analysis and prediction accuracy.

### R3: Investigate Low Fish Health Site

**Action:** Conduct detailed fish and habitat surveys at the site with the lowest FHI.

**Evidence:** Lowest FHI and lowest average fish count recorded.

**Benefit:** Helps identify habitat or ecological issues.

### R4: Monitor Temperature and pH Thresholds

**Action:** Track pH outside 6.5–8.5 and temperatures above 18°C.

**Evidence:** Highest temperature observed at AV-3.

**Benefit:** Provides early warning of water-quality stress.

### R5: Develop Power BI

**Action:** Create interactive dashboards showing WQI, FHI, DO, fish count, and trends.

**Evidence:** Python outputs are ready for dashboard integration.

**Benefit:** Improves stakeholder reporting and decision-making.

---

## System Architecture

The project follows the **Single Responsibility Principle**.

```
AvonRiverAnalysisApp (Orchestrator, CLASS 9)
    │
    ├── DataLoader (CLASS 1)
    │       └── load()  →  reads Excel with header=1, inspects shape and columns
    │
    ├── DataCleaner (CLASS 2)
    │       ├── clean()  →  runs Steps 2a–2j, returns clean DataFrame (70 rows × 18 cols)
    │       ├── _engineer_features()  →  creates FHI, WQI, status labels, Month, Season
    │       └── save()  →  exports Excel + CSV to outputs/cleaned_data/
    │
    ├── ExploratoryDataAnalyzer (CLASS 3)
    │       ├── analyze()  →  descriptive stats, site summary, species summary, monthly trend
    │       └── save_reports()  →  saves CSV files to outputs/reports/
    │
    ├── StatisticalAnalyzer (CLASS 4)
    │       ├── correlation_analysis()  →  Pearson matrix + key pair analysis with p-values
    │       ├── regression_analysis()  →  OLS + Scikit-learn metrics on full cleaned dataset
    │       └── save_reports()  →  saves correlation_matrix.csv + regression_results.txt
    │
    ├── VisualizationGenerator (CLASS 5)
    │       ├── fig01_histograms() through fig12_executive_dashboard()
    │       └── generate_all()  →  runs all 12 core charts in sequence
    │
    ├── RecommendationEngine (CLASS 6)
    │       ├── generate()  →  5 evidence-based recommendations from real dataset statistics
    │       └── save()  →  saves recommendations.txt
    │
    ├── ReportExporter (CLASS 7)
    │       └── export_assessment_mapping()  →  generates assessment_mapping.md
    │
    └── AdvancedMLAnalyzer (CLASS 8)
            ├── _create_class_target()  →  creates Water_Quality_Category from WQI quantile bins
            ├── _linear_regression_metrics()  →  aligned ML linear regression metric table
            ├── _kmeans_clustering()  →  K-Means + Elbow + Silhouette + PCA + cluster summary
            ├── _classification_models()  →  6 classifiers with accuracy, F1, confusion matrix
            ├── _time_series_analysis()  →  monthly trend + simple rolling forecast
            └── _save_notes()  →  saves advanced_ml_conclusion_and_limitations.txt
```

---

## Technology Stack

| Category | Tool | Version | Official Reference |
|----------|------|---------|--------------------|
| Language | Python | 3.9+ | https://www.python.org |
| Data manipulation | Pandas | ≥1.5 | https://pandas.pydata.org |
| Numeric computing | NumPy | ≥1.23 | https://numpy.org |
| Charting | Matplotlib | ≥3.6 | https://matplotlib.org |
| Statistical charts | Seaborn | ≥0.12 | https://seaborn.pydata.org |
| Machine learning | Scikit-Learn | ≥1.2 | https://scikit-learn.org |
| Statistical modelling | Statsmodels | ≥0.13 | https://www.statsmodels.org |
| Scientific stats | SciPy | ≥1.9 | https://scipy.org |
| Excel read/write | OpenPyXL | ≥3.0 | https://openpyxl.readthedocs.io |
| BI dashboard (recommended) | Power BI | Current | https://powerbi.microsoft.com |

---

## Project Structure

```
avon_project/
│
├── main.py                        ← Run this file (9 classes)
├── requirements.txt               ← All Python libraries needed
├── README.md                      ← This file
│
├── data/
│   └── Data_Set_Assignmnet_1-V0.1_20426.xlsx   ← Original dataset
│
└── outputs/
    ├── cleaned_data/
    │   ├── cleaned_avon_river_data.xlsx
    │   └── cleaned_avon_river_data.csv
    │
    ├── figures/
    │   ├── fig01_histograms.png
    │   ├── fig02_boxplots.png
    │   ├── fig03_correlation_heatmap.png
    │   ├── fig04_scatter_do_vs_fishcount.png
    │   ├── fig05_scatter_temp_vs_do.png
    │   ├── fig06_scatter_ph_vs_fishcount.png
    │   ├── fig07_regression_plot.png
    │   ├── fig08_pairplot.png
    │   ├── fig09_site_comparison.png
    │   ├── fig10_species_distribution.png
    │   ├── fig11_monthly_trends.png
    │   ├── fig12_executive_dashboard.png
    │   ├── fig13_kmeans_elbow_method.png
    │   ├── fig14_kmeans_silhouette_score.png
    │   ├── fig15_pca_kmeans_clusters.png
    │   ├── fig16_classification_model_comparison.png
    │   └── fig17_time_series_trend_forecast.png
    │
    ├── reports/
    │   ├── descriptive_statistics.csv
    │   ├── correlation_matrix.csv
    │   ├── regression_results.txt
    │   ├── site_summary.csv
    │   ├── species_summary.csv
    │   ├── recommendations.txt
    │   ├── assessment_mapping.md
    │   └── advanced_ml_conclusion_and_limitations.txt
    │
    └── tables/
        ├── ml_created_class_target.csv
        ├── ml_linear_regression_metrics.csv
        ├── ml_kmeans_elbow_silhouette_scores.csv
        ├── ml_kmeans_clustered_records.csv
        ├── ml_pca_2d_cluster_points.csv
        ├── ml_kmeans_cluster_summary_table.csv
        ├── ml_classification_model_comparison.csv
        ├── ml_confusion_matrix_logistic_regression.csv
        ├── ml_confusion_matrix_svm.csv
        ├── ml_confusion_matrix_decision_tree.csv
        ├── ml_confusion_matrix_random_forest.csv
        ├── ml_confusion_matrix_gradient_boosting.csv
        ├── ml_confusion_matrix_neural_network___mlp.csv
        └── ml_time_series_monthly_trend_forecast.csv
```

---

## Installation

### Step 1: Make sure Python is installed

```bash
python --version
```

You need Python 3.9 or higher. Download from: https://www.python.org/downloads/

### Step 2: Clone or download the repository

```bash
git clone https://github.com/RevesBenj/MSE803DataAnalytics_Repo.git
cd MSE803DataAnalytics_Repo/Assessment1AvonRiver
```

### Step 3: Put your dataset in the data folder

Make sure the file `Data_Set_Assignmnet_1-V0.1_20426.xlsx` is inside the `data/` folder.
If the `data/` folder does not exist, create it first. The code also checks for the file
in the same directory as `main.py` as a fallback.

### Step 4: Install all required libraries

```bash
pip install -r requirements.txt
```

This installs: pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, scipy, openpyxl.

Alternatively, install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy openpyxl
```

**Note on thread environment:**
The code automatically sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` to `"1"`.
This prevents slow or hanging behaviour from too many math-library threads on small datasets.

---

## Usage Guide

### Run the full analysis

```bash
python main.py
```

This will run all 8 steps in sequence:

1. Load and inspect the Excel dataset (skip merged header row)
2. Clean and preprocess all data (Steps 2a–2j)
3. Run descriptive statistics and EDA
4. Run Pearson correlation analysis with p-values
5. Run multiple linear regression (OLS + Scikit-learn)
6. Generate 12 core PNG charts
7. Generate 5 environmental recommendations
8. Save all core reports
9. Run advanced ML analysis (K-Means, PCA, 6 classifiers, time series) + 5 ML charts + 14 ML tables

All outputs appear in the `outputs/` folder automatically.
You do not need to create any folders: the code makes them for you.

### Check results

After running, open these folders:

- `outputs/figures/`, view all 17 PNG charts
- `outputs/reports/`, read CSV and text reports
- `outputs/cleaned_data/`, use clean data in Power BI or Excel
- `outputs/tables/`, review all 14 advanced ML CSV tables

### Output folder reference links (GitHub)

| Folder | Link |
|--------|------|
| Cleaned data | [outputs/cleaned_data/](https://github.com/RevesBenj/MSE803DataAnalytics_Repo/tree/main/Assessment1AvonRiver/outputs/cleaned_data) |
| Figures | [outputs/figures/](https://github.com/RevesBenj/MSE803DataAnalytics_Repo/tree/main/Assessment1AvonRiver/outputs/figures) |
| Reports | [outputs/reports/](https://github.com/RevesBenj/MSE803DataAnalytics_Repo/tree/main/Assessment1AvonRiver/outputs/reports) |
| ML Tables | [outputs/tables/](https://github.com/RevesBenj/MSE803DataAnalytics_Repo/tree/main/Assessment1AvonRiver/outputs/tables) |

---

## Data Analysis Workflow

```text
Excel Dataset (.xlsx)
        ↓
DataLoader
- Load and inspect data
        ↓
DataCleaner
- Clean, transform, remove outliers
- Create WQI, FHI, and other features
        ↓
ExploratoryDataAnalyzer
- Descriptive statistics
- Site, species, and monthly analysis
        ↓
StatisticalAnalyzer
- Pearson Correlation
- Multiple Linear Regression
        ↓
VisualizationGenerator
- Generate Fig01–Fig12
        ↓
RecommendationEngine
- Generate 5 recommendations
        ↓
ReportExporter
- Create assessment mapping report
        ↓
AdvancedMLAnalyzer
- K-Means, PCA
- 6 Classification Models
- Time Series Analysis
- Generate Fig13–Fig17 and ML tables
        ↓
outputs/
- All reports, figures, and tables saved automatically
```


---

## Known Limitations

1. **Small dataset** – Only 70 records. More data and more years can improve result.

2. **Short time period** – Data only from Oct–Dec 2023. Cannot represent full yearly changes.

3. **Only 3 sites** – AV-1, AV-2, and AV-3 cover only part of Avon River.

4. **Missing variables** – Turbidity, nitrate, phosphate, flow rate, and habitat data are not available.

5. **Species bias** – Some species like Inanga appear in large groups, while eels are usually fewer.

6. **Correlation and regression limitations** – Relationships found do not prove direct cause and effect.

7. **Created ML labels** – `Water_Quality_Category` was generated from WQI and is not an official ecological classification.


---

## Future Enhancements

1. Collect data from more sites along the full Avon River
2. Add more water quality variables (turbidity, nutrients, flow rate)
3. Collect multiple years of data to do proper seasonal and long-term trend analysis
4. Use more advanced models (XGBoost, LightGBM) to improve fish count prediction
5. Build Power BI dashboard connected to live sensor data
6. Include human activity data (land use, stormwater pipe location) as additional predictors
7. Apply time series analysis (ARIMA, SARIMA) for long-term trend forecasting
8. Add species habitat suitability modelling
9. Expand AdvancedMLAnalyzer to include cross-validation for more robust model comparison
10. Add proper train/test split evaluation to classification models for generalisation testing


---

## License

This project is for academic submission only (MSE803 Assessment 1).
Not for commercial use.

---

## Author / Credits

**Student:** Benjelyn Reves Patiag

**Student ID:** 270770489

**Course:** MSE803 Data Analytics

**Institution:** Yoobee College of Creative Innovation

**GitHub:** [https://github.com/RevesBenj/MSE803DataAnalytics_Repo/tree/main/Assessment1AvonRiver](https://github.com/RevesBenj/MSE803DataAnalytics_Repo/tree/main/Assessment1AvonRiver)



**Official documentation references:**

| Library / Tool | Official Site |
|----------------|---------------|
| Python | https://www.python.org |
| Pandas | https://pandas.pydata.org |
| NumPy | https://numpy.org |
| Matplotlib | https://matplotlib.org |
| Seaborn | https://seaborn.pydata.org |
| Scikit-Learn | https://scikit-learn.org |
| Statsmodels | https://www.statsmodels.org |
| SciPy | https://scipy.org |
| OpenPyXL | https://openpyxl.readthedocs.io |
| Power BI | https://powerbi.microsoft.com |
| Git / GitHub | https://github.com |

---