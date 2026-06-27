# NZ Wellbeing Time-Series Forecasting activity

## Activity Goal

This activity uses the attached dataset:

`data/raw/wellbeing-statistics-2014-18-time-series.xlsx`

The goal is to forecast the next year of New Zealand wellbeing time-series data.

The activity applies and compares:

1. Linear Regression
2. XGBoost
3. Artificial Neural Network (ANN)
4. Long Short-Term Memory (LSTM)
5. ARIMA

The code focuses on **loading and preprocessing first**, because the uploaded Excel file is complex and not a simple CSV.

---

## Dataset Used

Dataset file:

```text
wellbeing-statistics-2014-18-time-series.xlsx
```

The workbook contains many wellbeing topics such as:

- Overall life satisfaction
- Life worthwhile
- Financial wellbeing
- Standard of living
- Self-rated health
- Safety and security
- Loneliness
- Trust
- Culture and identity
- Job satisfaction
- Housing condition

---

## Why Preprocessing Is Important

The Excel workbook has:

- Many sheets
- Multi-row headings
- Blank separator columns
- Footnotes in labels
- Estimate columns
- ASE columns
- Change columns
- Suppressed values like `…`
- Repeated demographic groups
- Years stored as rows

Because of this, the activity first converts the workbook into a clean long-format dataset.

The final model-ready dataset includes:

| Column | Meaning |
|---|---|
| Sheet | Excel sheet name |
| Topic | Wellbeing topic |
| Year | Survey year |
| Group_Type | Demographic group type |
| Group | Demographic group |
| Measure | Wellbeing measure |
| Value | Numeric estimate |
| Series_ID | Unique time-series ID |
| Lag_1 | Previous value |
| Lag_2 | Two-period previous value |
| Rolling_Mean_2 | Rolling mean feature |
| Year_Index | Numeric time index |

---

## activity Structure

```text
nz_wellbeing_forecasting_v2/
│
├── main.py
├── README.md
├── requirements.txt
│
├── data/
│   └── raw/
│       └── wellbeing-statistics-2014-18-time-series.xlsx
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   └── visualization.py
│
├── output/
│   ├── tables/
│   └── figures/
│
└── models/
```

---

## Installation

Create virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

---

## Run the activity

Run:

```bash
python main.py
```

The dataset path is already set in `main.py`.

---

## Workflow

### Step 1: Load Data

File:

```text
src/data_loader.py
```

Class:

```python
WellbeingDataLoader
```

This class:

- Checks if the Excel file exists.
- Loads all Excel sheets.
- Skips the Contents sheet.
- Keeps raw sheets for preprocessing.

### Step 2: Preprocess Data

File:

```text
src/preprocessing.py
```

Class:

```python
WellbeingPreprocessor
```

This class:

- Cleans labels.
- Removes footnotes.
- Converts values to numbers.
- Removes missing/suppressed values.
- Extracts year rows.
- Extracts estimate columns.
- Builds a unique `Series_ID`.
- Creates lag features.
- Creates rolling mean features.

### Step 3: Train Models

File:

```text
src/models.py
```

Class:

```python
ForecastModelTrainer
```

Models used:

| Model | Use |
|---|---|
| Linear Regression | Simple baseline |
| XGBoost | Tree boosting model |
| ANN | Deep learning model |
| LSTM | Deep learning time-series model |
| ARIMA | Statistical time-series model |

### Step 4: Evaluate Models

Metrics:

| Metric | Meaning |
|---|---|
| MAE | Mean absolute error |
| MSE | Mean squared error |
| RMSE | Root mean squared error |
| R² | Explained variation |
| MAPE | Percentage error |

The best panel model is selected by the lowest RMSE among Linear Regression, XGBoost, ANN, and LSTM. ARIMA is reported separately because it is fitted to one selected time series only.

### Step 5: Forecast Next Year

If the latest year is 2018, the default forecast year is 2019.

Output file:

```text
output/tables/next_year_forecast.csv
```

---

## Outputs

Tables:

```text
output/tables/clean_long_wellbeing_data.csv
output/tables/model_ready_series_data.csv
output/tables/data_quality_report.csv
output/tables/model_comparison_metrics.csv
output/tables/next_year_forecast.csv
```

Figures:

```text
output/figures/selected_series_trend.png
output/figures/model_comparison_rmse.png
output/figures/sample_next_year_forecast.png
```

Summary:

```text
output/activity_summary.txt
```

Saved models:

```text
models/linear_regression.pkl
models/xgboost.pkl
models/ann.keras
models/lstm.keras
models/arima.pkl
```

ANN uses scikit-learn `MLPRegressor`, so it runs fast. LSTM code is included as optional TensorFlow training. To run real LSTM training, install TensorFlow and run:

```bash
RUN_TENSORFLOW_LSTM=1 python main.py
```

If this variable is not set, LSTM uses a persistence fallback so the activity still runs on normal laptops.

---

## Important Dataset Limitation

This dataset has only a few survey years, mainly 2014, 2016, and 2018.

Because of this:

- Deep learning models may not perform strongly.
- ARIMA is limited because each single time series is short.
- Linear Regression and XGBoost may be more stable.
- The activity uses many wellbeing series together as panel data to improve model training.

This is honest and important for good analysis.

---

## Conclusion

This activity is use the  New Zealand wellbeing Excel dataset. It loads and preprocesses the exact workbook, creates a clean time-series dataset, trains five forecasting models, compares performance, selects the best panel model, reports ARIMA separately, and forecasts the next year.
