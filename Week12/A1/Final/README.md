
# Hotel Booking Data Analysis Project

## 1. Project Overview

This project analyzes the `hotel_bookings.csv` dataset.

The goal is to turn hotel booking data into useful business insights.

The project covers:

- Data loading
- Data inspection
- Data preprocessing and cleaning
- Feature engineering
- Exploratory data analysis
- Visualization output
- Machine learning model comparison
- Best-performing approach selection
- Business decision-making insights

The main business question is:

> How can hotel booking data help management improve cancellation control, room planning, occupancy management, and dynamic pricing?

---

## 2. Dataset

The dataset contains hotel booking records.

Important fields include:

- `hotel`
- `is_canceled`
- `lead_time`
- `arrival_date_year`
- `arrival_date_month`
- `arrival_date_day_of_month`
- `adults`
- `children`
- `babies`
- `reserved_room_type`
- `assigned_room_type`
- `adr`
- `customer_type`
- `market_segment`
- `deposit_type`
- `total_of_special_requests`

Target variable:

- `is_canceled`
  - `0` = booking was not canceled
  - `1` = booking was canceled

---

## 3. Project Structure

```text
hotel_booking_analysis_code/
│
├── hotel_booking_analysis.py
├── README.md
├── requirements.txt
│
└── output/
    ├── cleaned_hotel_bookings_sample.csv
    ├── analysis_summary.md
    │
    ├── figures/
    │   ├── 01_bookings_by_hotel.png
    │   ├── 02_monthly_booking_trend.png
    │   ├── 03_monthly_adr_trend.png
    │   ├── 04_cancellation_rate_by_hotel.png
    │   ├── 05_room_type_bookings.png
    │   ├── 06_adr_distribution.png
    │   ├── 07_cancellation_by_lead_time.png
    │   └── 08_correlation_heatmap.png
    │
    ├── tables/
    │   ├── 01_dataset_overview.csv
    │   ├── 02_column_quality_report.csv
    │   ├── 03_cleaning_report.csv
    │   ├── 04_hotel_summary.csv
    │   ├── 05_monthly_summary.csv
    │   ├── 06_room_type_summary.csv
    │   ├── 07_top_country_summary.csv
    │   ├── 10_model_comparison.csv
    │   └── 11_best_model_summary.csv
    │
    └── models/
        └── best_cancellation_model.joblib
```

---

## 4. How to Run

Put `hotel_bookings.csv` in the same folder as the Python file.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python hotel_booking_analysis.py --input hotel_bookings.csv
```

---

## 5. Methodology

### Step 1: Load Data

The project first loads the CSV file using Pandas.

It checks:

- Number of rows
- Number of columns
- Column names
- Data types
- Missing values
- Duplicate rows

This follows the data analytics process because raw data must be understood before analysis.

---

### Step 2: Data Cleaning

The cleaning steps are:

1. Remove exact duplicate records.
2. Fill missing `company` with `0`.
3. Fill missing `agent` with `0`.
4. Fill missing `country` with `Unknown`.
5. Fill missing `children` with `0`.
6. Remove bookings where total guests equal zero.
7. Remove invalid ADR values.
8. Create a correct `arrival_date`.
9. Remove invalid date records.

Reason:

Clean data gives more reliable analysis and more accurate model results.

---

### Step 3: Feature Engineering

New features created:

| Feature | Meaning |
|---|---|
| `total_guests` | adults + children + babies |
| `total_nights` | weekend nights + week nights |
| `arrival_date` | complete booking arrival date |
| `arrival_year_month` | month-level trend period |
| `has_children` | 1 if booking has children or babies |
| `revenue_estimate` | ADR × total nights |
| `is_peak_season` | 1 for July/August peak months |

These features make the data easier to analyze and useful for modelling.

---

### Step 4: Exploratory Data Analysis

EDA is used to understand patterns in the data.

The analysis includes:

- Bookings by hotel type
- Monthly booking trend
- Monthly ADR trend
- Cancellation rate by hotel
- Bookings by room type
- ADR distribution
- Cancellation rate by lead time group
- Correlation between numeric variables

---

### Step 5: Machine Learning

The project predicts booking cancellation.

Models used:

1. Logistic Regression
2. Random Forest Classifier

The dataset is split into:

- 80% training data
- 20% testing data

Preprocessing for modelling:

- Numeric features are scaled using `StandardScaler`
- Categorical features are encoded using `OneHotEncoder`

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The best model is selected using F1 score.

Reason:

F1 score is useful because cancellation prediction needs a balance between precision and recall.

---

## 6. Visualization Outputs

The script saves these charts:

### 1. Bookings by Hotel Type

Shows which hotel type has more bookings.

Business use:

- Helps compare demand between City Hotel and Resort Hotel.

### 2. Monthly Booking Trend

Shows booking volume over time.

Business use:

- Helps identify peak months and low-demand months.

### 3. Monthly ADR Trend

Shows average price movement over time.

Business use:

- Supports dynamic pricing decisions.

### 4. Cancellation Rate by Hotel Type

Shows which hotel has higher cancellation risk.

Business use:

- Helps design cancellation control strategies.

### 5. Room Type Bookings

Shows most popular reserved room types.

Business use:

- Helps room inventory and promotion planning.

### 6. ADR Distribution

Shows hotel price distribution.

Business use:

- Helps identify normal pricing and outlier pricing.

### 7. Cancellation by Lead Time

Shows how early booking affects cancellation.

Business use:

- Helps manage long-lead bookings and deposit policy.

### 8. Correlation Heatmap

Shows relationships between numeric variables.

Business use:

- Helps identify important factors for prediction.

---

## 7. Expected Results

Expected results should be similar to the previous analysis:

- Raw dataset has around 119,390 rows.
- Clean dataset should have around 86,000+ rows after duplicate and invalid record removal.
- Random Forest is expected to perform better than Logistic Regression.
- ROC-AUC is expected to be strong because booking cancellation has clear patterns from lead time, deposit type, special requests, and customer behavior.

Exact results may change slightly depending on Python and library versions.

---

## 8. Key Insights

### Insight 1: Cancellation is a major business issue

A high number of bookings are canceled.

Business decision:

- Improve cancellation policy.
- Use deposit strategy for high-risk bookings.
- Monitor high-risk segments.

---

### Insight 2: Lead time affects cancellation

Bookings made far in advance are more likely to be canceled.

Business decision:

- Require confirmation near arrival date.
- Apply stricter conditions for very long lead-time bookings.
- Use cancellation prediction model for early warning.

---

### Insight 3: ADR changes by month

Average daily rate changes across different months.

Business decision:

- Apply dynamic pricing during high-demand months.
- Offer discounts during low-demand months.
- Increase price during peak season.

---

### Insight 4: Room type demand is not equal

Some room types are booked more than others.

Business decision:

- Promote popular room types.
- Review unpopular room pricing.
- Improve room allocation strategy.

---

### Insight 5: Random Forest is the best approach

Random Forest is expected to perform better because it can capture complex patterns.

Business decision:

- Use Random Forest for cancellation risk prediction.
- Integrate prediction score into hotel booking system.

---

## 9. Best-Performing Approach

The best-performing approach is expected to be:

> Random Forest Classifier

Reason:

- Handles non-linear relationships.
- Works well with mixed hotel booking features.
- Captures complex patterns better than Logistic Regression.
- Produces stronger F1 score and ROC-AUC.

---

## 10. Business Recommendations

1. Use dynamic pricing during peak months.
2. Monitor high-risk cancellations using machine learning.
3. Offer discounts in low-demand months.
4. Improve deposit and confirmation policy for long lead-time bookings.
5. Allocate more rooms and staff during high booking months.
6. Promote the most profitable and popular room types.
7. Use dashboards to track occupancy, ADR, revenue, and cancellation trends.

---

## 11. Conclusion

This project shows how hotel booking data can support business decisions.

The cleaned dataset is used to understand booking behavior, cancellation risk, room demand, and pricing opportunities.

The best model is selected based on performance metrics.

The final output supports hotel managers in:

- Reducing cancellations
- Improving occupancy
- Increasing revenue
- Planning staff and room inventory
- Applying dynamic pricing
