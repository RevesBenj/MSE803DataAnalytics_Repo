# Credit Card Fraud Detection using SVM

## 1. Dataset and Problem Statement

This project performs a **classification task for fraud detection** using the `creditcard.csv` dataset.

The dataset contains credit card transaction records. The goal is to classify each transaction as:

- `0` = Normal transaction
- `1` = Fraud transaction

This is a **supervised machine learning classification problem** because the dataset already has a known target label called `Class`.

Fraud detection is important because fraudulent transactions are rare but high risk. In this dataset, the fraud records are much smaller than normal records, so the data is **highly imbalanced**. Because of this, accuracy alone is not enough. Precision, recall, F1-score, confusion matrix, ROC-AUC, and PR-AUC are also used.

---

## 2. Project Files

```text
Folder/
│
├── W7A2_FraudDetection_SVM.py
├── README.md
├── requirements.txt
│
└── outputs/
    ├── cleaning_summary.json
    │
    ├── staging/
    │   └── creditcard_cleaned_staging.csv
    ├── descriptive_statistics.csv
    ├── data_types.csv
    ├── missing_values.csv
    ├── model_metrics.json
    ├── classification_report.txt
    ├── confusion_matrix.csv
    ├── svm_fraud_model.joblib
    │
    └── figures/
        ├── class_distribution.png
        ├── amount_distribution.png
        ├── amount_by_class_boxplot.png
        ├── correlation_heatmap.png
        ├── top_target_correlations.png
        └── confusion_matrix.png
```

---

## 3. Technologies and Libraries Used

The project uses:

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

These tools are used for data loading, data cleaning, EDA, model training, model testing, visualization, and saving the trained model.

---

## 4. How to Run the Project

### Step 1: Install required libraries

```bash
pip install -r requirements.txt
```

### Step 2: Run the Python script

```bash
python W7A2_FraudDetection_SVM.py --data-path creditcard.csv --output-dir outputs
```

If the dataset is stored in another folder, update the path:

```bash
python W7A2_FraudDetection_SVM.py --data-path /path/to/creditcard.csv --output-dir outputs
```

---

## 5. Structure of the Code

### Main Classes

| Class Name | Purpose |
|---|---|
| `ProjectConfig` | Stores project settings |
| `DataLoader` | Loads the CSV dataset |
| `DataCleaner` | Handles missing values and duplicate records |
| `ExploratoryDataAnalysis` | Creates summary files and visual charts |
| `FeatureEngineer` | Separates features and target, then performs train-test split |
| `SVMFraudModel` | Builds, trains, predicts, and saves the SVM model |
| `ModelEvaluator` | Calculates model performance metrics |
| `FraudDetectionPipeline` | Runs the full end-to-end workflow |

---

## 6. Data Preprocessing and Cleaning Steps

The script performs the following cleaning steps:

### 6.1 Load Dataset

The dataset is loaded using Pandas.

```python
df = pd.read_csv(data_path)
```

### 6.2 Check Missing Values

The script checks missing values in all columns.

Output file:

```text
outputs/missing_values.csv
```

Result from this run:

```text
Missing values before cleaning: none
Missing values after cleaning: none
```

### 6.3 Remove Duplicate Records

Duplicate rows are removed because duplicate records can affect model training.

Result from this run:

```text
Original rows: 284807
Duplicates removed: 1081
Final rows after cleaning: 283726
```

### 6.4 Handle Missing Values

If missing numeric values exist, the script fills them using the median.

Reason:

Median is better than mean when the dataset may contain outliers.

### 6.5 Save Cleaned / Staging Dataset

The updated script now saves the cleaned dataset after removing duplicates and handling missing values.

Output file:

```text
outputs/staging/creditcard_cleaned_staging.csv
```

This staging file is the exact cleaned dataset used for EDA, feature preparation, training, and testing. This is useful for checking the cleaning result and for submitting evidence of preprocessing.

### 6.6 Feature Selection

The target column is:

```text
Class
```

The input features are all other columns:

```text
Time, V1, V2, ..., V28, Amount
```

### 6.7 Encoding

No categorical encoding was required because all columns are numeric.

### 6.8 Normalization / Scaling

The script uses `StandardScaler`.

Reason:

SVM is sensitive to feature scale. `Amount` and `Time` have different scale compared with PCA-style features like `V1` to `V28`.

---

## 7. Exploratory Data Analysis and Visualizations

EDA was completed before model training. The script creates summary reports and visual charts.

### 7.1 Descriptive Statistics

Saved here:

```text
outputs/descriptive_statistics.csv
```

This file shows mean, standard deviation, minimum, maximum, and quartile values.

### 7.2 Class Distribution

Saved here:

```text
outputs/figures/class_distribution.png
```

This chart shows that normal transactions are much higher than fraud transactions.

### 7.3 Transaction Amount Distribution

Saved here:

```text
outputs/figures/amount_distribution.png
```

This chart shows how transaction amounts are distributed.

### 7.4 Amount by Class Boxplot

Saved here:

```text
outputs/figures/amount_by_class_boxplot.png
```

This chart compares transaction amount between normal and fraud records.

### 7.5 Correlation Heatmap

Saved here:

```text
outputs/figures/correlation_heatmap.png
```

This chart shows relationship between numeric features.

### 7.6 Top Features Correlated with Fraud

Saved here:

```text
outputs/figures/top_target_correlations.png
```

This chart shows the top features most related to the fraud target.

---

## 8. Machine Learning Model Development

The model used is:

```text
Support Vector Machine using LinearSVC
```

`LinearSVC` is used because the dataset is large. It is more efficient than standard kernel SVM for large tabular datasets.

The model pipeline:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(class_weight="balanced"))
])
```

### Why `class_weight="balanced"` is used

Fraud records are very small compared with normal records. Without balancing, the model may mostly predict normal transactions only.

Using `class_weight="balanced"` gives more importance to the minority fraud class.

---

## 9. Training and Testing Process

The dataset is split into:

- 70% training data
- 30% testing data

The split uses `stratify=y`.

Reason:

Stratification keeps the fraud and normal ratio similar in both training and testing datasets.

Result from this run:

```text
Training records: 198,608
Testing records: 85,118
```

---

## 10. Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC / Average Precision
- Confusion Matrix

### 10.1 Results from This Run

| Metric | Score |
|---|---:|
| Accuracy | 0.9792 |
| Precision | 0.0670 |
| Recall | 0.8873 |
| F1-Score | 0.1246 |
| ROC-AUC | 0.9680 |
| PR-AUC / Average Precision | 0.6458 |

### 10.2 Confusion Matrix

| Actual / Predicted | Predicted Normal | Predicted Fraud |
|---|---:|---:|
| Actual Normal | 83221 | 1755 |
| Actual Fraud | 16 | 126 |

Confusion matrix image:

```text
outputs/figures/confusion_matrix.png
```

---

## 11. Discussion of Results and Findings

The model got high accuracy, but dataset is highly imbalanced so accuracy alone is not enough.

Recall score is high, which means most fraud transactions were detected successfully.

Precision is low because some normal transactions were also predicted as fraud.

In fraud detection, this is still useful because missing fraud is more risky than checking extra suspicious transactions.


The confusion matrix shows:

- Most normal transactions were correctly predicted as normal.
- Most fraud transactions were correctly detected.
- Some normal transactions were wrongly flagged as fraud.
- A small number of fraud transactions were missed.

This result shows that SVM can detect fraud patterns, but the model still needs improvement to reduce false alarms.

---

## 12. Conclusion

This project successfully completed the full machine learning workflow for fraud detection.

The work included:

- Dataset loading
- Data cleaning
- Missing value checking
- Duplicate removal
- Cleaned/staging dataset generation
- Feature selection
- Scaling
- Exploratory Data Analysis
- SVM model training
- Model testing
- Evaluation using several metrics
- Saving charts, reports, and trained model

The SVM model performed well in detecting many fraud records, especially based on recall. But the low precision shows that more work is needed to reduce false positive predictions.

---

