# User Knowledge Modeling Data Analysis activity

## 1. activity Overview

This activity analyses the **User Knowledge Modeling Dataset** from the uploaded ZIP file `user+knowledge+modeling.zip`. The dataset is an education-based classification dataset. It contains learner behaviour and exam-performance indicators, then uses these indicators to predict the learner knowledge level.

The main goal is to build a clean and reproducible data analysis activity that performs:

- data loading from the uploaded Excel workbook,
- data cleaning and preprocessing,
- exploratory data analysis,
- supervised classification,
- unsupervised clustering,
- model comparison,
- best-performing approach selection,
- output tables and figures for reporting.

The activity follows the normal data analytics workflow:

```text
Raw ZIP Dataset
      ↓
Extract Excel Workbook
      ↓
Load Training_Data and Test_Data Sheets
      ↓
Clean Columns and Labels
      ↓
Validate Missing Values and Duplicates
      ↓
EDA and Feature Understanding
      ↓
Classification and Clustering
      ↓
Model Evaluation
      ↓
Best Model and Key Insights
```

---

## 2. Dataset Description

The uploaded workbook contains three sheets:

| Sheet | Purpose |
|---|---|
| Information | Dataset metadata and attribute explanation |
| Training_Data | Main training dataset |
| Test_Data | Hold-out test dataset |

After cleaning, the dataset contains:

| Item | Value |
|---|---:|
| Training rows | 258 |
| Test rows | 145 |
| Total clean rows | 403 |
| Input features | 5 |
| Target variable | UNS |
| Missing values after cleaning | 0 |
| Duplicate rows after cleaning | 0 |

---

## 3. Variables Used

| Column | Meaning | Role |
|---|---|---|
| STG | Degree of study time for goal object materials | Input feature |
| SCG | Degree of repetition number for goal object materials | Input feature |
| STR | Degree of study time for related objects | Input feature |
| LPR | Exam performance for related objects | Input feature |
| PEG | Exam performance for goal objects | Input feature |
| UNS | User knowledge level | Target label |

The target label `UNS` has four classes:

| Knowledge Level | Count |
|---|---:|
| very_low | 50 |
| low | 129 |
| middle | 122 |
| high | 102 |

---

## 4. Data Loading and Preprocessing Focus

### 4.1 Loading Process

The original file is a legacy `.xls` workbook inside a ZIP file. The script extracts the ZIP file, converts the `.xls` file to `.xlsx` if required, then reads the `Training_Data` and `Test_Data` sheets.

### 4.2 Cleaning Process

The cleaning process includes:

1. Strip extra spaces from column names.
2. Keep only the six useful columns: `STG`, `SCG`, `STR`, `LPR`, `PEG`, and `UNS`.
3. Remove irrelevant workbook columns such as `Unnamed` and `Attribute Information`.
4. Convert feature columns to numeric values.
5. Standardise label values:
   - `Very Low`
   - `very_low`
   - `very low`
   are converted to one consistent value: `very_low`.
6. Check missing values.
7. Drop rows with invalid numeric or target data.
8. Remove duplicate rows.
9. Save clean data files into the `output/` folder.

### 4.3 Clean Data Result

The dataset was already high quality. After preprocessing:

- no missing values were found,
- no duplicate rows were found,
- all five input variables were numeric,
- class labels were standardised,
- training and test split was preserved from the original workbook.

---

## 5. Exploratory Data Analysis

EDA was used to understand the dataset before modelling.

### 5.1 Class Balance

The dataset is moderately imbalanced. The `low` and `middle` classes have more samples than `very_low`. Because of this, weighted precision, weighted recall, and weighted F1-score are used together with accuracy.

### 5.2 Feature Profile by Knowledge Level

Average feature values by `UNS` class:

| UNS | STG | SCG | STR | LPR | PEG |
|---|---:|---:|---:|---:|---:|
| high | 0.407 | 0.430 | 0.510 | 0.543 | 0.800 |
| low | 0.327 | 0.323 | 0.425 | 0.449 | 0.254 |
| middle | 0.375 | 0.367 | 0.491 | 0.386 | 0.531 |
| very_low | 0.259 | 0.262 | 0.354 | 0.269 | 0.096 |

Main insight: `PEG`, the exam performance for goal objects, is the strongest practical separator between knowledge levels. High-knowledge users have the highest average `PEG`, while very-low users have the lowest average `PEG`.

### 5.3 Correlation Summary

| Pair | Correlation | Interpretation |
|---|---:|---|
| STG and PEG | 0.199 | Weak positive relationship |
| SCG and PEG | 0.194 | Weak positive relationship |
| STR and PEG | 0.148 | Weak positive relationship |
| LPR and PEG | -0.039 | Almost no linear relationship |

The correlations are mostly weak. This means the target class is not explained by one simple linear relationship only. Non-linear classification models are suitable.

---

## 6. Classification Methodology

Classification was used because the dataset has a known target label: `UNS`.

The following models were trained using the original training sheet and evaluated using the original test sheet:

1. Logistic Regression
2. K-Nearest Neighbours
3. Support Vector Machine with RBF kernel
4. Decision Tree
5. Random Forest
6. Gradient Boosting
7. MLP Neural Network

Evaluation metrics used:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score

Weighted metrics are important because the class distribution is not perfectly balanced.

---

## 7. Classification Results

| Rank | Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
|---:|---|---:|---:|---:|---:|
| 1 | MLP Neural Network | 0.9586 | 0.9589 | 0.9586 | 0.9585 |
| 2 | Gradient Boosting | 0.9172 | 0.9179 | 0.9172 | 0.9173 |
| 3 | Decision Tree | 0.9172 | 0.9172 | 0.9172 | 0.9172 |
| 4 | Logistic Regression | 0.9103 | 0.9173 | 0.9103 | 0.9076 |
| 5 | SVM RBF | 0.9034 | 0.9260 | 0.9034 | 0.9028 |
| 6 | Random Forest | 0.9034 | 0.9093 | 0.9034 | 0.9028 |
| 7 | KNN | 0.7862 | 0.8322 | 0.7862 | 0.7762 |

---

## 8. Best-Performing Classification Approach

The best model is the **MLP Neural Network**.

It achieved:

| Metric | Score |
|---|---:|
| Accuracy | 95.86% |
| Weighted Precision | 95.89% |
| Weighted Recall | 95.86% |
| Weighted F1-score | 95.85% |

### Why MLP Performed Best

The dataset has weak simple correlations between variables, so the relationship between features and knowledge level is likely non-linear. MLP can learn non-linear decision boundaries. This helps it separate similar classes such as `low` and `middle` better than simpler methods.

---

## 9. Clustering Methodology

Clustering was also performed to check whether natural groups exist without using the `UNS` target label.

The following clustering methods were tested:

1. K-Means Clustering
2. Agglomerative Clustering

The features were standardised using Z-score scaling before clustering. This is important because distance-based clustering can be affected by feature scale.

Clustering was evaluated using:

- Silhouette Score
- Adjusted Rand Index
- Normalized Mutual Information

---

## 10. Clustering Results

| Best Cluster Model | k | Silhouette | Adjusted Rand Index | Normalized Mutual Information |
|---|---:|---:|---:|---:|
| K-Means | 5 | 0.1827 | 0.1218 | 0.1923 |

### Clustering Interpretation

K-Means with 5 clusters gave the highest silhouette score, but the score is still low. This means the dataset does not naturally separate into strong unsupervised clusters. Clustering is useful for exploration, but it is not the best final approach for this dataset.

---

## 11. Classification vs Clustering Comparison

| Approach | Result | Suitability |
|---|---|---|
| Classification | MLP reached 95.85% weighted F1-score | Best approach |
| Clustering | Best silhouette only 0.1827 | Useful for exploration only |

The best-performing approach is **supervised classification**, especially the **MLP Neural Network**, because the dataset already has reliable labels and the model predicts those labels very accurately.

---

## 12. Key Insights Discovered

1. The data is clean and ready after minor preprocessing.
2. `PEG` is the strongest practical indicator of user knowledge level.
3. Very-low knowledge users have much lower `PEG`, `STG`, `SCG`, and `LPR` averages.
4. High knowledge users show the highest average `PEG` and generally stronger study and repetition values.
5. Correlations are weak, so one simple linear rule is not enough.
6. Non-linear models perform better.
7. MLP Neural Network is the best-performing model.
8. Clustering gives weak natural group separation, so clustering should only support exploratory insight.
9. The original train/test split is useful and should be preserved for fair testing.
10. The activity can support educational decision-making by identifying likely learner knowledge levels from study behaviour and assessment performance.

---

## 13. activity Structure

```text
ukm_activity/
│
├── README.md
├── requirements.txt
├── user_knowledge_analysis.py
│
├── data/
│   └── user+knowledge+modeling.zip
│
├── output/
│   ├── analysis_summary.json
│   ├── best_classification_report.txt
│   ├── clean_full_data.csv
│   ├── clean_test_data.csv
│   └── clean_training_data.csv
│
├── tables/
│   ├── best_cluster_profile.csv
│   ├── best_confusion_matrix.csv
│   ├── class_distribution.csv
│   ├── classification_model_comparison.csv
│   ├── clustering_model_comparison.csv
│   ├── correlation_matrix.csv
│   ├── descriptive_statistics.csv
│   └── knowledge_level_feature_profile.csv
│
└── figures/
    ├── best_cluster_pca.png
    ├── best_confusion_matrix.png
    ├── class_distribution.png
    ├── correlation_heatmap.png
    └── feature_profile_by_class.png
```

---

## 14. How to Run the activity

### Step 1: Create a virtual environment

```bash
python -m venv .venv
```

### Step 2: Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

### Step 3: Install requirements

```bash
pip install -r requirements.txt
```

### Step 4: Run the analysis

```bash
python user_knowledge_analysis.py
```

---

## 15. Important Note About `.xls` File

The uploaded dataset uses an older Excel `.xls` format. Some Python environments need `xlrd` to read `.xls` files. This activity avoids that issue by converting the `.xls` file to `.xlsx` using LibreOffice if needed.

If LibreOffice is not installed, install `xlrd` instead:

```bash
pip install xlrd
```

---

## 16. Methodology Summary

The activity follows these steps:

1. Load the ZIP file.
2. Extract the Excel file.
3. Load training and test sheets.
4. Remove non-data columns.
5. Clean feature columns.
6. Standardise target labels.
7. Validate missing and duplicate records.
8. Save clean datasets.
9. Perform EDA.
10. Train classification models.
11. Evaluate classification models.
12. Run clustering models.
13. Compare all results.
14. Select the best approach.
15. Export final tables and figures.

---

## 17. Final Conclusion

The dataset is suitable for classification because it has clear labels for user knowledge level. After cleaning and preprocessing, several machine learning models were compared. The **MLP Neural Network** gave the best result with about **95.86% accuracy** and **95.85% weighted F1-score**.

Clustering was tested, but it produced weak separation. The best clustering result was K-Means with `k=5`, but the silhouette score was only **0.1827**. Therefore, clustering is useful only for exploratory understanding.

The final recommended approach is **supervised classification using MLP Neural Network**.

