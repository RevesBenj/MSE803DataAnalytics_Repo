# Week 9 - Activity 1: Clustering using KMeans

## Project Overview

This project performs clustering analysis on the **Fitness_App_User_Data.xlsx** dataset. The main goal is to clean the fitness app user data, find useful user groups using **K-Means clustering**, and explain.

K-Means is used because the task is unsupervised machine learning. The data does not have a correct cluster label. The model groups users based on similar fitness behavior.

Official help links used:

- https://scikit-learn.org/stable/modules/clustering.html
- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

---

## Dataset Columns

| Column | Meaning |
|---|---|
| `User_ID` | Unique user number. Not used for clustering. |
| `Age` | User age. |
| `Gender` | Male or Female. Used only for interpretation. |
| `Workouts_per_Week` | How many workouts user does per week. |
| `Avg_Session_Duration_Min` | Average workout session duration in minutes. |
| `Steps_per_Day` | Average daily step count. |
| `Subscription_Type` | Free or Premium user. Used only for interpretation. |
| `Churned` | 1 means user churned, 0 means user stayed. Not used for clustering, only used after clustering to understand risk. |

---

## Data Cleaning Done

The script does these cleaning steps:

1. Loads the Excel file using pandas.
2. Checks original row count and column count.
3. Removes duplicate rows.
4. Standardises text values in `Gender` and `Subscription_Type`.
5. Converts numeric columns into correct numeric data type.
6. Checks missing values before cleaning and saves the result.
7. Fills missing numeric values with median if any exists.
8. Fills missing categorical values with mode if any exists.
9. Removes inconsistent or impossible values.
10. Saves the cleaned dataset as CSV.

### Actual Cleaning Result

| Item | Result |
|---|---:|
| Original rows | 200 |
| Duplicate rows removed | 0 |
| Invalid / inconsistent rows removed | 1 |
| Final clean rows | 199 |
| Missing values after cleaning | 0 |

One row was removed because it had invalid negative session duration. Negative session duration is not logical for fitness app data.

---

## Feature Selection

The clustering model used these features:

- `Age`
- `Workouts_per_Week`
- `Avg_Session_Duration_Min`
- `Steps_per_Day`

These features were selected because they describe user fitness behaviour and activity level.

The following columns were not used for training:

| Column | Reason |
|---|---|
| `User_ID` | It is only identifier, not real behaviour. |
| `Gender` | Categorical value, used only later for interpretation. |
| `Subscription_Type` | Categorical value, used only later for interpretation. |
| `Churned` | This is outcome-like value, so it is not used to create clusters. It is used after clustering to check churn risk. |

---

## Preprocessing for K-Means

K-Means uses distance between data points. Because of this, feature scaling is important.

Example: `Steps_per_Day` has large numbers like 10,000, while `Workouts_per_Week` has small numbers like 1 to 7. Without scaling, step count can dominate the clustering result.

The script uses:

```python
StandardScaler()
```

This makes all selected features have similar scale before applying K-Means.

---

## Choosing the Best Number of Clusters

The script tests K values from 2 to 8.

Two methods are used:

1. **Elbow Method** using inertia.
2. **Silhouette Score** to measure how well-separated clusters are.

### K Selection Result

| K | Inertia | Silhouette Score |
|---:|---:|---:|
| 2 | 627.54 | 0.2051 |
| 3 | 514.41 | 0.2108 |
| 4 | 428.79 | 0.2327 |
| 5 | 372.23 | 0.2326 |
| 6 | 340.19 | 0.2203 |
| 7 | 313.12 | 0.2156 |
| 8 | 287.37 | 0.2251 |

The selected cluster number is:

```text
K = 4
```

Reason: K = 4 has the highest silhouette score in the tested range. K = 5 is very close, but K = 4 is simpler and easier to explain, so it is better for this activity.

---

## Cluster Summary

| Cluster | User Count | Age | Workouts / Week | Avg Session Min | Steps / Day | Churn Rate | Main Insight |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 62 | 32.84 | 2.13 | 46.97 | 4,700.42 | 0.29 | Moderate / mixed activity users; medium churn risk. |
| 1 | 52 | 46.21 | 1.21 | 39.57 | 11,252.31 | 0.10 | Moderate / mixed activity users; low churn risk. |
| 2 | 48 | 28.48 | 4.65 | 52.31 | 11,209.71 | 0.06 | Highly active users; low churn risk. |
| 3 | 37 | 51.38 | 4.97 | 35.36 | 8,065.43 | 0.05 | Moderate / mixed activity users; low churn risk. |

---

## Key Findings

### Cluster 0: Medium risk group

This group has the lowest daily steps and lower workouts. Their churn rate is highest compared with other clusters. This group may need engagement support, reminders, challenges, or free-to-premium motivation.

### Cluster 1: High steps but low workouts

This group has high daily steps but low number of workouts. They may be active in walking but not doing structured workout sessions. The fitness app can recommend simple guided workouts.

### Cluster 2: Best active users

This group is younger and very active. They have high workouts, long sessions, and high steps. Their churn rate is low. This group can be targeted for premium upgrades, badges, and loyalty rewards.

### Cluster 3: Older but consistent users

This group has high workouts per week but shorter session duration. They are consistent users and have low churn risk. The app can recommend safe and balanced workout plans.

---

## Visualizations Created

The script generates these charts inside `outputs/figures/`:

| File | Purpose |
|---|---|
| `01_missing_values_before_cleaning.png` | Shows missing values before cleaning. |
| `02_elbow_method.png` | Shows inertia for each K. |
| `03_silhouette_scores.png` | Shows silhouette score and selected K. |
| `04_pca_cluster_scatter.png` | Shows clusters in 2D using PCA. |
| `05_cluster_profile.png` | Compares feature profile of every cluster. |
| `06_churn_rate_by_cluster.png` | Shows churn rate per cluster. |

---

## Project Structure

```text
Week9_KMeans_Fitness_App/
│
├── Fitness_App_User_Data.xlsx
├── fitness_kmeans_clustering.py
├── README.md
├── requirements.txt
│
└── outputs/
    ├── cleaned_fitness_app_user_data.csv
    ├── clustered_fitness_app_user_data.csv
    ├── cluster_summary.csv
    ├── kmeans_k_selection_results.csv
    ├── missing_values_before_cleaning.csv
    ├── missing_values_after_cleaning.csv
    ├── pca_cluster_coordinates.csv
    ├── preprocessing_steps.txt
    │
    └── figures/
        ├── 01_missing_values_before_cleaning.png
        ├── 02_elbow_method.png
        ├── 03_silhouette_scores.png
        ├── 04_pca_cluster_scatter.png
        ├── 05_cluster_profile.png
        └── 06_churn_rate_by_cluster.png
```

---

## How to Run

### 1. Install required packages

```bash
pip install -r requirements.txt
```

### 2. Make sure the Excel file is in the same folder

```text
Fitness_App_User_Data.xlsx
```

### 3. Run the script

```bash
python fitness_kmeans_clustering.py
```

### 4. Check results

After running, all results will be saved inside:

```text
outputs/
```

---

## Main Python Libraries Used

| Library | Use |
|---|---|
| pandas | Load, clean, and save dataset. |
| numpy | Numerical support. |
| matplotlib | Create charts. |
| scikit-learn | Scaling, PCA, KMeans, silhouette score. |
| openpyxl | Allows pandas to read Excel file. |

---

## Conclusion

This activity shows how K-Means can group fitness app users based on behaviour. The best selected number of clusters is 4. The most important insight is that Cluster 0 has the highest churn risk and lowest steps, so this group may need more engagement strategy. Cluster 2 is the strongest active group with high workouts, long session duration, high steps, and low churn.
