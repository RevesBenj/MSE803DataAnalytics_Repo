# MSE803 Assessment 1: Output to Task Mapping
**Generated:** 2026-06-13 11:52:05

This document shows which generated output file supports each assessment task.

---

## Complete Output File Inventory

### Cleaned Data

| Output | Description |
|--------|-------------|
| `outputs/cleaned_data/cleaned_avon_river_data.xlsx` | Cleaned Avon River dataset in Excel format for marker review and Power BI/Excel import. |
| `outputs/cleaned_data/cleaned_avon_river_data.csv` | Cleaned Avon River dataset in CSV format for reproducible Python/BI use. |

### Main Reports

| Output | Description |
|--------|-------------|
| `outputs/reports/descriptive_statistics.csv` | Descriptive statistics, skewness, and kurtosis for numeric variables. |
| `outputs/reports/site_summary.csv` | Site-level summary for AV-1, AV-2, and AV-3. |
| `outputs/reports/species_summary.csv` | Species-level fish count and average size summary. |
| `outputs/reports/correlation_matrix.csv` | Pearson correlation matrix for water quality and fish population variables. |
| `outputs/reports/regression_results.txt` | Multiple linear regression metrics, coefficients, OLS summary, and limitations. |
| `outputs/reports/recommendations.txt` | Evidence-based environmental recommendations generated from analysis results. |
| `outputs/reports/assessment_mapping.md` | This task-to-output mapping file. |
| `outputs/reports/advanced_ml_conclusion_and_limitations.txt` | Advanced ML technical conclusions, limitations, and ethics notes. |

### Figures

| Output | Description |
|--------|-------------|
| `outputs/figures/fig01_histograms.png` | Univariate histograms for key variables. |
| `outputs/figures/fig02_boxplots.png` | Boxplots for spread and outlier detection. |
| `outputs/figures/fig03_correlation_heatmap.png` | Pearson correlation heatmap. |
| `outputs/figures/fig04_scatter_do_vs_fishcount.png` | Dissolved oxygen vs fish count scatter/regression chart. |
| `outputs/figures/fig05_scatter_temp_vs_do.png` | Temperature vs dissolved oxygen scatter/regression chart. |
| `outputs/figures/fig06_scatter_ph_vs_fishcount.png` | pH vs fish count scatter/regression chart. |
| `outputs/figures/fig07_regression_plot.png` | Actual vs predicted and residual plot for multiple linear regression. |
| `outputs/figures/fig08_pairplot.png` | Multivariate pairplot for numeric variables. |
| `outputs/figures/fig09_site_comparison.png` | Site comparison for WQI, FHI, DO, and fish count. |
| `outputs/figures/fig10_species_distribution.png` | Species distribution and average size chart. |
| `outputs/figures/fig11_monthly_trends.png` | Monthly trend analysis for DO, fish count, and WQI. |
| `outputs/figures/fig12_executive_dashboard.png` | Executive dashboard showing river health overview. |
| `outputs/figures/fig13_kmeans_elbow_method.png` | K-Means elbow method chart. |
| `outputs/figures/fig14_kmeans_silhouette_score.png` | K-Means silhouette score chart. |
| `outputs/figures/fig15_pca_kmeans_clusters.png` | PCA 2D visualisation of K-Means clusters. |
| `outputs/figures/fig16_classification_model_comparison.png` | Classification model comparison by macro F1-score. |
| `outputs/figures/fig17_time_series_trend_forecast.png` | Simple monthly time-series trend and rolling forecast. |

### Advanced ML Tables

| Output | Description |
|--------|-------------|
| `outputs/tables/ml_created_class_target.csv` | Created Water_Quality_Category labels from Water_Quality_Index for classification. |
| `outputs/tables/ml_linear_regression_metrics.csv` | Linear regression metrics for Fish_Count prediction. |
| `outputs/tables/ml_kmeans_elbow_silhouette_scores.csv` | K values, inertia, and silhouette scores for K-Means selection. |
| `outputs/tables/ml_kmeans_clustered_records.csv` | Cleaned records with assigned KMeans_Cluster labels. |
| `outputs/tables/ml_pca_2d_cluster_points.csv` | PCA1 and PCA2 coordinates used for the cluster visualisation. |
| `outputs/tables/ml_kmeans_cluster_summary_table.csv` | Mean pH, temperature, DO, fish count, and unavailable nutrient fields per cluster. |
| `outputs/tables/ml_classification_model_comparison.csv` | Accuracy, precision, recall, F1-score, and confusion matrix summary for all classifiers. |
| `outputs/tables/ml_confusion_matrix_logistic_regression.csv` | Confusion matrix for Logistic Regression. |
| `outputs/tables/ml_confusion_matrix_svm.csv` | Confusion matrix for SVM. |
| `outputs/tables/ml_confusion_matrix_decision_tree.csv` | Confusion matrix for Decision Tree. |
| `outputs/tables/ml_confusion_matrix_random_forest.csv` | Confusion matrix for Random Forest. |
| `outputs/tables/ml_confusion_matrix_gradient_boosting.csv` | Confusion matrix for Gradient Boosting. |
| `outputs/tables/ml_confusion_matrix_neural_network___mlp.csv` | Confusion matrix for Neural Network / MLP. |
| `outputs/tables/ml_time_series_monthly_trend_forecast.csv` | Monthly average fish count, WQI, and simple rolling forecast. |
---

## Task 1-A: Problem Identification

**Challenge 1: Poor water quality can reduce fish health and fish population**

| Output | Description |
|--------|-------------|
| `outputs/figures/fig03_correlation_heatmap.png` | Shows correlation between DO, pH, temperature, and fish count. |
| `outputs/figures/fig04_scatter_do_vs_fishcount.png` | Visualises direct relationship between DO and fish count. |
| `outputs/figures/fig05_scatter_temp_vs_do.png` | Shows how temperature relates to dissolved oxygen. |
| `outputs/figures/fig12_executive_dashboard.png` | Gives an overview of river health and possible problem areas. |
| `outputs/reports/correlation_matrix.csv` | Quantitative evidence of water quality-fish relationships. |
| `outputs/reports/site_summary.csv` | Compares water quality and fish indicators across three sites. |

**Challenge 2: Some sites/species show weaker fish population indicators**

| Output | Description |
|--------|-------------|
| `outputs/figures/fig09_site_comparison.png` | Compares WQI and FHI across AV-1, AV-2, and AV-3. |
| `outputs/figures/fig10_species_distribution.png` | Shows species fish count and average size patterns. |
| `outputs/figures/fig11_monthly_trends.png` | Shows time trend of fish count over Oct-Dec 2023. |
| `outputs/reports/species_summary.csv` | Provides species-level variation in fish count and fish size. |
| `outputs/tables/ml_kmeans_cluster_summary_table.csv` | Adds cluster-level evidence for hidden site/record patterns. |

---

## Task 1-B: Data Analysis Techniques

**Technique 1: Pearson Correlation Analysis**

| Output | Description |
|--------|-------------|
| `outputs/reports/correlation_matrix.csv` | Full Pearson correlation matrix for 7 variables. |
| `outputs/figures/fig03_correlation_heatmap.png` | Visual heatmap of all correlation values. |
| `outputs/figures/fig04_scatter_do_vs_fishcount.png` | Bivariate scatter showing correlation direction. |
| `outputs/figures/fig05_scatter_temp_vs_do.png` | Bivariate scatter showing temperature and DO relationship. |
| `outputs/figures/fig06_scatter_ph_vs_fishcount.png` | Bivariate scatter showing pH and fish count relationship. |

**Technique 2: Multiple Linear Regression**

| Output | Description |
|--------|-------------|
| `outputs/reports/regression_results.txt` | Coefficients, R², MAE, RMSE, OLS summary, and limitations. |
| `outputs/figures/fig07_regression_plot.png` | Actual vs predicted chart and residual plot. |
| `outputs/tables/ml_linear_regression_metrics.csv` | Extra ML metric table for Fish_Count prediction. |

**Extra Lecture-Based ML Techniques Added**

| Output | Description |
|--------|-------------|
| `outputs/tables/ml_classification_model_comparison.csv` | Compares Logistic Regression, SVM, Decision Tree, Random Forest, Gradient Boosting, and MLP. |
| `outputs/tables/ml_kmeans_elbow_silhouette_scores.csv` | Supports K-Means cluster selection. |
| `outputs/tables/ml_time_series_monthly_trend_forecast.csv` | Supports simple predictive trend analysis. |

---

## Task 1-C: Tools & Visualisation

**Tool 1: Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, Statsmodels)**

Python is used for data loading, cleaning, preprocessing, EDA, statistical analysis, visualisation, machine learning, and report/table export.

**Tool 2: Power BI / Excel (recommended for stakeholder dashboard)**

| Output | Description |
|--------|-------------|
| `outputs/cleaned_data/cleaned_avon_river_data.xlsx` | Can be imported into Power BI or Excel. |
| `outputs/cleaned_data/cleaned_avon_river_data.csv` | Can be imported into BI tools or reused in other analytics workflows. |
| `outputs/reports/site_summary.csv` | Good table for dashboard KPI cards by site. |
| `outputs/reports/species_summary.csv` | Good table for dashboard species filtering. |
| `outputs/tables/ml_kmeans_cluster_summary_table.csv` | Good table for cluster-based stakeholder explanation. |

---

## Task 1-D: Recommendations

| Recommendation | Assessment Alignment |
|----------------|----------------------|
| [R1] Prioritise Water Quality Protection at AV-3 | Task 1-A (poor water quality), Task 1-D |
| [R2] Increase Monitoring Frequency and Add Missing Environmental Variables | Task 1-A (poor fish population evidence), Task 1-D |
| [R3] Investigate Fish Health Concern at AV-1 | Task 1-A (fish population challenge), Task 1-D |
| [R4] Use Temperature and pH Alerts as Precautionary Controls | Task 1-D |
| [R5] Deploy Power BI / Excel Dashboard for Non-Technical Stakeholders | Task 1-C (Tool 2: Power BI / Interactive dashboard) |

---