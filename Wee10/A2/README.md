# Heart Disease Risk Analysis and Prediction Using Data Analytics and Machine Learning

## 1. Activity Overview

This Activity uses the uploaded **UCI Heart Disease dataset ZIP file** to analyse patient clinical data and predict whether a patient has heart disease risk. 

The Activity follows this complete data analytics workflow:

```text
Load dataset from data/heart+disease.zip
      ↓
Read processed.cleveland.data from the ZIP
      ↓
Data Cleaning and Preprocessing
      ↓
Exploratory Data Analysis (EDA)
      ↓
Data Visualization
      ↓
Machine Learning Model Training
      ↓
Model Evaluation
      ↓
Sample Patient Prediction
      ↓
Output Reports and Charts
```

The Activity supports **descriptive analytics**, **diagnostic analytics**, and **predictive analytics**. It describes the dataset, investigates patterns, and predicts heart disease presence.

---

## 2. Dataset Source

Dataset: **UCI Heart Disease Dataset**  
Dataset file used in this Activity:

```text
data/heart+disease.zip
```

File read inside the ZIP:

```text
processed.cleveland.data
```

The code does **not need to download the dataset from the internet**. It reads directly from the uploaded ZIP file.

The dataset contains clinical features such as:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG result
- Maximum heart rate
- Exercise-induced angina
- ST depression / oldpeak
- Slope
- Number of major vessels
- Thalassemia result
- Target diagnosis

The original target values are:

```text
0 = no heart disease
1, 2, 3, 4 = heart disease presence
```

In this Activity, the target is converted into binary classification:

```text
0 = No heart disease
1 = Heart disease present
```

---

## 3. Activity Structure

```text
heart_disease_Activity/
│
├── data/
│   └── heart+disease.zip
│
├── output/
│   ├── cleaned_heart_disease_dataset.csv
│   ├── dataset_summary.csv
│   ├── descriptive_statistics.csv
│   ├── correlation_with_target.csv
│   ├── model_metrics.csv
│   ├── sample_patient_prediction.csv
│   ├── classification_report_*.txt
│   ├── confusion_matrix_*.csv
│   ├── 01_target_distribution.png
│   ├── 02_age_vs_disease_boxplot.png
│   ├── 03_chest_pain_vs_disease.png
│   ├── 04_correlation_heatmap.png
│   └── 05_model_comparison.png
│
├── src/
│   └── heart_disease_analysis.py
│
├── requirements.txt
└── README.md
```

---

## 4. Code Design

The source code is divided into clear classes.

### `ActivityConfig`
Stores Activity settings such as:

- Dataset ZIP path
- File name inside ZIP
- Output folder
- Test size
- Random state
- Target column name

### `HeartDiseaseDataLoader`
Loads the dataset from:

```text
data/heart+disease.zip
```

It opens the ZIP file and reads:

```text
processed.cleveland.data
```

It also assigns the correct column names because the raw UCI file does not include headers.

### `HeartDiseaseCleaner`
Cleans the dataset by:

- Standardising column names
- Replacing `?` with missing values
- Converting all columns to numeric type
- Removing duplicate rows
- Converting target values into binary classes

### `HeartDiseaseEDA`
Creates summary outputs:

- Dataset summary
- Descriptive statistics
- Correlation with target

### `HeartDiseaseVisualizer`
Creates and saves visualizations in the output folder.

### `HeartDiseaseModelTrainer`
Handles:

- Train-test split
- Missing value imputation
- Feature scaling
- One-hot encoding
- Model training
- Model evaluation
- Sample patient prediction

### `HeartDiseaseAnalysisApp`
Main controller class that runs the full Activity workflow.

---

## 5. Development Process

### Step 1: Load Dataset

The dataset is loaded from the local ZIP file using Python `zipfile`.

```python
with zipfile.ZipFile("data/heart+disease.zip", "r") as zip_file:
    with zip_file.open("processed.cleveland.data") as data_file:
        df = pd.read_csv(data_file, header=None, names=column_names, na_values="?")
```

This is better for submission because the Activity can run offline.

---

### Step 2: Data Cleaning

Cleaning steps:

- Replace `?` values with NaN
- Convert all columns to numeric format
- Remove duplicated rows
- Convert the target to binary class
- Use median imputation for numerical missing values
- Use most frequent imputation for categorical missing values

This is important because missing or dirty data can produce wrong analysis and poor model performance.

---

### Step 3: Exploratory Data Analysis

EDA helps understand the dataset before modelling.

The Activity creates:

- Dataset summary
- Descriptive statistics
- Correlation analysis
- Charts

EDA is useful because it shows patterns, outliers, missing values, and possible relationships between features and heart disease.

---

## 6. Visualizations

The Activity saves five main visualizations inside the `output` folder.

### 1. Heart Disease Distribution

File:

```text
output/01_target_distribution.png
```

Purpose:

Shows how many patients have heart disease and how many do not.

Rationale:

This checks whether the dataset is balanced or imbalanced.

---

### 2. Age vs Heart Disease Boxplot

File:

```text
output/02_age_vs_disease_boxplot.png
```

Purpose:

Shows how patient age is distributed between disease and no disease groups.

Rationale:

Age is a common health risk factor, so this graph helps compare age patterns between groups.

---

### 3. Chest Pain Type vs Heart Disease

File:

```text
output/03_chest_pain_vs_disease.png
```

Purpose:

Shows how chest pain type is related to heart disease status.

Rationale:

Chest pain is clinically important, so this graph helps identify which chest pain group has more disease cases.

---

### 4. Correlation Heatmap

File:

```text
output/04_correlation_heatmap.png
```

Purpose:

Shows correlation between numerical features.

Rationale:

This helps identify variables that move together and variables that may be related to heart disease.

Important note:

Correlation does not mean causation. A strong relationship does not automatically mean one variable causes another.

---

### 5. Model Comparison Chart

File:

```text
output/05_model_comparison.png
```

Purpose:

Compares machine learning models using Accuracy, Precision, Recall, and F1 Score.

Rationale:

This makes it easier to select the best model based on performance, not only assumption.

---

## 7. Machine Learning Models Used

This is a classification problem because the target is categorical:

```text
0 = No disease
1 = Heart disease
```

The Activity trains four supervised learning models:

### 1. Logistic Regression
Simple and easy to interpret baseline model.

### 2. Decision Tree
Easy to understand and explain using decision rules.

### 3. Random Forest
Combines many decision trees to improve prediction stability.

### 4. Support Vector Machine
Useful classification model that can handle non-linear patterns.

---

## 8. Model Evaluation Metrics

The Activity evaluates models using:

### Accuracy
Measures the percentage of correct predictions.

### Precision
Measures how many predicted heart disease cases were actually correct.

### Recall
Measures how many real heart disease cases were correctly detected.

### F1 Score
Combines precision and recall into one balanced score.

In healthcare prediction, recall and F1 score are important because missing a disease case can be serious.

---

## 9. Expected Results

When the script runs successfully, it creates these result files:

```text
output/model_metrics.csv
output/sample_patient_prediction.csv
```

The best model is selected using the highest F1 Score.

Expected important features may include:

- Chest pain type
- Exercise-induced angina
- Oldpeak
- Maximum heart rate
- Number of major vessels
- Thalassemia result

The exact model result may be slightly different depending on package versions and the train-test split.

---

## 10. Sample Patient Prediction

The script includes one sample patient prediction.

Example input:

```text
Age: 55
Sex: Male
Chest pain type: 4
Resting blood pressure: 140
Cholesterol: 250
Maximum heart rate: 135
Exercise-induced angina: Yes
Oldpeak: 2.3
```

The model predicts:

```text
0 = No Heart Disease Risk
1 = Heart Disease Risk
```

Important: This output is for learning only. It is not medical advice.

---

## 11. Rationale Behind the Solution

This solution is suitable because:

- The dataset is healthcare-related and meaningful.
- The target variable is suitable for classification.
- EDA helps understand the data before modelling.
- Visualizations make the result easier to understand.
- Multiple models allow fair comparison.
- The output folder saves all results for reporting.
- Local ZIP loading makes the Activity reproducible and offline-ready.

This follows a proper data analytics workflow where data is collected, cleaned, analysed, visualized, modelled, and interpreted.

---

## 12. How to Run the Activity

### Step 1: Check dataset file

Make sure this file exists:

```text
data/heart+disease.zip
```

### Step 2: Install requirements

```bash
pip install -r requirements.txt
```

### Step 3: Run the script

From the Activity root folder:

```bash
python src/heart_disease_analysis.py
```

You can also run it from another folder because the code now uses the script location to find the Activity root.

### Step 4: Check output folder

All result files and charts will be saved in:

```text
output/
```

---

## 13. Ethical Note

This Activity uses a public dataset for education. Healthcare prediction must be handled carefully because data decisions can affect people.

Good practice:

- Protect patient privacy
- Avoid unfair bias
- Explain model limitations
- Use results only as decision support
- Do not treat predictions as final medical diagnosis

This Activity is not medical advice and should not replace doctors or clinical diagnosis.

---

## 14. Conclusion

This Activity shows how data analytics and machine learning can support heart disease risk analysis. The process includes data loading from ZIP, data cleaning, EDA, visualization, model training, model evaluation, and prediction.

The best model is chosen based on F1 Score. This makes the solution more reliable for a healthcare classification problem.
