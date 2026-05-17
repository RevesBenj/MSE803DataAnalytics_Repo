# -------------------------------------------------------
# Week 6 A1 - IRIS Dataset Analysis using Traditional Machine Learning Model: SVM
# Author : Benjelyn Reves Patiag
# Date   : 17 May 2026
# Description:
# 1. Import required packages
# 2. Load Iris dataset from CSV
# 3. Clean and inspect the dataset
# 4. Visualise Iris dataset using scatter plot
# 5. Train and test SVM model using linear kernel
# 6. Evaluate testing dataset using accuracy, precision, recall, F1-score, and confusion matrix
# -------------------------------------------------------


import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# -----------------------------
# 1. File paths
# -----------------------------
DATA_PATH = "data/Iris.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 2. Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("IRIS DATASET LOADED")
print("=" * 60)
print(df.head())
print("\nDataset shape:", df.shape)

# -----------------------------
# 3. Clean and inspect dataset
# -----------------------------
print("\n" + "=" * 60)
print("DATA CLEANING CHECK")
print("=" * 60)

# Remove Id column because it is only an identifier, not a useful ML feature
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

# Remove duplicate rows if there are any
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates()

# Check missing values
missing_values = df.isnull().sum()

print("Missing values per column:")
print(missing_values)
print("\nDuplicate rows removed:", duplicate_count)
print("\nClean dataset shape:", df.shape)
print("\nData types:")
print(df.dtypes)
print("\nClass count:")
print(df["Species"].value_counts())

# Save cleaning summary
cleaning_summary = pd.DataFrame({
    "Column": df.columns,
    "Missing_Values": [df[col].isnull().sum() for col in df.columns],
    "Data_Type": [df[col].dtype for col in df.columns],
})
cleaning_summary.to_csv(os.path.join(OUTPUT_DIR, "cleaning_summary.csv"), index=False)

# -----------------------------
# 4. Visualise dataset
# -----------------------------
plt.figure(figsize=(8, 6))

species_list = df["Species"].unique()
for species in species_list:
    species_data = df[df["Species"] == species]
    plt.scatter(
        species_data["PetalLengthCm"],
        species_data["PetalWidthCm"],
        label=species,
        alpha=0.8
    )

plt.title("Iris Dataset Scatter Plot: Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "iris_scatter_plot.png"), dpi=300)
plt.close()

# -----------------------------
# 5. Prepare features and target
# -----------------------------
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]

# Convert text labels into numeric labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing dataset
# stratify keeps the class balance in train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# Scale features because SVM is distance-based and works better when values are standardised
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 6. Train SVM model
# -----------------------------
model = SVC(kernel="linear")
model.fit(X_train_scaled, y_train)

# -----------------------------
# 7. Predict and evaluate
# -----------------------------
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
)

cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("SVM MODEL EVALUATION RESULTS - TESTING DATASET")
print("=" * 60)
print(f"Kernel: Linear")
print(f"Testing Accuracy : {accuracy:.4f}")
print(f"Testing Precision: {precision:.4f}")
print(f"Testing Recall   : {recall:.4f}")
print(f"Testing F1-score : {f1:.4f}")
print("\nClassification Report:")
print(report)
print("Confusion Matrix:")
print(cm)

# Save metrics into text file
results_text = f"""SVM MODEL EVALUATION RESULTS - TESTING DATASET
============================================================
Model: Support Vector Machine (SVM)
Kernel: Linear
Dataset: Iris CSV Dataset
Train/Test Split: 80% training, 20% testing
Random State: 42

Testing Accuracy : {accuracy:.4f}
Testing Precision: {precision:.4f}
Testing Recall   : {recall:.4f}
Testing F1-score : {f1:.4f}

Classification Report:
{report}

Confusion Matrix:
{cm}
"""

with open(os.path.join(OUTPUT_DIR, "svm_results.txt"), "w", encoding="utf-8") as file:
    file.write(results_text)

# Save confusion matrix image
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)
disp.plot()
plt.title("SVM Linear Kernel - Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

# Save simple screenshot-style result image
fig, ax = plt.subplots(figsize=(9, 5))
ax.axis("off")
ax.text(
    0.02,
    0.95,
    results_text,
    fontsize=10,
    va="top",
    family="monospace"
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "svm_results_screenshot.png"), dpi=300)
plt.close()

print("\nOutput files saved in:", OUTPUT_DIR)
