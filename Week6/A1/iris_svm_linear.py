# -------------------------------------------------------
# Week 6 A1 - IRIS Dataset Analysis using Traditional Machine Learning Model: SVM
# Author : Benjelyn Reves Patiag
# Date   : 17 May 2026
# Description:
# 1. Load the Iris dataset from iris.zip if available, otherwise use sklearn Iris dataset.
# 2. Clean and check the dataset.
# 3. Visualise the Iris dataset using scatter plot.
# 4. Train and test Support Vector Machine (SVM) using linear kernel.
# 5. Save evaluation results and screenshots.
# -------------------------------------------------------



from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# -----------------------------
# File locations
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "iris.zip"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------
# 1. Load Iris dataset
# -----------------------------
def load_iris_dataset() -> pd.DataFrame:
    """Load Iris dataset from uploaded zip file, or fallback to sklearn dataset."""

    column_names = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]

    # Try to load from iris.zip first.
    if ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            with zip_ref.open("iris.data") as file:
                df = pd.read_csv(file, header=None, names=column_names)
        print("Dataset loaded from iris.zip")
        return df

    # If zip file is not available, load from sklearn built-in dataset.
    iris = datasets.load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    df = df.drop(columns=["target"])
    print("Dataset loaded from sklearn")
    return df


# -----------------------------
# 2. Clean dataset
# -----------------------------
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean basic issues such as missing values and duplicate rows."""

    print("\n--- Initial Dataset Info ---")
    print(df.head())
    print("\nRows and columns:", df.shape)
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # Remove empty rows if any exist.
    df = df.dropna()

    # Check duplicate rows.
    # In Iris dataset, duplicate rows can be real repeated flower measurements.
    # So we report duplicates but we do not remove them.
    duplicate_count = df.duplicated().sum()
    print("Duplicate rows found:", duplicate_count)

    # Make sure numeric columns are correct numeric type.
    numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows again if numeric conversion created missing value.
    df = df.dropna()

    print("\n--- Dataset After Cleaning ---")
    print("Rows and columns:", df.shape)
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    return df


# -----------------------------
# 3. Visualise dataset
# -----------------------------
def create_scatter_plot(df: pd.DataFrame) -> None:
    """Create scatter plot using petal length and petal width."""

    plt.figure(figsize=(8, 6))

    # Plot each species separately, so the classes are easy to see.
    for species_name in df["species"].unique():
        subset = df[df["species"] == species_name]
        plt.scatter(
            subset["petal_length"],
            subset["petal_width"],
            label=species_name,
            alpha=0.8,
        )

    plt.title("Iris Dataset Scatter Plot: Petal Length vs Petal Width")
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.legend(title="Species")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    scatter_path = OUTPUT_DIR / "iris_scatter_plot.png"
    plt.savefig(scatter_path, dpi=150)
    plt.close()

    print(f"\nScatter plot saved to: {scatter_path}")


# -----------------------------
# 4. Train and test SVM model
# -----------------------------
def train_test_svm(df: pd.DataFrame) -> None:
    """Train and test SVM model using linear kernel."""

    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = df[feature_cols]
    y = df["species"]

    # Split data into training and testing datasets.
    # stratify=y keeps each flower class balanced in train and test data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Create SVM classifier using linear kernel.
    model = SVC(kernel="linear")

    # Train the model using training data.
    model.fit(X_train, y_train)

    # Predict result using testing data.
    y_pred = model.predict(X_test)

    # Evaluate model result.
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))

    print("\n--- SVM Linear Kernel Evaluation Results ---")
    print("Testing Accuracy:", round(accuracy, 4))
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(pd.DataFrame(matrix, index=sorted(y.unique()), columns=sorted(y.unique())))

    # Save the results into text file.
    result_text = (
        "SVM Linear Kernel Evaluation Results\n"
        "====================================\n\n"
        f"Training rows: {len(X_train)}\n"
        f"Testing rows: {len(X_test)}\n"
        f"Testing Accuracy: {accuracy:.4f}\n\n"
        "Classification Report:\n"
        f"{report}\n"
        "Confusion Matrix:\n"
        f"{pd.DataFrame(matrix, index=sorted(y.unique()), columns=sorted(y.unique()))}\n"
    )

    results_path = OUTPUT_DIR / "svm_evaluation_results.txt"
    results_path.write_text(result_text, encoding="utf-8")
    print(f"\nEvaluation results saved to: {results_path}")

    # Save screenshot-like image of result summary.
    create_results_screenshot(accuracy, report, matrix, sorted(y.unique()))


# -----------------------------
# 5. Create screenshot of results
# -----------------------------
def create_results_screenshot(accuracy, report, matrix, labels) -> None:
    """Create image screenshot containing evaluation results."""

    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    text_output = (
        "SVM Linear Kernel Evaluation Results\n"
        "====================================\n\n"
        f"Testing Accuracy: {accuracy:.4f}\n\n"
        "Classification Report:\n"
        f"{report}\n"
        "Confusion Matrix:\n"
        f"{matrix_df}"
    )

    plt.figure(figsize=(10, 7))
    plt.axis("off")
    plt.text(0.01, 0.99, text_output, va="top", ha="left", family="monospace", fontsize=10)
    plt.tight_layout()

    screenshot_path = OUTPUT_DIR / "svm_results_screenshot.png"
    plt.savefig(screenshot_path, dpi=150)
    plt.close()

    print(f"Results screenshot saved to: {screenshot_path}")


# -----------------------------
# Main program
# -----------------------------
if __name__ == "__main__":
    iris_df = load_iris_dataset()
    iris_df = clean_dataset(iris_df)
    create_scatter_plot(iris_df)
    train_test_svm(iris_df)
