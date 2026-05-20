# -------------------------------------------------------
# Week 6 A2 - WINE Dataset Analysis using Traditional Machine Learning Model: SVM
# Author : Benjelyn Reves Patiag
# Date   : 20 May 2026
# Description:
# Task: Load, clean, visualize, train, test, and evaluate Linear SVM model.
# -------------------------------------------------------


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "wine.data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Wine dataset column names from UCI Wine dataset
COLUMN_NAMES = [
    "class",
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315_of_diluted_wines",
    "proline",
]

CLASS_NAMES = {
    1: "Class 1",
    2: "Class 2",
    3: "Class 3",
}


def load_and_clean_data() -> pd.DataFrame:
    """Load the Wine CSV-like data file and perform basic cleaning checks."""
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES)

    # Remove duplicate records if present
    duplicate_count_before = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)

    # Convert all columns to numeric, invalid values become NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows with missing values if any exist
    missing_count_before = int(df.isnull().sum().sum())
    df = df.dropna().reset_index(drop=True)

    # Save cleaning summary
    summary = {
        "rows_after_cleaning": len(df),
        "columns": len(df.columns),
        "missing_values_removed": missing_count_before,
        "duplicate_rows_removed": int(duplicate_count_before),
        "class_distribution": df["class"].value_counts().sort_index().to_dict(),
    }

    with open(OUTPUT_DIR / "cleaning_summary.txt", "w", encoding="utf-8") as file:
        for key, value in summary.items():
            file.write(f"{key}: {value}\n")

    return df


def visualize_data(df: pd.DataFrame) -> None:
    """Create scatter plot for selected Wine features."""
    plt.figure(figsize=(8, 6))

    for class_value, group in df.groupby("class"):
        plt.scatter(
            group["alcohol"],
            group["flavanoids"],
            label=CLASS_NAMES.get(class_value, str(class_value)),
            alpha=0.8,
        )

    plt.xlabel("Alcohol")
    plt.ylabel("Flavanoids")
    plt.title("Wine Dataset Scatter Plot: Alcohol vs Flavanoids")
    plt.legend(title="Wine Class")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "wine_scatter_plot.png", dpi=200)
    plt.close()


def train_and_evaluate(df: pd.DataFrame) -> None:
    """Train and test Linear SVM model and save evaluation results."""
    X = df.drop(columns=["class"])
    y = df["class"]

    # Split dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Linear SVM model with scaling pipeline
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear", random_state=42)),
        ]
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict testing dataset
    y_pred = model.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    result_text = f"""Traditional Machine Learning Model: Support Vector Machine (SVM)
Dataset: Wine Dataset
Kernel: Linear
Train/Test Split: 80% training, 20% testing
Testing Samples: {len(y_test)}

Evaluation Metrics on Testing Dataset:
Accuracy : {accuracy:.4f}
Precision: {precision:.4f}
Recall   : {recall:.4f}
F1-score : {f1:.4f}

Classification Report:
{report}
Confusion Matrix:
{matrix}
"""

    print(result_text)

    with open(OUTPUT_DIR / "evaluation_results.txt", "w", encoding="utf-8") as file:
        file.write(result_text)

    # Create result screenshot as image
    plt.figure(figsize=(9, 7))
    plt.axis("off")
    plt.text(
        0.02,
        0.98,
        result_text,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "svm_results_screenshot.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    wine_df = load_and_clean_data()
    visualize_data(wine_df)
    train_and_evaluate(wine_df)
