# -------------------------------------------------------
# Week 7 A2 - Fraud Detection Classification using Support Vector Machine (SVM)
# Author : Benjelyn Reves Patiag
# Date   : 28 May 2026
# Description:
# This script uses OOP structure so each task is clean:
#     - Load dataset
#     - Clean dataset
#     - Perform EDA
#     - Prepare features
#     - Train SVM model
#     - Evaluate model
#     - Save outputs and charts
# Dataset: creditcard.csv
# Problem:
#     Classify credit card transactions as:
#     0 = Normal transaction
#     1 = Fraud transaction

# -------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")


@dataclass
class ProjectConfig:
    """This class keeps all project settings in one place."""

    data_path: str
    output_dir: str = "outputs"
    staging_folder: str = "staging"
    cleaned_dataset_name: str = "creditcard_cleaned_staging.csv"
    test_size: float = 0.30
    random_state: int = 42
    target_column: str = "Class"
    # SVM can be slow on big data, but LinearSVC is good for large dataset.
    max_iter: int = 10000


class DataLoader:
    """This class loads the CSV dataset."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def load_data(self) -> pd.DataFrame:
        """Load CSV file into pandas DataFrame."""
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.config.data_path}")

        # Read the CSV file.
        df = pd.read_csv(self.config.data_path)

        # Return loaded dataset.
        return df


class DataCleaner:
    """This class handles data cleaning and basic preparation."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.cleaning_summary: Dict[str, object] = {}

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing values and duplicates."""
        cleaned_df = df.copy()

        # Save original size before cleaning.
        self.cleaning_summary["original_rows"] = int(cleaned_df.shape[0])
        self.cleaning_summary["original_columns"] = int(cleaned_df.shape[1])

        # Check missing values before cleaning.
        missing_before = cleaned_df.isnull().sum()
        self.cleaning_summary["missing_values_before"] = missing_before[missing_before > 0].to_dict()

        # Remove exact duplicate rows because duplicate transaction rows can bias the model.
        duplicates_before = int(cleaned_df.duplicated().sum())
        cleaned_df = cleaned_df.drop_duplicates()
        self.cleaning_summary["duplicates_removed"] = duplicates_before

        # Fill missing numeric values using median.
        # Median is safer than mean because fraud data can contain outliers.
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        for column in numeric_columns:
            if cleaned_df[column].isnull().sum() > 0:
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())

        # Fill missing categorical values using mode if categorical columns exist.
        categorical_columns = cleaned_df.select_dtypes(exclude=[np.number]).columns.tolist()
        for column in categorical_columns:
            if cleaned_df[column].isnull().sum() > 0:
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])

        # Check missing values after cleaning.
        missing_after = cleaned_df.isnull().sum()
        self.cleaning_summary["missing_values_after"] = missing_after[missing_after > 0].to_dict()

        # Save final size after cleaning.
        self.cleaning_summary["final_rows"] = int(cleaned_df.shape[0])
        self.cleaning_summary["final_columns"] = int(cleaned_df.shape[1])

        return cleaned_df

    def save_cleaned_dataset(self, cleaned_df: pd.DataFrame, output_dir: Path) -> Path:
        """Save the cleaned/staging dataset into a CSV file.

        This file is the clean dataset after duplicate removal and missing-value handling.
        It is useful because the user can inspect the exact dataset used for EDA and model training.
        """
        # Create staging folder inside the main output folder.
        staging_dir = output_dir / self.config.staging_folder
        staging_dir.mkdir(parents=True, exist_ok=True)

        # Build the full output path for the cleaned dataset.
        cleaned_dataset_path = staging_dir / self.config.cleaned_dataset_name

        # Save cleaned data without the pandas index column.
        cleaned_df.to_csv(cleaned_dataset_path, index=False)

        # Save path into summary so it is documented also.
        self.cleaning_summary["cleaned_staging_dataset"] = str(cleaned_dataset_path)

        return cleaned_dataset_path

    def save_cleaning_summary(self, output_dir: Path) -> None:
        """Save cleaning summary into JSON file."""
        with open(output_dir / "cleaning_summary.json", "w", encoding="utf-8") as file:
            json.dump(self.cleaning_summary, file, indent=4)


class ExploratoryDataAnalysis:
    """This class creates EDA reports and visualizations."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def run_eda(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Run full EDA and save summary files and charts."""
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Save dataset information and descriptive statistics.
        df.describe().to_csv(output_dir / "descriptive_statistics.csv")
        df.dtypes.astype(str).to_csv(output_dir / "data_types.csv", header=["data_type"])
        df.isnull().sum().to_csv(output_dir / "missing_values.csv", header=["missing_count"])

        # Chart 1: Target distribution.
        self._plot_class_distribution(df, figures_dir)

        # Chart 2: Transaction amount distribution.
        self._plot_amount_distribution(df, figures_dir)

        # Chart 3: Fraud vs normal amount boxplot.
        self._plot_amount_by_class(df, figures_dir)

        # Chart 4: Correlation heatmap with target.
        self._plot_correlation_heatmap(df, figures_dir)

        # Chart 5: Top features correlated with fraud.
        self._plot_top_target_correlations(df, figures_dir)

    def _plot_class_distribution(self, df: pd.DataFrame, figures_dir: Path) -> None:
        """Create bar chart for normal and fraud records."""
        counts = df[self.config.target_column].value_counts().sort_index()

        plt.figure(figsize=(7, 5))
        counts.plot(kind="bar")
        plt.title("Class Distribution: Normal vs Fraud")
        plt.xlabel("Class (0 = Normal, 1 = Fraud)")
        plt.ylabel("Number of Records")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(figures_dir / "class_distribution.png", dpi=150)
        plt.close()

    def _plot_amount_distribution(self, df: pd.DataFrame, figures_dir: Path) -> None:
        """Create histogram for transaction amount."""
        plt.figure(figsize=(8, 5))
        df["Amount"].plot(kind="hist", bins=50)
        plt.title("Transaction Amount Distribution")
        plt.xlabel("Amount")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(figures_dir / "amount_distribution.png", dpi=150)
        plt.close()

    def _plot_amount_by_class(self, df: pd.DataFrame, figures_dir: Path) -> None:
        """Create boxplot to compare transaction amount by class."""
        plt.figure(figsize=(7, 5))
        df.boxplot(column="Amount", by=self.config.target_column)
        plt.title("Transaction Amount by Class")
        plt.suptitle("")
        plt.xlabel("Class (0 = Normal, 1 = Fraud)")
        plt.ylabel("Amount")
        plt.tight_layout()
        plt.savefig(figures_dir / "amount_by_class_boxplot.png", dpi=150)
        plt.close()

    def _plot_correlation_heatmap(self, df: pd.DataFrame, figures_dir: Path) -> None:
        """Create simple heatmap for correlation between variables."""
        corr = df.corr(numeric_only=True)

        plt.figure(figsize=(12, 10))
        plt.imshow(corr, aspect="auto")
        plt.colorbar()
        plt.title("Correlation Heatmap")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
        plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
        plt.tight_layout()
        plt.savefig(figures_dir / "correlation_heatmap.png", dpi=150)
        plt.close()

    def _plot_top_target_correlations(self, df: pd.DataFrame, figures_dir: Path) -> None:
        """Create bar chart of features most related to fraud class."""
        corr = df.corr(numeric_only=True)[self.config.target_column]
        corr = corr.drop(self.config.target_column).abs().sort_values(ascending=False).head(10)

        plt.figure(figsize=(8, 5))
        corr.sort_values().plot(kind="barh")
        plt.title("Top 10 Features Correlated with Fraud Class")
        plt.xlabel("Absolute Correlation with Class")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(figures_dir / "top_target_correlations.png", dpi=150)
        plt.close()


class FeatureEngineer:
    """This class selects features and prepares train and test data."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate input features X and target label y."""
        if self.config.target_column not in df.columns:
            raise ValueError(f"Target column '{self.config.target_column}' does not exist.")

        # X contains independent variables.
        x = df.drop(columns=[self.config.target_column])

        # y contains the target fraud label.
        y = df[self.config.target_column]

        return x, y

    def train_test_split_data(
        self, x: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing dataset."""
        return train_test_split(
            x,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,  # keep fraud ratio same in train and test
        )


class SVMFraudModel:
    """This class creates, trains, and predicts using the SVM model."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.model: Pipeline | None = None

    def build_model(self) -> Pipeline:
        """Build SVM model pipeline with scaling."""
        # Scaling is important because SVM is sensitive to different feature ranges.
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svm",
                    LinearSVC(
                        class_weight="balanced",  # helps because fraud class is very small
                        max_iter=self.config.max_iter,
                        random_state=self.config.random_state,
                    ),
                ),
            ]
        )
        return self.model

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the SVM model."""
        if self.model is None:
            self.build_model()

        # Fit means model learns pattern from training data.
        self.model.fit(x_train, y_train)

    def predict(self, x_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class labels and decision scores."""
        if self.model is None:
            raise ValueError("Model is not trained yet.")

        # Predict gives final class 0 or 1.
        y_pred = self.model.predict(x_test)

        # Decision function gives confidence score for SVM boundary.
        decision_scores = self.model.decision_function(x_test)

        return y_pred, decision_scores

    def save_model(self, output_dir: Path) -> None:
        """Save trained model to file."""
        if self.model is None:
            raise ValueError("Model is not trained yet.")

        joblib.dump(self.model, output_dir / "svm_fraud_model.joblib")


class ModelEvaluator:
    """This class evaluates model performance."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def evaluate(
        self,
        y_test: pd.Series,
        y_pred: np.ndarray,
        decision_scores: np.ndarray,
        output_dir: Path,
    ) -> Dict[str, float]:
        """Calculate metrics and save results."""
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Calculate important classification metrics.
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, decision_scores)),
            "average_precision_pr_auc": float(average_precision_score(y_test, decision_scores)),
        }

        # Save metrics as JSON.
        with open(output_dir / "model_metrics.json", "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

        # Save classification report as text.
        report = classification_report(
            y_test,
            y_pred,
            target_names=["Normal", "Fraud"],
            zero_division=0,
        )
        with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as file:
            file.write(report)

        # Save confusion matrix as CSV.
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Normal", "Actual Fraud"],
            columns=["Predicted Normal", "Predicted Fraud"],
        )
        cm_df.to_csv(output_dir / "confusion_matrix.csv")

        # Save confusion matrix chart.
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Normal", "Fraud"],
        )
        display.plot(values_format="d")
        plt.title("Confusion Matrix - SVM Fraud Detection")
        plt.tight_layout()
        plt.savefig(figures_dir / "confusion_matrix.png", dpi=150)
        plt.close()

        return metrics


class FraudDetectionPipeline:
    """This class controls the whole end-to-end fraud detection workflow."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Run complete project pipeline."""
        print("Starting Fraud Detection SVM Pipeline...")

        # Step 1: Load dataset.
        loader = DataLoader(self.config)
        df = loader.load_data()
        print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

        # Step 2: Clean dataset.
        cleaner = DataCleaner(self.config)
        df_clean = cleaner.clean_data(df)

        # Save the cleaned/staging dataset so the cleaned result can be reviewed or reused.
        cleaned_dataset_path = cleaner.save_cleaned_dataset(df_clean, self.output_dir)

        # Save the cleaning summary after saving the staging dataset path.
        cleaner.save_cleaning_summary(self.output_dir)

        print(f"Cleaned dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        print(f"Cleaned/staging dataset saved: {cleaned_dataset_path}")

        # Step 3: Run EDA and save charts.
        eda = ExploratoryDataAnalysis(self.config)
        eda.run_eda(df_clean, self.output_dir)
        print("EDA completed and charts saved.")

        # Step 4: Feature selection and train-test split.
        engineer = FeatureEngineer(self.config)
        x, y = engineer.split_features_target(df_clean)
        x_train, x_test, y_train, y_test = engineer.train_test_split_data(x, y)
        print(f"Training records: {x_train.shape[0]}")
        print(f"Testing records: {x_test.shape[0]}")

        # Step 5: Train SVM model.
        svm_model = SVMFraudModel(self.config)
        svm_model.build_model()
        svm_model.train(x_train, y_train)
        svm_model.save_model(self.output_dir)
        print("SVM model trained and saved.")

        # Step 6: Predict and evaluate.
        y_pred, decision_scores = svm_model.predict(x_test)
        evaluator = ModelEvaluator(self.config)
        metrics = evaluator.evaluate(y_test, y_pred, decision_scores, self.output_dir)

        # Print final metrics.
        print("\nModel Evaluation Results")
        print("------------------------")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        print("\nPipeline finished successfully.")
        print(f"All outputs saved in: {self.output_dir}")


def parse_arguments() -> argparse.Namespace:
    """Read command line arguments."""
    parser = argparse.ArgumentParser(description="Fraud Detection using SVM")
    parser.add_argument(
        "--data-path",
        type=str,
        default="creditcard.csv",
        help="Path to creditcard.csv file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Folder where reports, charts, and model will be saved",
    )
    return parser.parse_args()


def main() -> None:
    """Main program entry point."""
    args = parse_arguments()

    # Create config object.
    config = ProjectConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
    )

    # Run full project pipeline.
    pipeline = FraudDetectionPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()