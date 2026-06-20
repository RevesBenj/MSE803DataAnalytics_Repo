# -------------------------------------------------------
# Week 10 A2 - Heart Disease Data Analytics and Prediction Project
# Author : Benjelyn Reves Patiag
# Date   : 20 June 2026
# Description:
# This script performs:
# 1. Load dataset from data/heart+disease.zip
# 2. Data cleaning and preprocessing
# 3. Exploratory data analysis
# 4. Visualization generation
# 5. Machine learning classification
# 6. Model comparison
# 7. Prediction for a sample patient

# Dataset expected:
# - data/heart+disease.zip
# - Preferred file inside ZIP: processed.cleveland.data
# -------------------------------------------------------



from __future__ import annotations

import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


@dataclass
class ProjectConfig:
    """Central configuration for project paths and model settings."""

    # Path is automatically based on project folder, not current terminal folder.
    base_dir: Path = Path(__file__).resolve().parents[1]
    data_dir_name: str = "data"
    output_dir_name: str = "output"
    dataset_zip_name: str = "heart+disease.zip"
    dataset_file_inside_zip: str = "processed.cleveland.data"
    test_size: float = 0.30
    random_state: int = 42
    target_column: str = "target"

    @property
    def data_path(self) -> Path:
        path = self.base_dir / self.data_dir_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def output_path(self) -> Path:
        path = self.base_dir / self.output_dir_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def dataset_zip_path(self) -> Path:
        return self.data_path / self.dataset_zip_name


class HeartDiseaseDataLoader:
    """Loads heart disease data from the uploaded ZIP file.

    The UCI ZIP contains several files. This project uses
    processed.cleveland.data because it has the standard 14 columns used
    in many heart disease classification examples.
    """

    COLUMN_NAMES = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]

    def __init__(self, config: ProjectConfig):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        zip_path = self.config.dataset_zip_path

        if not zip_path.exists():
            raise FileNotFoundError(
                f"Dataset ZIP not found: {zip_path}\n"
                f"Please place the uploaded heart+disease.zip file inside the data folder."
            )

        with zipfile.ZipFile(zip_path, "r") as zip_file:
            available_files = zip_file.namelist()

            if self.config.dataset_file_inside_zip not in available_files:
                raise FileNotFoundError(
                    f"{self.config.dataset_file_inside_zip} not found inside {zip_path}.\n"
                    f"Available files: {available_files}"
                )

            with zip_file.open(self.config.dataset_file_inside_zip) as data_file:
                df = pd.read_csv(
                    data_file,
                    header=None,
                    names=self.COLUMN_NAMES,
                    na_values="?",
                )

        print(f"Dataset loaded from: {zip_path}")
        print(f"Dataset file used inside ZIP: {self.config.dataset_file_inside_zip}")
        return df


class HeartDiseaseCleaner:
    """Cleans and prepares the raw heart disease dataset."""

    def __init__(self, target_column: str):
        self.target_column = target_column

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Standardize column names.
        df.columns = [str(col).strip().lower() for col in df.columns]

        # Replace UCI missing value symbol with NaN.
        df = df.replace("?", np.nan)

        # Convert all columns to numeric because UCI columns are numeric coded.
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove duplicate records.
        df = df.drop_duplicates()

        # Convert original target: 0 = no disease, 1/2/3/4 = disease.
        df[self.target_column] = df[self.target_column].apply(lambda x: 0 if x == 0 else 1)

        return df


class HeartDiseaseEDA:
    """Creates descriptive statistics and summary outputs."""

    def __init__(self, config: ProjectConfig):
        self.config = config

    def save_summary(self, df: pd.DataFrame) -> None:
        summary = pd.DataFrame(
            {
                "Metric": [
                    "Rows",
                    "Columns",
                    "Duplicate rows after cleaning",
                    "Missing values after cleaning",
                    "No disease count",
                    "Heart disease count",
                ],
                "Value": [
                    df.shape[0],
                    df.shape[1],
                    df.duplicated().sum(),
                    int(df.isna().sum().sum()),
                    int((df[self.config.target_column] == 0).sum()),
                    int((df[self.config.target_column] == 1).sum()),
                ],
            }
        )
        summary.to_csv(self.config.output_path / "dataset_summary.csv", index=False)
        df.describe().T.to_csv(self.config.output_path / "descriptive_statistics.csv")

        corr = df.corr(numeric_only=True)[self.config.target_column].sort_values(ascending=False)
        corr.to_csv(self.config.output_path / "correlation_with_target.csv")


class HeartDiseaseVisualizer:
    """Generates clear visualizations for EDA and model results."""

    def __init__(self, config: ProjectConfig):
        self.config = config

    def plot_target_distribution(self, df: pd.DataFrame) -> None:
        counts = df[self.config.target_column].value_counts().sort_index()
        labels = ["No Disease" if idx == 0 else "Disease" for idx in counts.index]

        plt.figure(figsize=(7, 5))
        plt.bar(labels, counts.values)
        plt.title("Heart Disease Distribution")
        plt.xlabel("Heart Disease Status")
        plt.ylabel("Number of Patients")
        plt.tight_layout()
        plt.savefig(self.config.output_path / "01_target_distribution.png", dpi=300)
        plt.close()

    def plot_age_distribution(self, df: pd.DataFrame) -> None:
        plt.figure(figsize=(8, 5))
        ax = sns.boxplot(data=df, x=self.config.target_column, y="age")
        ax.set_title("Age Distribution by Heart Disease Status")
        ax.set_xlabel("Target: 0 = No Disease, 1 = Heart Disease")
        ax.set_ylabel("Age")
        plt.tight_layout()
        plt.savefig(self.config.output_path / "02_age_vs_disease_boxplot.png", dpi=300)
        plt.close()

    def plot_chest_pain_distribution(self, df: pd.DataFrame) -> None:
        cp_counts = pd.crosstab(df["cp"], df[self.config.target_column])
        cp_counts = cp_counts.rename(columns={0: "No Disease", 1: "Disease"})

        plt.figure(figsize=(8, 5))
        cp_counts.plot(kind="bar", ax=plt.gca())
        plt.title("Chest Pain Type vs Heart Disease")
        plt.xlabel("Chest Pain Type")
        plt.ylabel("Number of Patients")
        plt.legend(title="Target")
        plt.tight_layout()
        plt.savefig(self.config.output_path / "03_chest_pain_vs_disease.png", dpi=300)
        plt.close()

    def plot_correlation_heatmap(self, df: pd.DataFrame) -> None:
        plt.figure(figsize=(12, 9))
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(self.config.output_path / "04_correlation_heatmap.png", dpi=300)
        plt.close()

    def plot_model_comparison(self, metrics_df: pd.DataFrame) -> None:
        plt.figure(figsize=(9, 5))
        plot_df = metrics_df.melt(
            id_vars="Model",
            value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
            var_name="Metric",
            value_name="Score",
        )
        ax = sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette="deep")
        ax.set_title("Machine Learning Model Comparison")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(self.config.output_path / "05_model_comparison.png", dpi=300)
        plt.close()

    def save_all_eda_charts(self, df: pd.DataFrame) -> None:
        self.plot_target_distribution(df)
        self.plot_age_distribution(df)
        self.plot_chest_pain_distribution(df)
        self.plot_correlation_heatmap(df)


class HeartDiseaseModelTrainer:
    """Builds, trains, evaluates and compares classification models."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.models: Dict[str, Pipeline] = {}
        self.best_model_name: str | None = None
        self.best_model: Pipeline | None = None
        self.feature_columns: List[str] = []

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        categorical_features = [col for col in categorical_features if col in X.columns]
        numerical_features = [col for col in X.columns if col not in categorical_features]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numerical_features),
                ("cat", categorical_pipeline, categorical_features),
            ]
        )

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]
        self.feature_columns = X.columns.tolist()

        return train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        preprocessor = self._build_preprocessor(X_train)

        base_models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.config.random_state),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=self.config.random_state),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=self.config.random_state),
            "SVM": SVC(kernel="rbf", probability=True, random_state=self.config.random_state),
        }

        for name, model in base_models.items():
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)
            self.models[name] = pipeline

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        results = []

        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            results.append(
                {
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, zero_division=0),
                    "Recall": recall_score(y_test, y_pred, zero_division=0),
                    "F1 Score": f1_score(y_test, y_pred, zero_division=0),
                }
            )

            report = classification_report(y_test, y_pred)
            report_file = self.config.output_path / f"classification_report_{name.replace(' ', '_').lower()}.txt"
            with open(report_file, "w", encoding="utf-8") as file:
                file.write(report)

            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=["Actual No Disease", "Actual Disease"],
                columns=["Predicted No Disease", "Predicted Disease"],
            )
            cm_df.to_csv(self.config.output_path / f"confusion_matrix_{name.replace(' ', '_').lower()}.csv")

        metrics_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
        metrics_df.to_csv(self.config.output_path / "model_metrics.csv", index=False)

        self.best_model_name = metrics_df.iloc[0]["Model"]
        self.best_model = self.models[self.best_model_name]

        return metrics_df

    def predict_sample_patient(self) -> pd.DataFrame:
        if self.best_model is None:
            raise ValueError("Train and evaluate models first before prediction.")

        # Example patient data only. This is not medical advice.
        sample_patient = pd.DataFrame(
            [
                {
                    "age": 55,
                    "sex": 1,
                    "cp": 4,
                    "trestbps": 140,
                    "chol": 250,
                    "fbs": 0,
                    "restecg": 1,
                    "thalach": 135,
                    "exang": 1,
                    "oldpeak": 2.3,
                    "slope": 2,
                    "ca": 1,
                    "thal": 7,
                }
            ]
        )

        sample_patient = sample_patient[self.feature_columns]
        prediction = self.best_model.predict(sample_patient)[0]
        probability = self.best_model.predict_proba(sample_patient)[0][1]

        result = pd.DataFrame(
            {
                "Best_Model": [self.best_model_name],
                "Predicted_Class": [int(prediction)],
                "Interpretation": ["Heart Disease Risk" if prediction == 1 else "No Heart Disease Risk"],
                "Disease_Probability": [round(float(probability), 4)],
            }
        )
        result.to_csv(self.config.output_path / "sample_patient_prediction.csv", index=False)
        return result


class HeartDiseaseAnalysisApp:
    """Main controller/orchestrator class."""

    def __init__(self):
        self.config = ProjectConfig()
        self.loader = HeartDiseaseDataLoader(self.config)
        self.cleaner = HeartDiseaseCleaner(self.config.target_column)
        self.eda = HeartDiseaseEDA(self.config)
        self.visualizer = HeartDiseaseVisualizer(self.config)
        self.trainer = HeartDiseaseModelTrainer(self.config)

    def run(self) -> None:
        print("Starting Heart Disease Data Analytics Project...")

        raw_df = self.loader.load_data()
        cleaned_df = self.cleaner.clean(raw_df)
        cleaned_df.to_csv(self.config.output_path / "cleaned_heart_disease_dataset.csv", index=False)

        self.eda.save_summary(cleaned_df)
        self.visualizer.save_all_eda_charts(cleaned_df)

        X_train, X_test, y_train, y_test = self.trainer.prepare_data(cleaned_df)
        self.trainer.train_models(X_train, y_train)
        metrics_df = self.trainer.evaluate_models(X_test, y_test)
        self.visualizer.plot_model_comparison(metrics_df)

        prediction_result = self.trainer.predict_sample_patient()

        print("\nModel Metrics:")
        print(metrics_df)
        print("\nSample Patient Prediction:")
        print(prediction_result)
        print(f"\nDone. All outputs saved in: {self.config.output_path}")


if __name__ == "__main__":
    app = HeartDiseaseAnalysisApp()
    app.run()
