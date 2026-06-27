"""
Week 9 - Activity 2: User Knowledge Modeling Data Analysis Project
Author: Benjelyn Reves Patiag
Date: 27 June 2026

Description:
This script loads the User Knowledge Modeling dataset, cleans it, performs EDA,
classification, clustering, and exports results/figures.

"""


from __future__ import annotations

import json
import os
import zipfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass
class ProjectPaths:
    """Central path configuration for reproducible outputs."""

    base_dir: Path = Path(__file__).resolve().parent
    raw_zip: Path = Path(__file__).resolve().parent / "data" / "user+knowledge+modeling.zip"

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def output_dir(self) -> Path:
        return self.base_dir / "output"

    @property
    def figures_dir(self) -> Path:
        return self.base_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.base_dir / "tables"

    def create_dirs(self) -> None:
        for folder in [self.data_dir, self.output_dir, self.figures_dir, self.tables_dir]:
            folder.mkdir(parents=True, exist_ok=True)


class DataLoader:
    """Load the original XLS file from the uploaded ZIP dataset."""

    def __init__(self, paths: ProjectPaths):
        self.paths = paths

    def extract_zip(self) -> Path:
        """Extract the uploaded ZIP file and return the extracted XLS path."""
        self.paths.create_dirs()
        if not self.paths.raw_zip.exists():
            raise FileNotFoundError(f"Dataset ZIP not found: {self.paths.raw_zip}")

        with zipfile.ZipFile(self.paths.raw_zip, "r") as zip_ref:
            zip_ref.extractall(self.paths.data_dir)

        xls_files = list(self.paths.data_dir.glob("*.xls"))
        if not xls_files:
            raise FileNotFoundError("No .xls file found after ZIP extraction.")
        return xls_files[0]

    def convert_xls_to_xlsx_if_needed(self, xls_path: Path) -> Path:
        """Convert legacy .xls to .xlsx using LibreOffice when xlrd is unavailable."""
        xlsx_path = xls_path.with_suffix(".xlsx")
        if xlsx_path.exists():
            return xlsx_path

        try:
            pd.ExcelFile(xls_path)
            return xls_path
        except Exception:
            subprocess.run(
                [
                    "libreoffice",
                    "--headless",
                    "--convert-to",
                    "xlsx",
                    "--outdir",
                    str(xls_path.parent),
                    str(xls_path),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return xlsx_path

    def load_sheets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        xls_path = self.extract_zip()
        workbook_path = self.convert_xls_to_xlsx_if_needed(xls_path)
        info = pd.read_excel(workbook_path, sheet_name="Information")
        train = pd.read_excel(workbook_path, sheet_name="Training_Data")
        test = pd.read_excel(workbook_path, sheet_name="Test_Data")
        return info, train, test


class DataCleaner:
    """Clean columns, labels, types, and remove non-data columns."""

    feature_cols = ["STG", "SCG", "STR", "LPR", "PEG"]
    target_col = "UNS"

    @staticmethod
    def _standardise_target(value: str) -> str:
        return str(value).strip().lower().replace("very low", "very_low")

    def clean_split(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(col).strip() for col in df.columns]
        keep_cols = self.feature_cols + [self.target_col]
        df = df[keep_cols]

        for col in self.feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df[self.target_col] = df[self.target_col].apply(self._standardise_target)
        df = df.dropna(subset=self.feature_cols + [self.target_col])
        df = df.drop_duplicates().reset_index(drop=True)
        return df

    def clean(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        clean_train = self.clean_split(train)
        clean_test = self.clean_split(test)
        full = pd.concat([clean_train.assign(split="train"), clean_test.assign(split="test")], ignore_index=True)
        return clean_train, clean_test, full


class ExploratoryAnalyzer:
    """Create descriptive summaries and basic plots."""

    def __init__(self, paths: ProjectPaths):
        self.paths = paths

    def save_tables(self, df: pd.DataFrame) -> None:
        numeric = df[["STG", "SCG", "STR", "LPR", "PEG"]]
        numeric.describe().T.to_csv(self.paths.tables_dir / "descriptive_statistics.csv")
        df["UNS"].value_counts().rename_axis("knowledge_level").reset_index(name="count").to_csv(
            self.paths.tables_dir / "class_distribution.csv", index=False
        )
        df.groupby("UNS")[["STG", "SCG", "STR", "LPR", "PEG"]].mean().round(4).to_csv(
            self.paths.tables_dir / "knowledge_level_feature_profile.csv"
        )
        numeric.corr().round(4).to_csv(self.paths.tables_dir / "correlation_matrix.csv")

    def create_figures(self, df: pd.DataFrame) -> None:
        numeric = df[["STG", "SCG", "STR", "LPR", "PEG"]]

        # Class distribution
        plt.figure(figsize=(7, 5))
        df["UNS"].value_counts().sort_index().plot(kind="bar")
        plt.title("Knowledge Level Class Distribution")
        plt.xlabel("Knowledge Level")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(self.paths.figures_dir / "class_distribution.png", dpi=200)
        plt.close()

        # Correlation heatmap without seaborn dependency
        corr = numeric.corr()
        plt.figure(figsize=(7, 6))
        plt.imshow(corr, aspect="auto")
        plt.xticks(range(len(corr.columns)), corr.columns)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.colorbar(label="Correlation")
        plt.title("Feature Correlation Heatmap")
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(self.paths.figures_dir / "correlation_heatmap.png", dpi=200)
        plt.close()

        # Feature profile by class
        profile = df.groupby("UNS")[["STG", "SCG", "STR", "LPR", "PEG"]].mean()
        plt.figure(figsize=(9, 5))
        profile.T.plot(kind="bar")
        plt.title("Average Feature Profile by Knowledge Level")
        plt.xlabel("Feature")
        plt.ylabel("Average Value")
        plt.legend(title="UNS")
        plt.tight_layout()
        plt.savefig(self.paths.figures_dir / "feature_profile_by_class.png", dpi=200)
        plt.close()


class ClassificationAnalyzer:
    """Train and evaluate supervised classification models."""

    def __init__(self, paths: ProjectPaths):
        self.paths = paths

    def candidate_models(self) -> Dict[str, object]:
        return {
            "Logistic Regression": Pipeline(
                [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]
            ),
            "KNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier())]),
            "SVM RBF": Pipeline([("scaler", StandardScaler()), ("model", SVC(kernel="rbf", class_weight="balanced"))]),
            "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, class_weight="balanced"),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "MLP Neural Network": Pipeline(
                [("scaler", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42))]
            ),
        }

    def run(self, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        X_train = train[["STG", "SCG", "STR", "LPR", "PEG"]]
        y_train = train["UNS"]
        X_test = test[["STG", "SCG", "STR", "LPR", "PEG"]]
        y_test = test["UNS"]

        rows = []
        best_name = None
        best_model = None
        best_f1 = -1
        best_pred = None

        for name, model in self.candidate_models().items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            row = {
                "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
                "recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
                "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
            }
            rows.append(row)
            if row["f1_weighted"] > best_f1:
                best_f1 = row["f1_weighted"]
                best_name = name
                best_model = model
                best_pred = pred

        results = pd.DataFrame(rows).sort_values("f1_weighted", ascending=False)
        results.to_csv(self.paths.tables_dir / "classification_model_comparison.csv", index=False)

        report = classification_report(y_test, best_pred, zero_division=0)
        (self.paths.output_dir / "best_classification_report.txt").write_text(
            f"Best model: {best_name}\n\n{report}", encoding="utf-8"
        )

        labels = sorted(y_test.unique())
        cm = confusion_matrix(y_test, best_pred, labels=labels)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(self.paths.tables_dir / "best_confusion_matrix.csv")

        plt.figure(figsize=(7, 6))
        plt.imshow(cm, aspect="auto")
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.yticks(range(len(labels)), labels)
        plt.title(f"Confusion Matrix - {best_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.colorbar(label="Count")
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(self.paths.figures_dir / "best_confusion_matrix.png", dpi=200)
        plt.close()

        return results


class ClusteringAnalyzer:
    """Run unsupervised clustering and compare cluster quality."""

    def __init__(self, paths: ProjectPaths):
        self.paths = paths

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        features = ["STG", "SCG", "STR", "LPR", "PEG"]
        X = df[features]
        X_scaled = StandardScaler().fit_transform(X)
        encoded_target = LabelEncoder().fit_transform(df["UNS"])

        rows = []
        for k in range(2, 7):
            kmeans_labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_scaled)
            rows.append(self._score_cluster("KMeans", k, X_scaled, encoded_target, kmeans_labels))

            agglom_labels = AgglomerativeClustering(n_clusters=k).fit_predict(X_scaled)
            rows.append(self._score_cluster("Agglomerative", k, X_scaled, encoded_target, agglom_labels))

        results = pd.DataFrame(rows).sort_values(["silhouette", "adjusted_rand_index"], ascending=False)
        results.to_csv(self.paths.tables_dir / "clustering_model_comparison.csv", index=False)

        best = results.iloc[0]
        best_k = int(best["k"])
        best_labels = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(X_scaled)
        clustered = df.copy()
        clustered["cluster"] = best_labels
        clustered.groupby("cluster")[features].mean().round(4).to_csv(self.paths.tables_dir / "best_cluster_profile.csv")

        pca = PCA(n_components=2, random_state=42)
        xy = pca.fit_transform(X_scaled)
        plt.figure(figsize=(7, 5))
        plt.scatter(xy[:, 0], xy[:, 1], c=best_labels, s=35)
        plt.title(f"Best Clustering Result - KMeans k={best_k}")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.tight_layout()
        plt.savefig(self.paths.figures_dir / "best_cluster_pca.png", dpi=200)
        plt.close()

        return results

    @staticmethod
    def _score_cluster(method: str, k: int, X_scaled: np.ndarray, y_true: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        return {
            "method": method,
            "k": k,
            "silhouette": silhouette_score(X_scaled, labels),
            "adjusted_rand_index": adjusted_rand_score(y_true, labels),
            "normalized_mutual_info": normalized_mutual_info_score(y_true, labels),
        }


class UserKnowledgeAnalysisApp:
    """Main OOP controller for the full analysis workflow."""

    def __init__(self):
        self.paths = ProjectPaths()

    def run(self) -> None:
        self.paths.create_dirs()
        loader = DataLoader(self.paths)
        cleaner = DataCleaner()
        eda = ExploratoryAnalyzer(self.paths)
        classifier = ClassificationAnalyzer(self.paths)
        clusterer = ClusteringAnalyzer(self.paths)

        info, raw_train, raw_test = loader.load_sheets()
        train, test, full = cleaner.clean(raw_train, raw_test)
        full_no_split = full.drop(columns="split")

        train.to_csv(self.paths.output_dir / "clean_training_data.csv", index=False)
        test.to_csv(self.paths.output_dir / "clean_test_data.csv", index=False)
        full.to_csv(self.paths.output_dir / "clean_full_data.csv", index=False)

        eda.save_tables(full_no_split)
        eda.create_figures(full_no_split)
        classification_results = classifier.run(train, test)
        clustering_results = clusterer.run(full_no_split)

        summary = {
            "raw_training_rows": int(raw_train.shape[0]),
            "raw_test_rows": int(raw_test.shape[0]),
            "clean_training_rows": int(train.shape[0]),
            "clean_test_rows": int(test.shape[0]),
            "total_clean_rows": int(full_no_split.shape[0]),
            "missing_values_after_cleaning": int(full_no_split.isna().sum().sum()),
            "duplicate_rows_after_cleaning": int(full_no_split.duplicated().sum()),
            "best_classification_model": classification_results.iloc[0].to_dict(),
            "best_clustering_model": clustering_results.iloc[0].to_dict(),
        }
        (self.paths.output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    UserKnowledgeAnalysisApp().run()
