"""
Hotel Booking Data Analysis Project
Author: MSE803 Data Analytics Student
Dataset: hotel_bookings.csv

Purpose:
This script loads the hotel booking dataset, performs data cleaning and preprocessing,
creates useful business summaries, trains multiple classification models, compares
model performance, and saves the results in an output folder.

Main target:
Predict whether a hotel booking will be cancelled or not.

How to run:
    python hotel_booking_analysis.py

Expected input:
    hotel_bookings.csv

Expected outputs:
    output/
        cleaned_hotel_bookings.csv
        cleaned_hotel_bookings_sample.csv
        hotel_summary.csv
        monthly_summary.csv
        room_type_summary.csv
        model_comparison.csv
        preprocessing_report.txt
"""

import os
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore")


@dataclass
class ProjectConfig:
    """Project configuration."""

    input_file: str = "hotel_bookings.csv"
    output_dir: str = "output"
    random_state: int = 42
    test_size: float = 0.20


class HotelDataLoader:
    """Class 1: Load and inspect the dataset."""

    def __init__(self, config: ProjectConfig):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        """Load CSV file into Pandas DataFrame."""
        if not os.path.exists(self.config.input_file):
            raise FileNotFoundError(
                f"Cannot find {self.config.input_file}. "
                "Please put hotel_bookings.csv in the same folder as this script."
            )

        df = pd.read_csv(self.config.input_file)
        return df

    def inspect_data(self, df: pd.DataFrame) -> str:
        """Return basic dataset inspection report."""
        report = []
        report.append("HOTEL BOOKING DATASET INSPECTION")
        report.append("=" * 45)
        report.append(f"Rows: {df.shape[0]}")
        report.append(f"Columns: {df.shape[1]}")
        report.append("")
        report.append("Column names:")
        report.append(", ".join(df.columns.tolist()))
        report.append("")
        report.append("Missing values:")
        report.append(str(df.isna().sum()[df.isna().sum() > 0]))
        report.append("")
        report.append(f"Duplicate rows: {df.duplicated().sum()}")
        return "\n".join(report)


class HotelDataCleaner:
    """Class 2: Clean and preprocess raw hotel data."""

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning rules and feature engineering."""
        df = df.copy()

        # 1. Remove exact duplicate rows.
        df = df.drop_duplicates()

        # 2. Fill missing children with 0 because missing normally means no children.
        df["children"] = df["children"].fillna(0)

        # 3. Fill missing country with Unknown.
        df["country"] = df["country"].fillna("Unknown")

        # 4. Agent and company are IDs. Missing means no agent/company.
        df["agent"] = df["agent"].fillna(0)
        df["company"] = df["company"].fillna(0)

        # 5. Convert ID columns into integer type.
        df["children"] = df["children"].astype(int)
        df["agent"] = df["agent"].astype(int)
        df["company"] = df["company"].astype(int)

        # 6. Remove impossible bookings with no guest.
        # A booking should have at least one adult, child, or baby.
        df = df[(df["adults"] + df["children"] + df["babies"]) > 0]

        # 7. Remove negative ADR because room price should not be negative.
        df = df[df["adr"] >= 0]

        # 8. Remove extreme ADR outlier using business rule.
        # The dataset has a very high unrealistic ADR value.
        df = df[df["adr"] <= 1000]

        # 9. Convert date column.
        df["reservation_status_date"] = pd.to_datetime(
            df["reservation_status_date"], errors="coerce"
        )

        # 10. Create arrival date.
        month_map = {
            "January": 1,
            "February": 2,
            "March": 3,
            "April": 4,
            "May": 5,
            "June": 6,
            "July": 7,
            "August": 8,
            "September": 9,
            "October": 10,
            "November": 11,
            "December": 12,
        }
        df["arrival_month_number"] = df["arrival_date_month"].map(month_map)

        df["arrival_date"] = pd.to_datetime(
            dict(
                year=df["arrival_date_year"],
                month=df["arrival_month_number"],
                day=df["arrival_date_day_of_month"],
            ),
            errors="coerce",
        )

        # 11. Feature engineering for analysis and modelling.
        df["total_guests"] = df["adults"] + df["children"] + df["babies"]
        df["total_stay_nights"] = (
            df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
        )
        df["has_agent"] = np.where(df["agent"] > 0, 1, 0)
        df["has_company"] = np.where(df["company"] > 0, 1, 0)
        df["room_changed"] = np.where(
            df["reserved_room_type"] != df["assigned_room_type"], 1, 0
        )
        df["is_peak_month"] = np.where(
            df["arrival_date_month"].isin(["July", "August", "December"]), 1, 0
        )

        # 12. Remove rows with invalid created dates.
        df = df.dropna(subset=["arrival_date"])

        return df


class HotelEDA:
    """Class 3: Create business summary tables."""

    def create_hotel_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarize bookings by hotel type."""
        return (
            df.groupby("hotel")
            .agg(
                bookings=("hotel", "count"),
                cancellation_rate=("is_canceled", "mean"),
                avg_adr=("adr", "mean"),
                avg_stay=("total_stay_nights", "mean"),
                avg_guests=("total_guests", "mean"),
            )
            .reset_index()
            .sort_values("bookings", ascending=False)
        )

    def create_monthly_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarize occupancy demand trend by month."""
        month_order = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

        monthly = (
            df.groupby("arrival_date_month")
            .agg(
                bookings=("hotel", "count"),
                cancellation_rate=("is_canceled", "mean"),
                avg_adr=("adr", "mean"),
                avg_stay=("total_stay_nights", "mean"),
            )
            .reindex(month_order)
            .reset_index()
            .rename(columns={"arrival_date_month": "month"})
        )
        return monthly

    def create_room_type_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarize demand by reserved room type."""
        return (
            df.groupby("reserved_room_type")
            .agg(
                bookings=("hotel", "count"),
                cancellation_rate=("is_canceled", "mean"),
                avg_adr=("adr", "mean"),
                avg_stay=("total_stay_nights", "mean"),
            )
            .reset_index()
            .sort_values("bookings", ascending=False)
        )


class HotelModelBuilder:
    """Class 4: Train and compare models."""

    def __init__(self, config: ProjectConfig):
        self.config = config

    def get_features(self) -> Tuple[List[str], List[str], List[str]]:
        """Select numerical and categorical features for modelling."""
        numerical_features = [
            "lead_time",
            "arrival_date_year",
            "arrival_month_number",
            "arrival_date_week_number",
            "arrival_date_day_of_month",
            "stays_in_weekend_nights",
            "stays_in_week_nights",
            "adults",
            "children",
            "babies",
            "is_repeated_guest",
            "previous_cancellations",
            "previous_bookings_not_canceled",
            "booking_changes",
            "days_in_waiting_list",
            "adr",
            "required_car_parking_spaces",
            "total_of_special_requests",
            "total_guests",
            "total_stay_nights",
            "has_agent",
            "has_company",
            "room_changed",
            "is_peak_month",
        ]

        categorical_features = [
            "hotel",
            "meal",
            "country",
            "market_segment",
            "distribution_channel",
            "reserved_room_type",
            "deposit_type",
            "customer_type",
        ]

        all_features = numerical_features + categorical_features
        return numerical_features, categorical_features, all_features

    def build_preprocessor(
        self, numerical_features: List[str], categorical_features: List[str]
    ) -> ColumnTransformer:
        """Create preprocessing pipeline for ML model."""
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numerical_features),
                ("cat", categorical_pipeline, categorical_features),
            ]
        )

        return preprocessor

    def evaluate_model(self, name: str, model: Pipeline, X_test, y_test) -> dict:
        """Evaluate model using classification metrics."""
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = np.nan

        return {
            "approach": name,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc, 4),
        }

    def train_and_compare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Train multiple models and compare results."""
        numerical_features, categorical_features, all_features = self.get_features()

        X = df[all_features]
        y = df["is_canceled"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        preprocessor = self.build_preprocessor(numerical_features, categorical_features)

        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.config.random_state,
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=8,
                class_weight="balanced",
                random_state=self.config.random_state,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=120,
                max_depth=12,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=self.config.random_state,
                n_jobs=-1,
            ),
        }

        results = []

        # Baseline model: predict majority class only.
        majority_class = y_train.mode()[0]
        baseline_pred = np.full(shape=len(y_test), fill_value=majority_class)
        results.append(
            {
                "approach": "Baseline Majority Class",
                "accuracy": round(accuracy_score(y_test, baseline_pred), 4),
                "precision": round(
                    precision_score(y_test, baseline_pred, zero_division=0), 4
                ),
                "recall": round(recall_score(y_test, baseline_pred, zero_division=0), 4),
                "f1": round(f1_score(y_test, baseline_pred, zero_division=0), 4),
                "roc_auc": 0.5000,
            }
        )

        # Train each model.
        for name, estimator in models.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", estimator),
                ]
            )

            pipeline.fit(X_train, y_train)
            results.append(self.evaluate_model(name, pipeline, X_test, y_test))

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("f1", ascending=False).reset_index(drop=True)
        return results_df


class ReportExporter:
    """Class 5: Save all outputs."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        os.makedirs(self.config.output_dir, exist_ok=True)

    def save_outputs(
        self,
        raw_report: str,
        clean_df: pd.DataFrame,
        hotel_summary: pd.DataFrame,
        monthly_summary: pd.DataFrame,
        room_type_summary: pd.DataFrame,
        model_comparison: pd.DataFrame,
    ) -> None:
        """Save reports and CSV files."""
        clean_df.to_csv(
            os.path.join(self.config.output_dir, "cleaned_hotel_bookings.csv"),
            index=False,
        )

        clean_df.head(1000).to_csv(
            os.path.join(self.config.output_dir, "cleaned_hotel_bookings_sample.csv"),
            index=False,
        )

        hotel_summary.to_csv(
            os.path.join(self.config.output_dir, "hotel_summary.csv"), index=False
        )

        monthly_summary.to_csv(
            os.path.join(self.config.output_dir, "monthly_summary.csv"), index=False
        )

        room_type_summary.to_csv(
            os.path.join(self.config.output_dir, "room_type_summary.csv"), index=False
        )

        model_comparison.to_csv(
            os.path.join(self.config.output_dir, "model_comparison.csv"), index=False
        )

        report_path = os.path.join(self.config.output_dir, "preprocessing_report.txt")
        with open(report_path, "w", encoding="utf-8") as file:
            file.write(raw_report)
            file.write("\n\n")
            file.write("CLEANED DATASET RESULT\n")
            file.write("=" * 45)
            file.write("\n")
            file.write(f"Cleaned rows: {clean_df.shape[0]}\n")
            file.write(f"Cleaned columns: {clean_df.shape[1]}\n")
            file.write(f"Remaining missing values: {clean_df.isna().sum().sum()}\n")
            file.write("\n")
            file.write("Model comparison:\n")
            file.write(model_comparison.to_string(index=False))


class HotelBookingAnalysisApp:
    """Class 6: Main controller / orchestrator."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.loader = HotelDataLoader(config)
        self.cleaner = HotelDataCleaner()
        self.eda = HotelEDA()
        self.model_builder = HotelModelBuilder(config)
        self.exporter = ReportExporter(config)

    def run(self) -> None:
        """Run full analysis workflow."""
        print("Step 1: Loading dataset...")
        raw_df = self.loader.load_data()
        raw_report = self.loader.inspect_data(raw_df)
        print(raw_report)

        print("\nStep 2: Cleaning and preprocessing data...")
        clean_df = self.cleaner.clean_data(raw_df)
        print(f"Cleaned rows: {clean_df.shape[0]}")
        print(f"Cleaned columns: {clean_df.shape[1]}")

        print("\nStep 3: Creating business summaries...")
        hotel_summary = self.eda.create_hotel_summary(clean_df)
        monthly_summary = self.eda.create_monthly_summary(clean_df)
        room_type_summary = self.eda.create_room_type_summary(clean_df)

        print("\nHotel summary:")
        print(hotel_summary)

        print("\nStep 4: Training and comparing models...")
        model_comparison = self.model_builder.train_and_compare(clean_df)
        print(model_comparison)

        best_model = model_comparison.iloc[0]
        print("\nBest-performing approach:")
        print(
            f"{best_model['approach']} "
            f"with F1={best_model['f1']} and ROC-AUC={best_model['roc_auc']}"
        )

        print("\nStep 5: Saving outputs...")
        self.exporter.save_outputs(
            raw_report,
            clean_df,
            hotel_summary,
            monthly_summary,
            room_type_summary,
            model_comparison,
        )

        print(f"\nDone. Outputs saved in: {self.config.output_dir}")


if __name__ == "__main__":
    config = ProjectConfig(
        input_file="hotel_bookings.csv",
        output_dir="output",
        random_state=42,
        test_size=0.20,
    )

    app = HotelBookingAnalysisApp(config)
    app.run()
