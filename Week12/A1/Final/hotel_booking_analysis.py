
"""
Hotel Booking Data Analysis Project
Author: MSE803 Data Analytics Project

This script performs:
1. Data loading and inspection
2. Data preprocessing and cleaning
3. Feature engineering
4. Exploratory data analysis
5. Visualization output
6. Machine learning comparison
7. Best-performing model identification

Dataset expected:
hotel_bookings.csv

Run:
python hotel_booking_analysis.py --input hotel_bookings.csv
"""

import argparse
import warnings
from pathlib import Path
from typing import Tuple, Dict, List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


class Config:
    """Central configuration for file paths and project settings."""

    def __init__(self, input_file: str):
        self.input_file = Path(input_file)
        self.output_dir = Path("output")
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.models_dir = self.output_dir / "models"

        self.target = "is_canceled"
        self.random_state = 42
        self.test_size = 0.20

        for folder in [self.output_dir, self.figures_dir, self.tables_dir, self.models_dir]:
            folder.mkdir(parents=True, exist_ok=True)


class DataLoader:
    """Loads the raw hotel booking dataset."""

    def __init__(self, config: Config):
        self.config = config

    def load(self) -> pd.DataFrame:
        """Read CSV data and return a pandas DataFrame."""
        if not self.config.input_file.exists():
            raise FileNotFoundError(f"File not found: {self.config.input_file}")

        df = pd.read_csv(self.config.input_file)
        print("\n=== DATA LOADED SUCCESSFULLY ===")
        print(f"Rows: {df.shape[0]:,}")
        print(f"Columns: {df.shape[1]:,}")
        return df


class DataInspector:
    """Creates basic inspection tables for raw data."""

    def __init__(self, config: Config):
        self.config = config

    def inspect(self, df: pd.DataFrame) -> None:
        """Save dataset overview, data types, and missing values."""
        overview = pd.DataFrame({
            "metric": [
                "rows",
                "columns",
                "duplicate_rows",
                "total_missing_values",
                "memory_mb"
            ],
            "value": [
                df.shape[0],
                df.shape[1],
                df.duplicated().sum(),
                df.isna().sum().sum(),
                round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            ]
        })

        dtype_table = pd.DataFrame({
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "missing_count": [df[col].isna().sum() for col in df.columns],
            "missing_percent": [round(df[col].isna().mean() * 100, 2) for col in df.columns],
            "unique_values": [df[col].nunique(dropna=True) for col in df.columns]
        }).sort_values("missing_percent", ascending=False)

        overview.to_csv(self.config.tables_dir / "01_dataset_overview.csv", index=False)
        dtype_table.to_csv(self.config.tables_dir / "02_column_quality_report.csv", index=False)

        print("\n=== RAW DATA INSPECTION SAVED ===")
        print(overview)


class DataCleaner:
    """Cleans and preprocesses the hotel booking dataset."""

    def __init__(self, config: Config):
        self.config = config

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning rules and return cleaned dataset."""
        cleaned = df.copy()

        # 1. Remove exact duplicate records.
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        duplicates_removed = before - len(cleaned)

        # 2. Fill missing values using business meaning.
        # In this dataset, missing company/agent usually means no company/agent.
        cleaned["company"] = cleaned["company"].fillna(0)
        cleaned["agent"] = cleaned["agent"].fillna(0)

        # Missing country is treated as Unknown, not deleted.
        cleaned["country"] = cleaned["country"].fillna("Unknown")

        # Children has very small missing count, fill with 0.
        cleaned["children"] = cleaned["children"].fillna(0)

        # 3. Convert data types.
        integer_columns = ["children", "agent", "company"]
        for col in integer_columns:
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].astype(int)

        # 4. Remove records with no guests.
        cleaned["total_guests"] = cleaned["adults"] + cleaned["children"] + cleaned["babies"]
        before_no_guests = len(cleaned)
        cleaned = cleaned[cleaned["total_guests"] > 0].copy()
        no_guest_removed = before_no_guests - len(cleaned)

        # 5. Remove invalid or extreme ADR values.
        # ADR below 0 is invalid. Very high ADR is likely data error.
        before_adr = len(cleaned)
        cleaned = cleaned[(cleaned["adr"] >= 0) & (cleaned["adr"] <= 1000)].copy()
        adr_removed = before_adr - len(cleaned)

        # 6. Create proper arrival date.
        cleaned["arrival_date"] = pd.to_datetime(
            cleaned["arrival_date_year"].astype(str) + "-" +
            cleaned["arrival_date_month"].astype(str) + "-" +
            cleaned["arrival_date_day_of_month"].astype(str),
            errors="coerce"
        )

        # 7. Remove impossible dates if any.
        before_date = len(cleaned)
        cleaned = cleaned.dropna(subset=["arrival_date"])
        date_removed = before_date - len(cleaned)

        # 8. Feature engineering.
        cleaned["total_nights"] = cleaned["stays_in_weekend_nights"] + cleaned["stays_in_week_nights"]
        cleaned["has_children"] = np.where((cleaned["children"] + cleaned["babies"]) > 0, 1, 0)
        cleaned["is_repeated_guest_flag"] = cleaned["is_repeated_guest"].astype(int)
        cleaned["arrival_month_num"] = cleaned["arrival_date"].dt.month
        cleaned["arrival_year_month"] = cleaned["arrival_date"].dt.to_period("M").astype(str)
        cleaned["revenue_estimate"] = cleaned["adr"] * cleaned["total_nights"]
        cleaned["is_peak_season"] = cleaned["arrival_date_month"].isin(["July", "August"]).astype(int)

        # 9. Save cleaning report.
        report = pd.DataFrame({
            "cleaning_step": [
                "Initial rows",
                "Exact duplicates removed",
                "Rows with zero guests removed",
                "Invalid/extreme ADR rows removed",
                "Invalid arrival dates removed",
                "Final clean rows"
            ],
            "value": [
                len(df),
                duplicates_removed,
                no_guest_removed,
                adr_removed,
                date_removed,
                len(cleaned)
            ]
        })
        report.to_csv(self.config.tables_dir / "03_cleaning_report.csv", index=False)

        # 10. Save sample cleaned dataset.
        cleaned.head(10000).to_csv(self.config.output_dir / "cleaned_hotel_bookings_sample.csv", index=False)

        print("\n=== DATA CLEANING COMPLETE ===")
        print(report)
        return cleaned


class EDAAnalyzer:
    """Performs descriptive analysis and creates summary tables."""

    def __init__(self, config: Config):
        self.config = config

    def analyze(self, df: pd.DataFrame) -> None:
        """Create summary tables."""
        hotel_summary = df.groupby("hotel").agg(
            bookings=("hotel", "count"),
            cancellation_rate=("is_canceled", "mean"),
            avg_adr=("adr", "mean"),
            avg_lead_time=("lead_time", "mean"),
            avg_total_nights=("total_nights", "mean"),
            avg_guests=("total_guests", "mean")
        ).reset_index()

        hotel_summary["cancellation_rate"] = (hotel_summary["cancellation_rate"] * 100).round(2)
        hotel_summary["avg_adr"] = hotel_summary["avg_adr"].round(2)

        monthly_summary = df.groupby("arrival_year_month").agg(
            bookings=("hotel", "count"),
            cancellation_rate=("is_canceled", "mean"),
            avg_adr=("adr", "mean"),
            estimated_revenue=("revenue_estimate", "sum")
        ).reset_index()

        monthly_summary["cancellation_rate"] = (monthly_summary["cancellation_rate"] * 100).round(2)
        monthly_summary["avg_adr"] = monthly_summary["avg_adr"].round(2)
        monthly_summary["estimated_revenue"] = monthly_summary["estimated_revenue"].round(2)

        room_type_summary = df.groupby("reserved_room_type").agg(
            bookings=("reserved_room_type", "count"),
            cancellation_rate=("is_canceled", "mean"),
            avg_adr=("adr", "mean"),
            avg_total_nights=("total_nights", "mean")
        ).reset_index().sort_values("bookings", ascending=False)

        room_type_summary["cancellation_rate"] = (room_type_summary["cancellation_rate"] * 100).round(2)
        room_type_summary["avg_adr"] = room_type_summary["avg_adr"].round(2)

        country_summary = df.groupby("country").agg(
            bookings=("country", "count"),
            cancellation_rate=("is_canceled", "mean"),
            avg_adr=("adr", "mean")
        ).reset_index().sort_values("bookings", ascending=False).head(15)

        country_summary["cancellation_rate"] = (country_summary["cancellation_rate"] * 100).round(2)
        country_summary["avg_adr"] = country_summary["avg_adr"].round(2)

        hotel_summary.to_csv(self.config.tables_dir / "04_hotel_summary.csv", index=False)
        monthly_summary.to_csv(self.config.tables_dir / "05_monthly_summary.csv", index=False)
        room_type_summary.to_csv(self.config.tables_dir / "06_room_type_summary.csv", index=False)
        country_summary.to_csv(self.config.tables_dir / "07_top_country_summary.csv", index=False)

        print("\n=== EDA SUMMARY TABLES SAVED ===")
        print(hotel_summary)


class VisualizationGenerator:
    """Creates and saves visualization outputs."""

    def __init__(self, config: Config):
        self.config = config

    def _savefig(self, filename: str) -> None:
        plt.tight_layout()
        plt.savefig(self.config.figures_dir / filename, dpi=160, bbox_inches="tight")
        plt.close()

    def plot_all(self, df: pd.DataFrame) -> None:
        """Generate all required visualizations."""
        # 1. Bookings by hotel.
        hotel_counts = df["hotel"].value_counts()
        plt.figure(figsize=(8, 5))
        hotel_counts.plot(kind="bar")
        plt.title("Bookings by Hotel Type")
        plt.xlabel("Hotel Type")
        plt.ylabel("Number of Bookings")
        self._savefig("01_bookings_by_hotel.png")

        # 2. Monthly occupancy proxy / booking trend.
        monthly = df.groupby("arrival_year_month").size()
        plt.figure(figsize=(12, 5))
        monthly.plot(kind="line", marker="o")
        plt.title("Monthly Booking Trend")
        plt.xlabel("Arrival Month")
        plt.ylabel("Number of Bookings")
        plt.xticks(rotation=45)
        self._savefig("02_monthly_booking_trend.png")

        # 3. Monthly average ADR trend.
        monthly_adr = df.groupby("arrival_year_month")["adr"].mean()
        plt.figure(figsize=(12, 5))
        monthly_adr.plot(kind="line", marker="o")
        plt.title("Average Daily Rate Trend")
        plt.xlabel("Arrival Month")
        plt.ylabel("Average Daily Rate")
        plt.xticks(rotation=45)
        self._savefig("03_monthly_adr_trend.png")

        # 4. Cancellation by hotel.
        cancel_by_hotel = df.groupby("hotel")["is_canceled"].mean() * 100
        plt.figure(figsize=(8, 5))
        cancel_by_hotel.plot(kind="bar")
        plt.title("Cancellation Rate by Hotel Type")
        plt.xlabel("Hotel Type")
        plt.ylabel("Cancellation Rate (%)")
        self._savefig("04_cancellation_rate_by_hotel.png")

        # 5. Room type popularity.
        room_counts = df["reserved_room_type"].value_counts().sort_values(ascending=False)
        plt.figure(figsize=(10, 5))
        room_counts.plot(kind="bar")
        plt.title("Bookings by Reserved Room Type")
        plt.xlabel("Reserved Room Type")
        plt.ylabel("Number of Bookings")
        self._savefig("05_room_type_bookings.png")

        # 6. ADR distribution.
        plt.figure(figsize=(10, 5))
        plt.hist(df["adr"], bins=50)
        plt.title("ADR Distribution After Cleaning")
        plt.xlabel("Average Daily Rate")
        plt.ylabel("Frequency")
        self._savefig("06_adr_distribution.png")

        # 7. Lead time vs cancellation.
        lead_bins = pd.cut(df["lead_time"], bins=[-1, 7, 30, 90, 180, 365, df["lead_time"].max()],
                           labels=["0-7", "8-30", "31-90", "91-180", "181-365", "365+"])
        lead_cancel = df.groupby(lead_bins)["is_canceled"].mean() * 100
        plt.figure(figsize=(10, 5))
        lead_cancel.plot(kind="bar")
        plt.title("Cancellation Rate by Lead Time Group")
        plt.xlabel("Lead Time Group (Days)")
        plt.ylabel("Cancellation Rate (%)")
        self._savefig("07_cancellation_by_lead_time.png")

        # 8. Correlation heatmap using matplotlib only.
        numeric_cols = [
            "is_canceled", "lead_time", "adults", "children", "babies",
            "previous_cancellations", "booking_changes", "days_in_waiting_list",
            "adr", "required_car_parking_spaces", "total_of_special_requests",
            "total_guests", "total_nights", "revenue_estimate"
        ]
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(11, 8))
        plt.imshow(corr, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Correlation Heatmap")
        self._savefig("08_correlation_heatmap.png")

        print("\n=== VISUALIZATIONS SAVED ===")
        print(f"Figures folder: {self.config.figures_dir}")


class ModelTrainer:
    """Builds and compares classification models for booking cancellation prediction."""

    def __init__(self, config: Config):
        self.config = config

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
        """Select features and target."""
        selected_features = [
            "hotel",
            "lead_time",
            "arrival_month_num",
            "stays_in_weekend_nights",
            "stays_in_week_nights",
            "adults",
            "children",
            "babies",
            "meal",
            "country",
            "market_segment",
            "distribution_channel",
            "is_repeated_guest",
            "previous_cancellations",
            "previous_bookings_not_canceled",
            "reserved_room_type",
            "assigned_room_type",
            "booking_changes",
            "deposit_type",
            "agent",
            "company",
            "days_in_waiting_list",
            "customer_type",
            "adr",
            "required_car_parking_spaces",
            "total_of_special_requests",
            "total_guests",
            "total_nights",
            "has_children",
            "is_peak_season"
        ]

        X = df[selected_features].copy()
        y = df[self.config.target].copy()

        # To keep training fast for classroom/project machines, use a stratified sample
        # when the cleaned dataset is large. Cleaning and EDA still use the full dataset.
        max_model_rows = 50000
        if len(X) > max_model_rows:
            sample_df = pd.concat([X, y.rename(self.config.target)], axis=1)
            sample_df, _ = train_test_split(
                sample_df,
                train_size=max_model_rows,
                random_state=self.config.random_state,
                stratify=sample_df[self.config.target]
            )
            y = sample_df[self.config.target]
            X = sample_df.drop(columns=[self.config.target])

        categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
        numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

        return X, y, categorical_features, numeric_features

    def train_and_compare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Train Logistic Regression and Random Forest, then compare results."""
        X, y, categorical_features, numeric_features = self.prepare_features(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )

        # Preprocessing for ML:
        # - Numeric features are scaled.
        # - Categorical features are one-hot encoded.
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_features)
            ]
        )

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(
                n_estimators=80,
                max_depth=16,
                min_samples_split=10,
                random_state=self.config.random_state,
                n_jobs=-1,
                class_weight="balanced_subsample"
            )
        }

        results = []
        best_model_name = None
        best_f1 = -1
        best_pipeline = None

        for model_name, model in models.items():
            print(f"\nTraining model: {model_name}")

            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)

            if hasattr(pipeline.named_steps["model"], "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob)
            else:
                roc_auc = np.nan

            metrics = {
                "model": model_name,
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "precision": round(precision_score(y_test, y_pred), 4),
                "recall": round(recall_score(y_test, y_pred), 4),
                "f1_score": round(f1_score(y_test, y_pred), 4),
                "roc_auc": round(roc_auc, 4)
            }

            results.append(metrics)

            # Save classification report.
            report = classification_report(y_test, y_pred, output_dict=True)
            pd.DataFrame(report).transpose().to_csv(
                self.config.tables_dir / f"08_classification_report_{model_name.replace(' ', '_').lower()}.csv"
            )

            # Save confusion matrix.
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, columns=["Predicted_Not_Canceled", "Predicted_Canceled"],
                                 index=["Actual_Not_Canceled", "Actual_Canceled"])
            cm_df.to_csv(self.config.tables_dir / f"09_confusion_matrix_{model_name.replace(' ', '_').lower()}.csv")

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_model_name = model_name
                best_pipeline = pipeline

        comparison = pd.DataFrame(results).sort_values("f1_score", ascending=False)
        comparison.to_csv(self.config.tables_dir / "10_model_comparison.csv", index=False)

        if best_pipeline is not None:
            joblib.dump(best_pipeline, self.config.models_dir / "best_cancellation_model.joblib")

        best_summary = pd.DataFrame({
            "best_model": [best_model_name],
            "selection_metric": ["f1_score"],
            "best_f1_score": [best_f1],
            "reason": ["F1 balances precision and recall, useful for cancellation prediction."]
        })
        best_summary.to_csv(self.config.tables_dir / "11_best_model_summary.csv", index=False)

        print("\n=== MODEL COMPARISON COMPLETE ===")
        print(comparison)
        print(f"\nBest model: {best_model_name} using F1 score.")
        return comparison


class ReportWriter:
    """Writes a simple text summary after analysis."""

    def __init__(self, config: Config):
        self.config = config

    def write_summary(self, df_raw: pd.DataFrame, df_clean: pd.DataFrame, model_results: pd.DataFrame) -> None:
        """Create a final summary markdown file."""
        best = model_results.iloc[0]

        monthly_peak = df_clean.groupby("arrival_year_month").size().sort_values(ascending=False).head(1)
        top_room = df_clean["reserved_room_type"].value_counts().head(1)
        hotel_cancel = (df_clean.groupby("hotel")["is_canceled"].mean() * 100).round(2)

        summary = f"""# Hotel Booking Data Analysis - Output Summary

## Dataset Size
- Raw rows: {len(df_raw):,}
- Clean rows: {len(df_clean):,}
- Columns after preprocessing: {df_clean.shape[1]}

## Cleaning Completed
- Removed exact duplicates.
- Filled missing company and agent as 0.
- Filled missing country as Unknown.
- Filled missing children as 0.
- Removed bookings with zero total guests.
- Removed invalid or extreme ADR values.
- Created arrival_date, total_guests, total_nights, revenue_estimate, and peak season features.

## Key Results
- Peak booking month: {monthly_peak.index[0]} with {int(monthly_peak.iloc[0]):,} bookings.
- Most common reserved room type: {top_room.index[0]} with {int(top_room.iloc[0]):,} bookings.
- Cancellation rate by hotel:
{hotel_cancel.to_string()}

## Best-Performing Approach
Best model: {best['model']}

Metrics:
- Accuracy: {best['accuracy']}
- Precision: {best['precision']}
- Recall: {best['recall']}
- F1 Score: {best['f1_score']}
- ROC-AUC: {best['roc_auc']}

## Business Insights
1. High lead time is linked with higher cancellation risk.
2. Some hotel types and customer types have different cancellation behavior.
3. ADR changes across months, supporting dynamic pricing.
4. Popular room types should be prioritized in inventory and promotion.
5. Cancellation prediction can support better staffing, overbooking control, and revenue planning.

## Saved Outputs
- Tables: output/tables/
- Figures: output/figures/
- Best model: output/models/best_cancellation_model.joblib
"""
        (self.config.output_dir / "analysis_summary.md").write_text(summary, encoding="utf-8")


class HotelBookingAnalysisApp:
    """Main orchestrator class."""

    def __init__(self, input_file: str):
        self.config = Config(input_file)

    def run(self) -> None:
        """Run the full analysis workflow."""
        loader = DataLoader(self.config)
        inspector = DataInspector(self.config)
        cleaner = DataCleaner(self.config)
        eda = EDAAnalyzer(self.config)
        visualizer = VisualizationGenerator(self.config)
        trainer = ModelTrainer(self.config)
        report_writer = ReportWriter(self.config)

        raw_df = loader.load()
        inspector.inspect(raw_df)

        clean_df = cleaner.clean(raw_df)
        eda.analyze(clean_df)
        visualizer.plot_all(clean_df)

        model_results = trainer.train_and_compare(clean_df)
        report_writer.write_summary(raw_df, clean_df, model_results)

        print("\n=== PROJECT FINISHED SUCCESSFULLY ===")
        print("Open the output folder to view figures, tables, model, and summary.")


def main():
    parser = argparse.ArgumentParser(description="Hotel Booking Data Analysis Project")
    parser.add_argument("--input", type=str, default="hotel_bookings.csv", help="Path to hotel_bookings.csv")
    args = parser.parse_args()

    app = HotelBookingAnalysisApp(args.input)
    app.run()


if __name__ == "__main__":
    main()
