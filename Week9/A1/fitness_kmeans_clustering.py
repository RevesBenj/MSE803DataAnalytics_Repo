"""
Week 9 - Activity 1: Clustering using KMeans
Author: Benjelyn Reves Patiag
Date: 7 June 2026

Description:
This script loads the Excel dataset, cleans it, applies K-Means clustering,
creates charts, and saves summary outputs.
Dataset: Fitness_App_User_Data.xlsx

"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class FitnessKMeansClustering:
    """This class keep all step in one place, so code is easy to follow."""

    def __init__(self, input_file: str, output_dir: str):
        # This is the input Excel file path.
        self.input_file = Path(input_file)

        # This is where all result files will be saved.
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # This is where all generated images will be saved.
        self.fig_dir = self.output_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        # These variables will be filled later in the process.
        self.raw_df = None
        self.clean_df = None
        self.model_df = None
        self.scaled_features = None
        self.scaler = None
        self.kmeans_model = None
        self.optimal_k = None
        self.cluster_summary = None
        self.k_results = None

        # These are the columns used for clustering.
        # User_ID is not used because it is just an ID, not behavior.
        # Churned is not used because it is outcome-like value, not a feature for grouping.
        self.feature_columns = [
            "Age",
            "Workouts_per_Week",
            "Avg_Session_Duration_Min",
            "Steps_per_Day",
        ]

    def load_data(self):
        """Load dataset from Excel."""
        # Read first sheet from Excel file.
        self.raw_df = pd.read_excel(self.input_file)
        return self.raw_df

    def clean_data(self):
        """Clean missing values, duplicate rows, data types and inconsistent text."""
        df = self.raw_df.copy()

        cleaning_log = []
        cleaning_log.append(f"Original rows: {len(df)}")
        cleaning_log.append(f"Original columns: {len(df.columns)}")

        # Remove duplicate rows if same full record exists.
        before_dup = len(df)
        df = df.drop_duplicates()
        after_dup = len(df)
        cleaning_log.append(f"Duplicate rows removed: {before_dup - after_dup}")

        # Clean text values so spelling/case will be consistent.
        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].astype(str).str.strip().str.title()

        if "Subscription_Type" in df.columns:
            df["Subscription_Type"] = df["Subscription_Type"].astype(str).str.strip().str.title()

        # Convert numeric columns safely. Bad values become NaN first.
        numeric_columns = [
            "User_ID",
            "Age",
            "Workouts_per_Week",
            "Avg_Session_Duration_Min",
            "Steps_per_Day",
            "Churned",
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Store missing values before fixing.
        missing_before = df.isna().sum()
        missing_before.to_csv(self.output_dir / "missing_values_before_cleaning.csv")

        # Fill missing numeric values using median because median is stable against outliers.
        for col in numeric_columns:
            if col in df.columns and df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
                cleaning_log.append(f"Missing numeric values in {col} filled using median.")

        # Fill missing text values using mode because it is the most common category.
        categorical_columns = ["Gender", "Subscription_Type"]
        for col in categorical_columns:
            if col in df.columns and df[col].isna().sum() > 0:
                mode_value = df[col].mode(dropna=True)[0]
                df[col] = df[col].fillna(mode_value)
                cleaning_log.append(f"Missing categorical values in {col} filled using mode.")

        # Fix logical ranges. This protects the model from impossible values.
        range_before = len(df)
        if "Age" in df.columns:
            df = df[(df["Age"] >= 13) & (df["Age"] <= 100)]
        if "Workouts_per_Week" in df.columns:
            df = df[(df["Workouts_per_Week"] >= 0) & (df["Workouts_per_Week"] <= 14)]
        if "Avg_Session_Duration_Min" in df.columns:
            df = df[(df["Avg_Session_Duration_Min"] > 0) & (df["Avg_Session_Duration_Min"] <= 240)]
        if "Steps_per_Day" in df.columns:
            df = df[(df["Steps_per_Day"] >= 0) & (df["Steps_per_Day"] <= 50000)]
        range_after = len(df)
        cleaning_log.append(f"Invalid / inconsistent rows removed: {range_before - range_after}")
        cleaning_log.append("Reason: values outside logical ranges, example negative workout/session values.")
        if "Churned" in df.columns:
            df["Churned"] = df["Churned"].astype(int)

        # Save missing values after fixing.
        missing_after = df.isna().sum()
        missing_after.to_csv(self.output_dir / "missing_values_after_cleaning.csv")

        cleaning_log.append(f"Rows after cleaning: {len(df)}")
        cleaning_log.append("No missing values found after cleaning." if missing_after.sum() == 0 else "Some missing values still exist.")
        cleaning_log.append("Data types were corrected for numeric columns and text columns were standardized.")

        # Save the cleaned dataset for submission evidence.
        self.clean_df = df.reset_index(drop=True)
        self.clean_df.to_csv(self.output_dir / "cleaned_fitness_app_user_data.csv", index=False)

        # Save cleaning log as simple text file.
        with open(self.output_dir / "preprocessing_steps.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(cleaning_log))

        return self.clean_df

    def prepare_features(self):
        """Scale features because KMeans is distance-based."""
        # Select only useful numeric behavior features.
        self.model_df = self.clean_df[self.feature_columns].copy()

        # StandardScaler makes all features same scale.
        # This is important because Steps_per_Day is much bigger than Age or Workouts.
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.model_df)
        return self.scaled_features

    def choose_optimal_k(self, min_k: int = 2, max_k: int = 8):
        """Try different K values and choose best K using silhouette score."""
        results = []

        for k in range(min_k, max_k + 1):
            # n_init='auto' is supported in new sklearn, but integer is safer for older systems.
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(self.scaled_features)

            inertia = model.inertia_
            silhouette = silhouette_score(self.scaled_features, labels)

            results.append({
                "k": k,
                "inertia": inertia,
                "silhouette_score": silhouette,
            })

        self.k_results = pd.DataFrame(results)
        self.k_results.to_csv(self.output_dir / "kmeans_k_selection_results.csv", index=False)

        # Choose K with the highest silhouette score.
        self.optimal_k = int(self.k_results.sort_values("silhouette_score", ascending=False).iloc[0]["k"])
        return self.optimal_k

    def train_final_model(self):
        """Train final KMeans model using selected K."""
        self.kmeans_model = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(self.scaled_features)

        # Add cluster label back to clean dataset.
        self.clean_df["Cluster"] = cluster_labels
        self.clean_df.to_csv(self.output_dir / "clustered_fitness_app_user_data.csv", index=False)
        return self.clean_df

    def create_cluster_summary(self):
        """Create human readable profile for every cluster."""
        # Main numeric profile of each cluster.
        numeric_summary = self.clean_df.groupby("Cluster")[self.feature_columns].mean().round(2)
        numeric_summary["User_Count"] = self.clean_df.groupby("Cluster").size()
        numeric_summary["Churn_Rate"] = self.clean_df.groupby("Cluster")["Churned"].mean().round(2)

        # Add most common category values for interpretation.
        numeric_summary["Most_Common_Gender"] = self.clean_df.groupby("Cluster")["Gender"].agg(lambda x: x.mode()[0])
        numeric_summary["Most_Common_Subscription"] = self.clean_df.groupby("Cluster")["Subscription_Type"].agg(lambda x: x.mode()[0])

        # Create simple business meaning based on activity level.
        summary = numeric_summary.reset_index()
        summary["Insight"] = summary.apply(self._make_cluster_insight, axis=1)

        self.cluster_summary = summary
        self.cluster_summary.to_csv(self.output_dir / "cluster_summary.csv", index=False)
        return self.cluster_summary

    def _make_cluster_insight(self, row):
        """Make simple label for marker and presentation."""
        workouts = row["Workouts_per_Week"]
        duration = row["Avg_Session_Duration_Min"]
        steps = row["Steps_per_Day"]
        churn = row["Churn_Rate"]

        if workouts >= 4.5 and duration >= 45 and steps >= 9000:
            activity = "Highly active users"
        elif workouts <= 2.5 and duration <= 35 and steps <= 7000:
            activity = "Low activity users"
        else:
            activity = "Moderate / mixed activity users"

        if churn >= 0.45:
            risk = "higher churn risk"
        elif churn <= 0.20:
            risk = "low churn risk"
        else:
            risk = "medium churn risk"

        return f"{activity}; {risk}."

    def create_visualizations(self):
        """Create charts needed for presentation and report evidence."""
        # 1. Missing values chart before cleaning.
        missing_before = pd.read_csv(self.output_dir / "missing_values_before_cleaning.csv", index_col=0)
        plt.figure(figsize=(9, 5))
        plt.bar(missing_before.index, missing_before.iloc[:, 0])
        plt.title("Missing Values Before Cleaning")
        plt.xlabel("Column")
        plt.ylabel("Missing Count")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(self.fig_dir / "01_missing_values_before_cleaning.png", dpi=180)
        plt.close()

        # 2. Elbow chart.
        plt.figure(figsize=(8, 5))
        plt.plot(self.k_results["k"], self.k_results["inertia"], marker="o")
        plt.title("Elbow Method for K-Means")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Inertia")
        plt.tight_layout()
        plt.savefig(self.fig_dir / "02_elbow_method.png", dpi=180)
        plt.close()

        # 3. Silhouette score chart.
        plt.figure(figsize=(8, 5))
        plt.plot(self.k_results["k"], self.k_results["silhouette_score"], marker="o")
        plt.axvline(self.optimal_k, linestyle="--")
        plt.title(f"Silhouette Score by K (Selected K = {self.optimal_k})")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Silhouette Score")
        plt.tight_layout()
        plt.savefig(self.fig_dir / "03_silhouette_scores.png", dpi=180)
        plt.close()

        # 4. PCA cluster scatter plot to show clusters in 2D.
        pca = PCA(n_components=2, random_state=42)
        components = pca.fit_transform(self.scaled_features)
        pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
        pca_df["Cluster"] = self.clean_df["Cluster"]
        pca_df.to_csv(self.output_dir / "pca_cluster_coordinates.csv", index=False)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(pca_df["PC1"], pca_df["PC2"], c=pca_df["Cluster"], s=50, alpha=0.80)
        plt.title("K-Means User Clusters Visualized with PCA")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()
        plt.savefig(self.fig_dir / "04_pca_cluster_scatter.png", dpi=180)
        plt.close()

        # 5. Cluster profile chart with average standardized values.
        profile = self.clean_df.groupby("Cluster")[self.feature_columns].mean()
        profile_scaled = (profile - profile.mean()) / profile.std(ddof=0)
        profile_scaled.T.plot(kind="bar", figsize=(10, 6))
        plt.title("Cluster Profile: Standardized Average Feature Values")
        plt.xlabel("Feature")
        plt.ylabel("Relative Level")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(self.fig_dir / "05_cluster_profile.png", dpi=180)
        plt.close()

        # 6. Churn rate by cluster.
        churn_rate = self.clean_df.groupby("Cluster")["Churned"].mean()
        plt.figure(figsize=(7, 5))
        plt.bar(churn_rate.index.astype(str), churn_rate.values)
        plt.title("Churn Rate by Cluster")
        plt.xlabel("Cluster")
        plt.ylabel("Churn Rate")
        plt.ylim(0, max(0.1, churn_rate.max() + 0.15))
        plt.tight_layout()
        plt.savefig(self.fig_dir / "06_churn_rate_by_cluster.png", dpi=180)
        plt.close()

    def run_all(self):
        """Run full analysis pipeline."""
        self.load_data()
        self.clean_data()
        self.prepare_features()
        self.choose_optimal_k()
        self.train_final_model()
        self.create_cluster_summary()
        self.create_visualizations()

        print("Analysis completed successfully.")
        print(f"Selected optimal K: {self.optimal_k}")
        print("Cluster summary:")
        print(self.cluster_summary.to_string(index=False))


if __name__ == "__main__":
    # Change this path if your dataset is in another folder.
    INPUT_FILE = "Fitness_App_User_Data.xlsx"
    OUTPUT_DIR = "outputs"

    analysis = FitnessKMeansClustering(INPUT_FILE, OUTPUT_DIR)
    analysis.run_all()
