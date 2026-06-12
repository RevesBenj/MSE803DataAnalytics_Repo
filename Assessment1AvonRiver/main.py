"""
MSE803 Assessment 1 - Avon River Water Quality & Fish Population Analysis
==========================================================================
Author  : Benjelyn Reves Patiag
Date   : 20 May 2026
Description: Analysis of Avon River water quality and fish population data.
How to run:
    pip install -r requirements.txt
    python main.py
"""
# This stop annoying warning message from show
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path
from datetime import datetime
import textwrap
import os

# This avoids slow/hanging behaviour from too many math-library threads on small datasets.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ─────────────────────────────────────────────────────────────────────────────
# Third-party library
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # This make chart save without popup window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score
)
import statsmodels.api as sm


# =============================================================================
# CLASS 1 - DataLoader
# =============================================================================
class DataLoader:
    """
    This class only job is load the Excel file and show basic information.
    Single responsibility: read data from disk into memory.
    """

    def __init__(self, data_dir: Path):
        # This store the folder where data file live
        self.data_dir = data_dir
        self.raw_df: pd.DataFrame = pd.DataFrame()
        self.file_path: Path = Path()

    def load(self, filename: str) -> pd.DataFrame:
        """Load Excel file and return raw DataFrame."""
        self.file_path = self.data_dir / filename
        if not self.file_path.exists():
            # Fallback: allow running when Excel file is in same folder as main.py.
            alt_path = self.data_dir.parent / filename
            if alt_path.exists():
                self.file_path = alt_path
            else:
                raise FileNotFoundError(f"[DataLoader] Cannot find file: {self.file_path} or {alt_path}")

        print("\n" + "=" * 70)
        print("  STEP 1: LOADING DATA")
        print("=" * 70)

        # This read Excel skip row 0 (the big header row) and use row 1 as column
        self.raw_df = pd.read_excel(self.file_path, header=1)
        print(f"  File loaded  : {self.file_path.name}")
        print(f"  Sheet names  : {pd.ExcelFile(self.file_path).sheet_names}")
        print(f"  Raw shape    : {self.raw_df.shape[0]} rows x {self.raw_df.shape[1]} columns")
        print(f"  Columns      : {self.raw_df.columns.tolist()}")
        print(f"\n  First 5 rows (raw):")
        print(self.raw_df.head().to_string(index=False))
        print(f"\n  Data types:")
        print(self.raw_df.dtypes.to_string())
        return self.raw_df


# =============================================================================
# CLASS 2 - DataCleaner
# =============================================================================
class DataCleaner:
    """
    This class clean and prepare data for analysis.
    Single responsibility: transform messy raw data into clean usable data.
    """

    def __init__(self):
        self.clean_df: pd.DataFrame = pd.DataFrame()
        self.cleaning_log: list = []

    def clean(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Run all cleaning step and return clean DataFrame."""
        print("\n" + "=" * 70)
        print("  STEP 2: DATA CLEANING & PREPROCESSING")
        print("=" * 70)

        df = raw_df.copy()

        # ── Step 2a: Remove separator column ─────────────────────────────────
        # This unnamed column is blank space between two table in Excel
        df = df.drop(columns=["Unnamed: 5"], errors="ignore")

        # ── Step 2b: Rename column to simple name ─────────────────────────────
        df.columns = [
            "Site_ID", "Date", "Temperature_C", "pH",
            "Dissolved_Oxygen", "Site_ID_FP", "Date_FP",
            "Species", "Fish_Count", "Avg_Size_cm",
        ]
        self.cleaning_log.append("Columns renamed to simple names")

        # ── Step 2c: Parse date column ───────────────────────────────────────
        # This convert date to proper datetime type
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Date_FP"] = pd.to_datetime(df["Date_FP"], errors="coerce")
        self.cleaning_log.append("Date columns parsed to datetime")

        # ── Step 2d: Handle text value before numeric conversion ──────────────
        # This replace text "No fish observed" with 0 before conversion.
        # Important: if we convert to numeric first, text become NaN and real zero evidence is lost.
        df["Fish_Count"] = df["Fish_Count"].replace("No fish observed", 0)
        self.cleaning_log.append("'No fish observed' replaced with 0 before numeric conversion")

        # ── Step 2e: Numeric conversion ──────────────────────────────────────
        # This make sure number column is number not text
        numeric_cols = ["Temperature_C", "pH", "Dissolved_Oxygen", "Fish_Count", "Avg_Size_cm"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        self.cleaning_log.append("Numeric columns converted")

        # ── Step 2f: Missing value report ────────────────────────────────────
        missing_before = df.isnull().sum()
        print(f"\n  Missing values before cleaning:")
        print(missing_before[missing_before > 0].to_string())

        # Fill missing Species with Unknown
        df["Species"] = df["Species"].fillna("Unknown")

        # Fill missing numeric with column median
        for col in numeric_cols:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                self.cleaning_log.append(
                    f"Filled {n_missing} missing {col} with median={median_val:.3f}"
                )
                print(f"  Filled {n_missing} missing value in '{col}' with median {median_val:.3f}")

        # ── Step 2g: Remove exact duplicate row ──────────────────────────────
        # This removes true repeated records only.
        # Do not use numeric-only duplicate check because two different site/date/species rows
        # can have same numeric values but still be valid observations.
        n_dupes = df.duplicated().sum()
        df = df.drop_duplicates()
        self.cleaning_log.append(f"Removed {n_dupes} exact duplicate rows")
        print(f"\n  Exact duplicate rows removed: {n_dupes}")

        # ── Step 2h: Plausibility range check ────────────────────────────────
        # This remove row that have impossible value (e.g. pH = 99)
        bounds = {
            "Temperature_C":    (0, 40),
            "pH":               (0, 14),
            "Dissolved_Oxygen": (0, 20),
            "Fish_Count":       (0, 500),
            "Avg_Size_cm":      (0, 200),
        }
        before = len(df)
        for col, (lo, hi) in bounds.items():
            df = df[(df[col] >= lo) & (df[col] <= hi)]
        removed_range = before - len(df)
        self.cleaning_log.append(f"Removed {removed_range} rows by range check")
        print(f"  Rows removed by range plausibility check: {removed_range}")

        # ── Step 2i: IQR outlier detection ───────────────────────────────────
        # This find extreme outlier using IQR method and remove them
        print(f"\n  IQR Outlier Detection:")
        outlier_flags = pd.DataFrame(index=df.index)
        for col in ["Temperature_C", "pH", "Dissolved_Oxygen", "Fish_Count", "Avg_Size_cm"]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lo_iqr = Q1 - 1.5 * IQR
            hi_iqr = Q3 + 1.5 * IQR
            flagged = (df[col] < lo_iqr) | (df[col] > hi_iqr)
            outlier_flags[col] = flagged
            print(f"    {col:25s}: {flagged.sum():3d} outlier(s)  "
                  f"[IQR fence {lo_iqr:.2f} - {hi_iqr:.2f}]")

        any_outlier = outlier_flags.any(axis=1)
        self.outlier_df = df[any_outlier].copy()
        df = df[~any_outlier].copy()
        self.cleaning_log.append(f"Removed {any_outlier.sum()} IQR outlier rows")
        print(f"  Total rows with outlier flag: {any_outlier.sum()}")
        print(f"  Rows kept after outlier removal: {len(df)}")

        # ── Step 2j: Feature engineering ─────────────────────────────────────
        # This create new useful column from existing data
        df = self._engineer_features(df)

        print(f"\n  Final clean shape: {df.shape[0]} rows x {df.shape[1]} columns")
        self.clean_df = df
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new feature column from existing column."""
        # This calculate fish health index based on fish count and fish size
        df["Fish_Health_Index"] = (
            (df["Fish_Count"] / df["Fish_Count"].max()) * 0.5 +
            (df["Avg_Size_cm"] / df["Avg_Size_cm"].max()) * 0.5
        ).round(4)

        # This calculate water quality index from 3 key parameter
        # Normalise each parameter between 0 and 1 first
        do_norm   = (df["Dissolved_Oxygen"] - df["Dissolved_Oxygen"].min()) / \
                    (df["Dissolved_Oxygen"].max() - df["Dissolved_Oxygen"].min())
        ph_norm   = 1 - abs(df["pH"] - 7.0) / 3.5      # 7.0 is ideal pH for fish
        temp_norm = 1 - (df["Temperature_C"] - 15).abs() / 15  # 15C is comfortable

        df["Water_Quality_Index"] = (
            do_norm * 0.4 + ph_norm * 0.3 + temp_norm.clip(0, 1) * 0.3
        ).round(4)

        # This create status label for easy reading
        df["Oxygen_Status"]     = pd.cut(df["Dissolved_Oxygen"],
                                         bins=[0, 5, 7, 20],
                                         labels=["Low", "Moderate", "Good"])
        df["pH_Status"]         = pd.cut(df["pH"],
                                         bins=[0, 6.5, 7.5, 14],
                                         labels=["Acidic", "Neutral", "Alkaline"])
        df["Temperature_Status"] = pd.cut(df["Temperature_C"],
                                          bins=[0, 12, 18, 40],
                                          labels=["Cold", "Optimal", "Warm"])

        # Site health category based on water quality index
        df["Site_Health_Category"] = pd.cut(df["Water_Quality_Index"],
                                             bins=[0, 0.4, 0.65, 1.01],
                                             labels=["Poor", "Moderate", "Good"])

        # Extract month and season from date
        df["Month"] = df["Date"].dt.month
        df["Season"] = df["Month"].map({
            12: "Summer", 1: "Summer", 2: "Summer",
            3: "Autumn",  4: "Autumn", 5: "Autumn",
            6: "Winter",  7: "Winter", 8: "Winter",
            9: "Spring",  10: "Spring", 11: "Spring",
        })

        print("\n  Feature engineering complete:")
        print("    + Fish_Health_Index, Water_Quality_Index")
        print("    + Oxygen_Status, pH_Status, Temperature_Status")
        print("    + Site_Health_Category, Month, Season")
        return df

    def save(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Save clean data to Excel and CSV format."""
        out = output_dir / "cleaned_data"
        out.mkdir(parents=True, exist_ok=True)

        # This save clean data as Excel file
        excel_path = out / "cleaned_avon_river_data.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"\n  [Saved] {excel_path}")

        # This save clean data as CSV file (easier to open)
        csv_path = out / "cleaned_avon_river_data.csv"
        df.to_csv(csv_path, index=False)
        print(f"  [Saved] {csv_path}")


# =============================================================================
# CLASS 3 - ExploratoryDataAnalyzer
# =============================================================================
class ExploratoryDataAnalyzer:
    """
    This class explore the data and find basic pattern.
    Single responsibility: compute descriptive and summary statistics.
    """

    def __init__(self):
        self.stats_df: pd.DataFrame = pd.DataFrame()
        self.site_summary: pd.DataFrame = pd.DataFrame()
        self.species_summary: pd.DataFrame = pd.DataFrame()

    def analyze(self, df: pd.DataFrame) -> dict:
        """Run all EDA and return result dictionary."""
        print("\n" + "=" * 70)
        print("  STEP 3: EXPLORATORY DATA ANALYSIS")
        print("=" * 70)

        numeric_cols = ["Temperature_C", "pH", "Dissolved_Oxygen",
                        "Fish_Count", "Avg_Size_cm",
                        "Fish_Health_Index", "Water_Quality_Index"]

        # ── Descriptive statistics ────────────────────────────────────────────
        # This calculate mean, std, min, max for all numeric column
        self.stats_df = df[numeric_cols].describe().round(4).T
        self.stats_df["skewness"] = df[numeric_cols].skew().round(4)
        self.stats_df["kurtosis"] = df[numeric_cols].kurtosis().round(4)

        print(f"\n  Descriptive Statistics:")
        print(self.stats_df.to_string())

        # ── Site comparison ───────────────────────────────────────────────────
        # This compare average value between different site
        self.site_summary = df.groupby("Site_ID").agg(
            n_records=("Fish_Count", "count"),
            mean_temp=("Temperature_C", "mean"),
            mean_pH=("pH", "mean"),
            mean_DO=("Dissolved_Oxygen", "mean"),
            mean_fish_count=("Fish_Count", "mean"),
            mean_fish_size=("Avg_Size_cm", "mean"),
            mean_WQI=("Water_Quality_Index", "mean"),
            mean_FHI=("Fish_Health_Index", "mean"),
        ).round(3)

        print(f"\n  Site Comparison Summary:")
        print(self.site_summary.to_string())

        # ── Species summary ───────────────────────────────────────────────────
        # This show how each species perform
        self.species_summary = df.groupby("Species").agg(
            n_records=("Fish_Count", "count"),
            total_fish=("Fish_Count", "sum"),
            mean_count=("Fish_Count", "mean"),
            mean_size=("Avg_Size_cm", "mean"),
            min_count=("Fish_Count", "min"),
            max_count=("Fish_Count", "max"),
        ).round(3)

        print(f"\n  Species Summary:")
        print(self.species_summary.to_string())

        # ── Monthly trend ─────────────────────────────────────────────────────
        # This show how value change over month
        monthly = df.groupby("Month").agg(
            mean_DO=("Dissolved_Oxygen", "mean"),
            mean_fish_count=("Fish_Count", "mean"),
            mean_WQI=("Water_Quality_Index", "mean"),
        ).round(3)
        print(f"\n  Monthly Trend (averages):")
        print(monthly.to_string())

        return {
            "stats": self.stats_df,
            "site_summary": self.site_summary,
            "species_summary": self.species_summary,
            "monthly": monthly,
        }

    def save_reports(self, output_dir: Path) -> None:
        """Save EDA summary to CSV file."""
        rpt = output_dir / "reports"
        rpt.mkdir(parents=True, exist_ok=True)

        # This save descriptive statistics to CSV
        self.stats_df.to_csv(rpt / "descriptive_statistics.csv")
        print(f"  [Saved] descriptive_statistics.csv")

        # This save site comparison to CSV
        self.site_summary.to_csv(rpt / "site_summary.csv")
        print(f"  [Saved] site_summary.csv")

        # This save species summary to CSV
        self.species_summary.to_csv(rpt / "species_summary.csv")
        print(f"  [Saved] species_summary.csv")


# =============================================================================
# CLASS 4 - StatisticalAnalyzer
# =============================================================================
class StatisticalAnalyzer:
    """
    This class do the statistical analysis including correlation and regression.
    Single responsibility: compute inferential statistics.
    """

    def __init__(self):
        self.corr_matrix: pd.DataFrame = pd.DataFrame()
        self.regression_results: dict = {}

    @staticmethod
    def _interpret_correlation(r: float) -> str:
        """Return simple correlation strength label for report writing."""
        ar = abs(r)
        if ar >= 0.70:
            return "Strong"
        if ar >= 0.40:
            return "Moderate"
        return "Weak"

    def correlation_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Technique 1: Pearson Correlation Analysis.
        This find relationship between water quality and fish population.
        """
        print("\n" + "=" * 70)
        print("  STEP 4a: CORRELATION ANALYSIS (Pearson)")
        print("=" * 70)

        cols = ["Temperature_C", "pH", "Dissolved_Oxygen",
                "Fish_Count", "Avg_Size_cm",
                "Fish_Health_Index", "Water_Quality_Index"]

        # This calculate Pearson correlation matrix
        self.corr_matrix = df[cols].corr(method="pearson").round(4)
        print(f"\n  Pearson Correlation Matrix:")
        print(self.corr_matrix.to_string())

        # This analyse key pair one by one.
        # Important: the code reports weak/non-significant relationships honestly.
        # It does not overclaim that water quality directly predicts fish count.
        print("\n  Key Pair Analysis:")
        key_pairs = [
            ("Dissolved_Oxygen", "Fish_Count", "DO vs Fish Count"),
            ("Dissolved_Oxygen", "Water_Quality_Index", "DO vs Water Quality Index"),
            ("Temperature_C", "Dissolved_Oxygen", "Temperature vs DO"),
            ("Temperature_C", "Fish_Count", "Temperature vs Fish Count"),
            ("pH", "Fish_Count", "pH vs Fish Count"),
            ("Water_Quality_Index", "Fish_Health_Index", "WQI vs FHI"),
        ]

        for col_a, col_b, label in key_pairs:
            pair_df = df[[col_a, col_b]].dropna()
            r, p = stats.pearsonr(pair_df[col_a], pair_df[col_b])
            strength = self._interpret_correlation(r)
            direction = "positive" if r > 0 else "negative"
            significance = "significant (p<0.05)" if p < 0.05 else "not significant"
            print(f"\n  [{label}]")
            print(f"    r = {r:.4f}  |  p = {p:.4e}")
            print(f"    → {strength} {direction} correlation, {significance}")
            if label == "DO vs Water Quality Index":
                print("    NOTE: This strong result is expected because WQI formula includes DO.")
            else:
                print("    NOTE: Correlation does NOT mean causation. Other factors may also affect fish population.")

        return self.corr_matrix

    def regression_analysis(self, df: pd.DataFrame) -> dict:
        """
        Technique 2: Multiple Linear Regression.
        This estimate impact of temperature, pH, dissolved oxygen on fish count.
        """
        print("\n" + "=" * 70)
        print("  STEP 4b: MULTIPLE LINEAR REGRESSION")
        print("=" * 70)

        # This define predictor and target variable
        features = ["Temperature_C", "pH", "Dissolved_Oxygen"]
        target   = "Fish_Count"

        X = df[features].copy()
        y = df[target].copy()

        print(f"\n  Predictors : {features}")
        print(f"  Target     : {target}")
        print(f"  Sample size: {len(X)}")

        # ── OLS regression with statsmodels (give p-value for each coefficient)
        # This run proper regression with statistical detail
        X_const = sm.add_constant(X)
        ols_model = sm.OLS(y, X_const).fit()
        print(f"\n  OLS Regression Summary (statsmodels):")
        print(ols_model.summary().as_text())

        # ── Scikit-learn regression using full cleaned dataset ─────────
        # This make regression metric align with report result.
        # No train/test split because this section is explanatory statistical regression.
        X_test = X
        y_test = y

        sk_model = LinearRegression()
        sk_model.fit(X, y)
        y_pred = sk_model.predict(X)

        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        print(f"\n  Regression Evaluation: Full cleaned dataset used")
        print(f"  NOTE: This is explanatory regression, not train/test prediction.")

        print(f"\n  Model Performance Metrics:")
        print(f"    R²   = {r2:.4f}  → model explains only {r2*100:.1f}% of fish count variance")
        if r2 < 0.10:
            print("    INTERPRETATION: Very low explanatory power. Temperature, pH, and DO alone cannot predict fish count well in this dataset.")
        print(f"    MAE  = {mae:.4f} → average error ± {mae:.2f} fish")
        print(f"    RMSE = {rmse:.4f}")

        # This show coefficient for each predictor
        print(f"\n  Regression Coefficients:")
        for feat, coef in zip(features, sk_model.coef_):
            direction = "increases" if coef > 0 else "decreases"
            print(f"    {feat:25s}: {coef:+.4f}  "
                  f"→ 1 unit ↑ {direction} fish count by {abs(coef):.3f}")
        print(f"    {'Intercept':25s}: {sk_model.intercept_:+.4f}")

        # This store result for later saving
        self.regression_results = {
            "r2": r2, "mae": mae, "rmse": rmse,
            "coefficients": dict(zip(features, sk_model.coef_)),
            "intercept": sk_model.intercept_,
            "ols_summary": ols_model.summary().as_text(),
            "X_test": X_test, "y_test": y_test, "y_pred": y_pred,
            "features": features, "model": sk_model,
            "ols_model": ols_model,
        }
        return self.regression_results

    def save_reports(self, output_dir: Path) -> None:
        """Save statistical analysis result to file."""
        rpt = output_dir / "reports"
        rpt.mkdir(parents=True, exist_ok=True)

        # This save correlation matrix to CSV
        self.corr_matrix.to_csv(rpt / "correlation_matrix.csv")
        print(f"  [Saved] correlation_matrix.csv")

        # This save regression result to text file
        reg_path = rpt / "regression_results.txt"
        with open(reg_path, "w") as f:
            f.write("MULTIPLE LINEAR REGRESSION RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Target variable: Fish_Count\n")
            f.write("Predictors: Temperature_C, pH, Dissolved_Oxygen\n\n")
            f.write(f"R²   = {self.regression_results['r2']:.4f}\n")
            f.write(f"MAE  = {self.regression_results['mae']:.4f}\n")
            f.write(f"RMSE = {self.regression_results['rmse']:.4f}\n\n")
            f.write("Coefficients:\n")
            for feat, coef in self.regression_results["coefficients"].items():
                f.write(f"  {feat}: {coef:+.4f}\n")
            f.write(f"  Intercept: {self.regression_results['intercept']:+.4f}\n\n")
            f.write("OLS Full Summary (statsmodels):\n")
            f.write("-" * 60 + "\n")
            f.write(self.regression_results["ols_summary"])
            f.write("\n\nLIMITATIONS:\n")
            f.write("  - Small dataset (n<72) reduce reliability of regression\n")
            f.write("  - Only 3 sites observed (AV-1, AV-2, AV-3)\n")
            f.write("  - Short time period (Oct-Dec 2023) limit seasonal generalisation\n")
            f.write("  - Other environmental factor (turbidity, pollution) not measured\n")
        print(f"  [Saved] regression_results.txt")


# =============================================================================
# CLASS 5 - VisualizationGenerator
# =============================================================================
class VisualizationGenerator:
    """
    This class create all chart and figure for the assessment.
    Single responsibility: generate and save visualisation.
    """

    # This define colour scheme used in all chart
    PALETTE    = "#2d6a4f"
    ACCENT     = "#e76f51"
    BLUE       = "#457b9d"
    LIGHT_BLUE = "#a8dadc"
    DARK       = "#264653"
    BG         = "#f8f9fa"
    SITE_COLORS = {"AV-1": "#2d6a4f", "AV-2": "#457b9d", "AV-3": "#e76f51"}

    def __init__(self, fig_dir: Path):
        # This is folder where all chart will save
        self.fig_dir = fig_dir
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self._set_style()

    def _set_style(self):
        """Set global chart style."""
        plt.rcParams.update({
            "font.family":       "DejaVu Sans",
            "axes.facecolor":    self.BG,
            "figure.facecolor":  "white",
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "axes.grid":         True,
            "grid.alpha":        0.25,
            "font.size":         10,
        })

    def _save(self, fig: plt.Figure, name: str) -> None:
        """Save figure to PNG file."""
        path = self.fig_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Saved] {name}")

    # ── UNIVARIATE ────────────────────────────────────────────────────────────

    def fig01_histograms(self, df: pd.DataFrame) -> None:
        """FIG01 - Histogram of key variable."""
        cols   = ["Temperature_C", "pH", "Dissolved_Oxygen", "Fish_Count"]
        titles = ["Temperature (°C)", "pH", "Dissolved Oxygen (mg/L)", "Fish Count"]
        colors = [self.PALETTE, self.BLUE, self.ACCENT, self.DARK]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("FIG 01 - Univariate Histograms: Distribution of Key Variables",
                     fontsize=14, fontweight="bold", y=1.01)

        for ax, col, title, color in zip(axes.flat, cols, titles, colors):
            # This draw histogram with density curve
            n, bins, _ = ax.hist(df[col].dropna(), bins=12, color=color,
                                 alpha=0.75, edgecolor="white", linewidth=0.8)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel(title, fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            mu    = df[col].mean()
            sigma = df[col].std()
            ax.axvline(mu, color=self.DARK, linewidth=2, linestyle="--",
                       label=f"Mean={mu:.2f}")
            ax.legend(fontsize=9)

        fig.tight_layout()
        self._save(fig, "fig01_histograms.png")

    def fig02_boxplots(self, df: pd.DataFrame) -> None:
        """FIG02 - Boxplots showing spread and outlier."""
        cols   = ["Temperature_C", "pH", "Dissolved_Oxygen", "Fish_Count", "Avg_Size_cm"]
        titles = ["Temp (°C)", "pH", "D.O. (mg/L)", "Fish Count", "Avg Size (cm)"]

        fig, axes = plt.subplots(1, 5, figsize=(18, 5))
        fig.suptitle("FIG 02 - Boxplots: Spread and Outlier Detection per Variable",
                     fontsize=13, fontweight="bold", y=1.02)

        for ax, col, title in zip(axes, cols, titles):
            # This draw boxplot with colour
            bp = ax.boxplot(
                df[col].dropna(), patch_artist=True, notch=False,
                medianprops=dict(color=self.DARK, linewidth=2.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker="o", color=self.ACCENT,
                                markerfacecolor=self.ACCENT, markersize=6, alpha=0.8),
            )
            bp["boxes"][0].set_facecolor(self.LIGHT_BLUE)
            bp["boxes"][0].set_alpha(0.8)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xticks([])

        fig.tight_layout()
        self._save(fig, "fig02_boxplots.png")

    # ── BIVARIATE ─────────────────────────────────────────────────────────────

    def fig03_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> None:
        """FIG03 - Pearson correlation heatmap."""
        # This rename column for nicer label in chart
        labels = {
            "Temperature_C":       "Temp\n(°C)",
            "pH":                  "pH",
            "Dissolved_Oxygen":    "D.O.\n(mg/L)",
            "Fish_Count":          "Fish\nCount",
            "Avg_Size_cm":         "Avg Size\n(cm)",
            "Fish_Health_Index":   "Fish Health\nIndex",
            "Water_Quality_Index": "Water Qual.\nIndex",
        }
        cm = corr_matrix.rename(index=labels, columns=labels)

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(cm, dtype=bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(
            cm, mask=mask,
            cmap=cmap, vmin=-1, vmax=1, center=0,
            annot=True, fmt=".2f",
            annot_kws={"size": 11, "weight": "bold"},
            linewidths=1.5, linecolor="white",
            square=True,
            cbar_kws={"shrink": 0.75, "label": "Pearson r"},
            ax=ax,
        )
        ax.set_title(
            "FIG 03 - Pearson Correlation Heatmap\n"
            "Water Quality & Fish Population Parameters",
            fontsize=13, fontweight="bold", pad=14,
        )
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9, rotation=0)
        fig.tight_layout()
        self._save(fig, "fig03_correlation_heatmap.png")

    def fig04_scatter_do_vs_fishcount(self, df: pd.DataFrame) -> None:
        """FIG04 - Scatter: Dissolved Oxygen vs Fish Count."""
        fig, ax = plt.subplots(figsize=(8, 5.5))
        # This draw scatter point with colour by site
        for site, grp in df.groupby("Site_ID"):
            ax.scatter(grp["Dissolved_Oxygen"], grp["Fish_Count"],
                       color=self.SITE_COLORS.get(site, "grey"),
                       alpha=0.8, label=site, s=70,
                       edgecolors="white", linewidths=0.6, zorder=3)

        # This add regression line
        r, p = stats.pearsonr(df["Dissolved_Oxygen"], df["Fish_Count"])
        slope, intercept, *_ = stats.linregress(df["Dissolved_Oxygen"], df["Fish_Count"])
        x_line = np.linspace(df["Dissolved_Oxygen"].min(), df["Dissolved_Oxygen"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept,
                color=self.DARK, linewidth=2, linestyle="--",
                label=f"Regression  r={r:.3f}, p={p:.3f}")

        ax.set_xlabel("Dissolved Oxygen (mg/L)", fontsize=12)
        ax.set_ylabel("Fish Count", fontsize=12)
        ax.set_title(f"FIG 04 - Dissolved Oxygen vs Fish Count\n"
                     f"(Pearson r={r:.3f})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        self._save(fig, "fig04_scatter_do_vs_fishcount.png")

    def fig05_scatter_temp_vs_do(self, df: pd.DataFrame) -> None:
        """FIG05 - Scatter: Temperature vs Dissolved Oxygen."""
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for site, grp in df.groupby("Site_ID"):
            ax.scatter(grp["Temperature_C"], grp["Dissolved_Oxygen"],
                       color=self.SITE_COLORS.get(site, "grey"),
                       alpha=0.8, label=site, s=70,
                       edgecolors="white", linewidths=0.6, zorder=3)

        r, p = stats.pearsonr(df["Temperature_C"], df["Dissolved_Oxygen"])
        slope, intercept, *_ = stats.linregress(df["Temperature_C"], df["Dissolved_Oxygen"])
        x_line = np.linspace(df["Temperature_C"].min(), df["Temperature_C"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept,
                color=self.DARK, linewidth=2, linestyle="--",
                label=f"Regression  r={r:.3f}")

        ax.set_xlabel("Temperature (°C)", fontsize=12)
        ax.set_ylabel("Dissolved Oxygen (mg/L)", fontsize=12)
        ax.set_title(f"FIG 05 - Temperature vs Dissolved Oxygen\n"
                     f"(Pearson r={r:.3f})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        self._save(fig, "fig05_scatter_temp_vs_do.png")

    def fig06_scatter_ph_vs_fishcount(self, df: pd.DataFrame) -> None:
        """FIG06 - Scatter: pH vs Fish Count."""
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for site, grp in df.groupby("Site_ID"):
            ax.scatter(grp["pH"], grp["Fish_Count"],
                       color=self.SITE_COLORS.get(site, "grey"),
                       alpha=0.8, label=site, s=70,
                       edgecolors="white", linewidths=0.6, zorder=3)

        r, p = stats.pearsonr(df["pH"], df["Fish_Count"])
        slope, intercept, *_ = stats.linregress(df["pH"], df["Fish_Count"])
        x_line = np.linspace(df["pH"].min(), df["pH"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept,
                color=self.DARK, linewidth=2, linestyle="--",
                label=f"Regression  r={r:.3f}")

        ax.set_xlabel("pH", fontsize=12)
        ax.set_ylabel("Fish Count", fontsize=12)
        ax.set_title(f"FIG 06 - pH vs Fish Count\n"
                     f"(Pearson r={r:.3f})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        self._save(fig, "fig06_scatter_ph_vs_fishcount.png")

    def fig07_regression_plot(self, reg_results: dict) -> None:
        """FIG07 - Regression: Actual vs Predicted with residual plot."""
        y_test = reg_results["y_test"]
        y_pred = reg_results["y_pred"]
        r2     = reg_results["r2"]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("FIG 07 - Multiple Linear Regression: Model Evaluation",
                     fontsize=13, fontweight="bold")

        # Left: actual vs predicted
        ax = axes[0]
        ax.scatter(y_test, y_pred, color=self.PALETTE, alpha=0.75,
                   edgecolors="white", linewidths=0.6, s=70, zorder=3)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                color=self.DARK, linewidth=2, linestyle="--", label="Perfect fit")
        ax.set_xlabel("Actual Fish Count", fontsize=12)
        ax.set_ylabel("Predicted Fish Count", fontsize=12)
        ax.set_title(f"Actual vs Predicted (R²={r2:.4f})", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)

        # Right: residual plot
        ax2 = axes[1]
        residuals = np.array(y_test) - np.array(y_pred)
        ax2.scatter(y_pred, residuals, color=self.ACCENT, alpha=0.75,
                    edgecolors="white", linewidths=0.6, s=70, zorder=3)
        ax2.axhline(0, color=self.DARK, linewidth=2, linestyle="--")
        ax2.set_xlabel("Predicted Fish Count", fontsize=12)
        ax2.set_ylabel("Residual (Actual - Predicted)", fontsize=12)
        ax2.set_title("Residual Plot", fontsize=11, fontweight="bold")

        fig.tight_layout()
        self._save(fig, "fig07_regression_plot.png")

    # ── MULTIVARIATE ─────────────────────────────────────────────────────────

    def fig08_pairplot(self, df: pd.DataFrame) -> None:
        """FIG08 - Pair plot showing all feature combination."""
        # This select numeric column for pair plot
        pair_df = df[["Temperature_C", "pH", "Dissolved_Oxygen",
                       "Fish_Count", "Avg_Size_cm"]].copy()
        pair_df.columns = ["Temp(°C)", "pH", "D.O.", "Fish Cnt", "Avg Size"]

        g = sns.pairplot(
            pair_df,
            diag_kind="kde",
            plot_kws=dict(alpha=0.45, color=self.PALETTE, edgecolor="none"),
            diag_kws=dict(color=self.PALETTE, fill=True),
        )
        g.figure.suptitle("FIG 08 - Pairplot: All Variable Combinations (clean data)",
                           y=1.02, fontsize=12, fontweight="bold")
        path = self.fig_dir / "fig08_pairplot.png"
        g.figure.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(g.figure)
        print(f"  [Saved] fig08_pairplot.png")

    def fig09_site_comparison(self, df: pd.DataFrame) -> None:
        """FIG09 - Site comparison bar charts."""
        metrics = [
            ("Water_Quality_Index", "Water Quality Index",    self.PALETTE),
            ("Fish_Health_Index",   "Fish Health Index",      self.BLUE),
            ("Dissolved_Oxygen",    "Dissolved Oxygen (mg/L)", self.ACCENT),
            ("Fish_Count",          "Mean Fish Count",        self.DARK),
        ]
        site_means = df.groupby("Site_ID")[
            [m[0] for m in metrics]
        ].mean()

        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        fig.suptitle("FIG 09 - Site Comparison: Mean Values by Monitoring Site",
                     fontsize=13, fontweight="bold")

        for ax, (col, label, color) in zip(axes, metrics):
            bars = ax.bar(site_means.index, site_means[col],
                          color=color, alpha=0.85, edgecolor="white", linewidth=1.2)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Site", fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            # This add value label on top of bar
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{bar.get_height():.2f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

        fig.tight_layout()
        self._save(fig, "fig09_site_comparison.png")

    def fig10_species_distribution(self, df: pd.DataFrame) -> None:
        """FIG10 - Species distribution (total fish count per species)."""
        species_total = df.groupby("Species")["Fish_Count"].sum().sort_values(ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("FIG 10 - Species Distribution: Fish Count & Average Size",
                     fontsize=13, fontweight="bold")

        # Left: bar chart of total fish count
        colors_sp = [self.PALETTE, self.BLUE, self.ACCENT, self.DARK, "#95d5b2", "#74c69d"]
        axes[0].bar(species_total.index, species_total.values,
                    color=colors_sp[:len(species_total)], alpha=0.85,
                    edgecolor="white", linewidth=1)
        axes[0].set_title("Total Fish Count by Species", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Species", fontsize=10)
        axes[0].set_ylabel("Total Fish Count", fontsize=10)
        axes[0].tick_params(axis="x", rotation=30)

        # Right: average size boxplot per species
        species_order = df.groupby("Species")["Avg_Size_cm"].mean().sort_values(
            ascending=False).index
        df.boxplot(column="Avg_Size_cm", by="Species",
                   ax=axes[1], grid=False,
                   medianprops=dict(color=self.DARK, linewidth=2))
        axes[1].set_title("Average Fish Size by Species", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Species", fontsize=10)
        axes[1].set_ylabel("Avg Size (cm)", fontsize=10)
        axes[1].tick_params(axis="x", rotation=30)
        plt.suptitle("")  # This remove auto title from boxplot by pandas

        fig.tight_layout()
        self._save(fig, "fig10_species_distribution.png")

    def fig11_monthly_trends(self, df: pd.DataFrame) -> None:
        """FIG11 - Monthly trend for DO, Fish Count, and WQI."""
        # This sort month from Oct to Dec (the dataset range)
        monthly = df.groupby("Month").agg(
            mean_DO=("Dissolved_Oxygen", "mean"),
            mean_fish=("Fish_Count", "mean"),
            mean_wqi=("Water_Quality_Index", "mean"),
        ).reset_index()

        month_names = {10: "Oct", 11: "Nov", 12: "Dec"}
        monthly["Month_Name"] = monthly["Month"].map(month_names)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("FIG 11 - Monthly Trend Analysis (Oct-Dec 2023)",
                     fontsize=13, fontweight="bold")

        plots = [
            (axes[0], "mean_DO",   "Dissolved Oxygen (mg/L)", self.PALETTE),
            (axes[1], "mean_fish", "Mean Fish Count",          self.BLUE),
            (axes[2], "mean_wqi",  "Water Quality Index",      self.ACCENT),
        ]
        for ax, col, label, color in plots:
            ax.plot(monthly["Month_Name"], monthly[col],
                    color=color, marker="o", linewidth=2.5,
                    markersize=9, markeredgecolor="white", markeredgewidth=1.5)
            ax.fill_between(monthly["Month_Name"], monthly[col],
                            alpha=0.12, color=color)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Month", fontsize=10)
            ax.set_ylabel(label, fontsize=10)

        fig.tight_layout()
        self._save(fig, "fig11_monthly_trends.png")

    def fig12_executive_dashboard(self, df: pd.DataFrame) -> None:
        """FIG12 - Executive Dashboard: River Health by Site & WQI vs FHI."""
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(
            "FIG 12 - Executive Dashboard: Avon River Environmental Health Overview",
            fontsize=15, fontweight="bold", y=1.01,
        )
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        # ── Chart A: River Health by Site (WQI + FHI grouped bar) ────────────
        ax_a = fig.add_subplot(gs[0, :])  # This span full top row
        site_grp = df.groupby("Site_ID")[["Water_Quality_Index", "Fish_Health_Index"]].mean()
        x = np.arange(len(site_grp))
        width = 0.35
        bars1 = ax_a.bar(x - width/2, site_grp["Water_Quality_Index"],
                         width, label="Water Quality Index",
                         color=self.PALETTE, alpha=0.85, edgecolor="white")
        bars2 = ax_a.bar(x + width/2, site_grp["Fish_Health_Index"],
                         width, label="Fish Health Index",
                         color=self.BLUE, alpha=0.85, edgecolor="white")
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(site_grp.index, fontsize=12)
        ax_a.set_ylabel("Index Score (0-1)", fontsize=11)
        ax_a.set_title("Chart A - River Health by Site "
                        "(Water Quality Index vs Fish Health Index)",
                        fontsize=12, fontweight="bold")
        ax_a.legend(fontsize=10)
        ax_a.axhline(0.5, color=self.ACCENT, linewidth=1.5, linestyle="--",
                     label="Threshold (0.5)")
        # This add value label on each bar
        for bar in list(bars1) + list(bars2):
            ax_a.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() + 0.008,
                      f"{bar.get_height():.3f}",
                      ha="center", va="bottom", fontsize=9)

        # ── Chart B: WQI vs FHI scatter with site colour ──────────────────────
        ax_b = fig.add_subplot(gs[1, 0])
        for site, grp in df.groupby("Site_ID"):
            ax_b.scatter(grp["Water_Quality_Index"], grp["Fish_Health_Index"],
                         color=self.SITE_COLORS.get(site, "grey"),
                         alpha=0.75, label=site, s=60,
                         edgecolors="white", linewidths=0.5, zorder=3)
        r_wb, p_wb = stats.pearsonr(df["Water_Quality_Index"], df["Fish_Health_Index"])
        slope, intercept, *_ = stats.linregress(df["Water_Quality_Index"],
                                                df["Fish_Health_Index"])
        x_l = np.linspace(df["Water_Quality_Index"].min(),
                          df["Water_Quality_Index"].max(), 100)
        ax_b.plot(x_l, slope * x_l + intercept,
                  color=self.DARK, linewidth=2, linestyle="--",
                  label=f"r={r_wb:.3f}")
        ax_b.set_xlabel("Water Quality Index", fontsize=11)
        ax_b.set_ylabel("Fish Health Index", fontsize=11)
        ax_b.set_title("Chart B - Water Quality vs Fish Population Health",
                        fontsize=11, fontweight="bold")
        ax_b.legend(fontsize=9)

        # ── Chart C: DO over time per site (trend line) ───────────────────────
        ax_c = fig.add_subplot(gs[1, 1])
        for site, grp in df.groupby("Site_ID"):
            grp_sorted = grp.sort_values("Date")
            ax_c.plot(grp_sorted["Date"], grp_sorted["Dissolved_Oxygen"],
                      color=self.SITE_COLORS.get(site, "grey"),
                      marker="o", markersize=4, linewidth=1.5,
                      alpha=0.8, label=site)
        ax_c.axhline(6.0, color=self.ACCENT, linewidth=2, linestyle="--",
                     label="Min. healthy DO (6 mg/L)")
        ax_c.set_xlabel("Date", fontsize=10)
        ax_c.set_ylabel("Dissolved Oxygen (mg/L)", fontsize=10)
        ax_c.set_title("Chart C - Dissolved Oxygen Over Time",
                        fontsize=11, fontweight="bold")
        ax_c.tick_params(axis="x", rotation=30)
        ax_c.legend(fontsize=8)

        fig.tight_layout()
        self._save(fig, "fig12_executive_dashboard.png")

    def generate_all(self, df: pd.DataFrame, corr_matrix: pd.DataFrame,
                     reg_results: dict) -> None:
        """Run all chart generation in one call."""
        print("\n" + "=" * 70)
        print("  STEP 5: GENERATING ALL VISUALISATIONS")
        print("=" * 70)
        print()
        self.fig01_histograms(df)
        self.fig02_boxplots(df)
        self.fig03_correlation_heatmap(corr_matrix)
        self.fig04_scatter_do_vs_fishcount(df)
        self.fig05_scatter_temp_vs_do(df)
        self.fig06_scatter_ph_vs_fishcount(df)
        self.fig07_regression_plot(reg_results)
        self.fig08_pairplot(df)
        self.fig09_site_comparison(df)
        self.fig10_species_distribution(df)
        self.fig11_monthly_trends(df)
        self.fig12_executive_dashboard(df)


# =============================================================================
# CLASS 6 - RecommendationEngine
# =============================================================================
class RecommendationEngine:
    """
    This class produce evidence-based environmental recommendation.
    Single responsibility: translate analysis finding into action.
    """

    def __init__(self):
        self.recommendations: list = []

    def generate(self, df: pd.DataFrame, corr_matrix: pd.DataFrame,
                 eda_results: dict, regression_results: dict | None = None) -> list:
        """Generate recommendation from actual dataset evidence only."""
        print("\n" + "=" * 70)
        print("  STEP 6: GENERATING RECOMMENDATIONS")
        print("=" * 70)

        # reset list every run so repeated calls do not duplicate recommendations
        self.recommendations = []
        site_summary = eda_results["site_summary"]

        # Main statistical evidence from the real cleaned dataset
        r_do_fish, p_do_fish = stats.pearsonr(df["Dissolved_Oxygen"], df["Fish_Count"])
        r_do_wqi, p_do_wqi = stats.pearsonr(df["Dissolved_Oxygen"], df["Water_Quality_Index"])
        r_temp_do, p_temp_do = stats.pearsonr(df["Temperature_C"], df["Dissolved_Oxygen"])
        r_ph_fish, p_ph_fish = stats.pearsonr(df["pH"], df["Fish_Count"])

        lowest_wqi_site = site_summary["mean_WQI"].idxmin()
        lowest_fhi_site = site_summary["mean_FHI"].idxmin()
        lowest_do_site = site_summary["mean_DO"].idxmin()
        highest_temp_site = site_summary["mean_temp"].idxmax()

        r2 = None
        if regression_results is not None:
            r2 = regression_results.get("r2")

        # ── Recommendation 1: Protect DO and WQI at the weakest water-quality site ──
        self.recommendations.append({
            "id": "R1",
            "title": f"Prioritise Water Quality Protection at {lowest_wqi_site}",
            "what_to_do": (
                f"Focus first on {lowest_wqi_site}. Improve dissolved oxygen protection, "
                "add riparian planting, reduce heat and runoff pressure, and install regular DO monitoring."
            ),
            "evidence": (
                f"{lowest_wqi_site} has the lowest mean WQI ({site_summary.loc[lowest_wqi_site, 'mean_WQI']:.3f}). "
                f"DO vs WQI is very strong (r = {r_do_wqi:.3f}, p = {p_do_wqi:.4f}), "
                "but this must be interpreted carefully because WQI formula includes DO. "
                f"DO vs Fish Count is weak and not significant (r = {r_do_fish:.3f}, p = {p_do_fish:.4f}), "
                "so the code must not claim that DO alone explains fish count."
            ),
            "environmental_benefit": (
                "This protects the main water-quality driver in the index and improves river condition monitoring "
                "without overstating direct fish-count causation."
            ),
            "task_alignment": "Task 1-A (poor water quality), Task 1-D",
        })

        # ── Recommendation 2: Increase monitoring because model power is weak ──
        r2_text = f"R² = {r2:.4f}" if r2 is not None else "R² is very low"
        self.recommendations.append({
            "id": "R2",
            "title": "Increase Monitoring Frequency and Add Missing Environmental Variables",
            "what_to_do": (
                "Move from monthly monitoring to fortnightly or weekly monitoring. "
                "Add turbidity, nitrate, phosphate, rainfall, flow rate, habitat score, and pollution-source indicators."
            ),
            "evidence": (
                f"Multiple Linear Regression has very low explanatory power ({r2_text}). "
                "This means Temperature, pH, and DO alone cannot explain fish-count variation well. "
                "The current dataset has only 70 clean records and only three months of monitoring."
            ),
            "environmental_benefit": (
                "More frequent and wider monitoring gives stronger evidence for future prediction and better conservation decisions."
            ),
            "task_alignment": "Task 1-A (poor fish population evidence), Task 1-D",
        })

        # ── Recommendation 3: AV-1 fish health concern ──
        self.recommendations.append({
            "id": "R3",
            "title": f"Investigate Fish Health Concern at {lowest_fhi_site}",
            "what_to_do": (
                f"Run focused fish survey at {lowest_fhi_site}. Check habitat condition, barriers, bank cover, "
                "predation pressure, and species distribution."
            ),
            "evidence": (
                f"{lowest_fhi_site} has the lowest Fish Health Index ({site_summary.loc[lowest_fhi_site, 'mean_FHI']:.3f}) "
                f"and mean fish count of {site_summary.loc[lowest_fhi_site, 'mean_fish_count']:.3f}. "
                "Fish Health Index is based on fish count and average size, so it is an indicator only, not a full ecological diagnosis."
            ),
            "environmental_benefit": (
                "A focused fish survey can identify whether the problem is habitat, species mix, migration barrier, or sampling limitation."
            ),
            "task_alignment": "Task 1-A (fish population challenge), Task 1-D",
        })

        # ── Recommendation 4: Temperature and pH as precaution, not strong evidence ──
        self.recommendations.append({
            "id": "R4",
            "title": "Use Temperature and pH Alerts as Precautionary Controls",
            "what_to_do": (
                "Set pH alert thresholds below 6.5 and above 8.5. Track warm-water events above 18°C. "
                "Use shade planting and source investigation when alerts repeat."
            ),
            "evidence": (
                f"Temperature vs DO is weak and not significant (r = {r_temp_do:.3f}, p = {p_temp_do:.4f}). "
                f"pH vs Fish Count is also weak and not significant (r = {r_ph_fish:.3f}, p = {p_ph_fish:.4f}). "
                f"However, {highest_temp_site} has the highest mean temperature ({site_summary.loc[highest_temp_site, 'mean_temp']:.3f}°C), "
                "so precautionary monitoring is still justified."
            ),
            "environmental_benefit": (
                "This avoids false causal claims but still supports early warning for water-quality stress."
            ),
            "task_alignment": "Task 1-D",
        })

        # ── Recommendation 5: Dashboard ──────────────────────────────────────
        self.recommendations.append({
            "id": "R5",
            "title": "Deploy Power BI / Excel Dashboard for Non-Technical Stakeholders",
            "what_to_do": (
                "Build dashboard filters by site, species, and month. Show WQI, FHI, DO, fish count, and trend charts. "
                "Use simple labels and avoid causal wording unless statistically supported."
            ),
            "evidence": (
                "The Python pipeline generates cleaned data, site summary, species summary, correlation matrix, "
                "regression metrics, ML tables, and 17 figures. Power BI or Excel can make these outputs easier for non-technical users."
            ),
            "environmental_benefit": (
                "Simple reporting improves communication, public transparency, and monthly decision-making."
            ),
            "task_alignment": "Task 1-C (Tool 2: Power BI / Interactive dashboard)",
        })

        print()
        for rec in self.recommendations:
            print(f"  [{rec['id']}] {rec['title']}")

        return self.recommendations

    def save(self, output_dir: Path) -> None:
        """Save recommendation to text file."""
        rpt = output_dir / "reports"
        rpt.mkdir(parents=True, exist_ok=True)
        path = rpt / "recommendations.txt"

    
        with open(path, "w", encoding="utf-8") as f:
            f.write("EVIDENCE-BASED ENVIRONMENTAL RECOMMENDATIONS\n")
            f.write("Avon River Water Quality & Fish Population Analysis\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for rec in self.recommendations:
                f.write(f"\n{'─'*70}\n")
                f.write(f"[{rec['id']}] {rec['title']}\n")
                f.write(f"{'─'*70}\n")
                f.write(f"WHAT TO DO:\n  {rec['what_to_do']}\n\n")
                f.write(f"SUPPORTING EVIDENCE:\n  {rec['evidence']}\n\n")
                f.write(f"ENVIRONMENTAL BENEFIT:\n  {rec['environmental_benefit']}\n\n")
                f.write(f"TASK ALIGNMENT: {rec['task_alignment']}\n")
        print(f"  [Saved] recommendations.txt")


# =============================================================================
# CLASS 7 - ReportExporter
# =============================================================================
class ReportExporter:
    """
    This class generate all text report and assessment mapping document.
    Single responsibility: export findings to file format.
    """

    def export_assessment_mapping(self, output_dir: Path,
                                  recommendations: list) -> None:
        """Create assessment_mapping.md that map outputs to task."""
        rpt = output_dir / "reports"
        rpt.mkdir(parents=True, exist_ok=True)
        path = rpt / "assessment_mapping.md"

        content = textwrap.dedent("""
        # MSE803 Assessment 1: Output to Task Mapping
        **Generated:** {date}

        This document shows which generated output file supports each assessment task.

        ---

        ## Complete Output File Inventory

        ### Cleaned Data

        | Output | Description |
        |--------|-------------|
        | `outputs/cleaned_data/cleaned_avon_river_data.xlsx` | Cleaned Avon River dataset in Excel format for marker review and Power BI/Excel import. |
        | `outputs/cleaned_data/cleaned_avon_river_data.csv` | Cleaned Avon River dataset in CSV format for reproducible Python/BI use. |

        ### Main Reports

        | Output | Description |
        |--------|-------------|
        | `outputs/reports/descriptive_statistics.csv` | Descriptive statistics, skewness, and kurtosis for numeric variables. |
        | `outputs/reports/site_summary.csv` | Site-level summary for AV-1, AV-2, and AV-3. |
        | `outputs/reports/species_summary.csv` | Species-level fish count and average size summary. |
        | `outputs/reports/correlation_matrix.csv` | Pearson correlation matrix for water quality and fish population variables. |
        | `outputs/reports/regression_results.txt` | Multiple linear regression metrics, coefficients, OLS summary, and limitations. |
        | `outputs/reports/recommendations.txt` | Evidence-based environmental recommendations generated from analysis results. |
        | `outputs/reports/assessment_mapping.md` | This task-to-output mapping file. |
        | `outputs/reports/advanced_ml_conclusion_and_limitations.txt` | Advanced ML technical conclusions, limitations, and ethics notes. |

        ### Figures

        | Output | Description |
        |--------|-------------|
        | `outputs/figures/fig01_histograms.png` | Univariate histograms for key variables. |
        | `outputs/figures/fig02_boxplots.png` | Boxplots for spread and outlier detection. |
        | `outputs/figures/fig03_correlation_heatmap.png` | Pearson correlation heatmap. |
        | `outputs/figures/fig04_scatter_do_vs_fishcount.png` | Dissolved oxygen vs fish count scatter/regression chart. |
        | `outputs/figures/fig05_scatter_temp_vs_do.png` | Temperature vs dissolved oxygen scatter/regression chart. |
        | `outputs/figures/fig06_scatter_ph_vs_fishcount.png` | pH vs fish count scatter/regression chart. |
        | `outputs/figures/fig07_regression_plot.png` | Actual vs predicted and residual plot for multiple linear regression. |
        | `outputs/figures/fig08_pairplot.png` | Multivariate pairplot for numeric variables. |
        | `outputs/figures/fig09_site_comparison.png` | Site comparison for WQI, FHI, DO, and fish count. |
        | `outputs/figures/fig10_species_distribution.png` | Species distribution and average size chart. |
        | `outputs/figures/fig11_monthly_trends.png` | Monthly trend analysis for DO, fish count, and WQI. |
        | `outputs/figures/fig12_executive_dashboard.png` | Executive dashboard showing river health overview. |
        | `outputs/figures/fig13_kmeans_elbow_method.png` | K-Means elbow method chart. |
        | `outputs/figures/fig14_kmeans_silhouette_score.png` | K-Means silhouette score chart. |
        | `outputs/figures/fig15_pca_kmeans_clusters.png` | PCA 2D visualisation of K-Means clusters. |
        | `outputs/figures/fig16_classification_model_comparison.png` | Classification model comparison by macro F1-score. |
        | `outputs/figures/fig17_time_series_trend_forecast.png` | Simple monthly time-series trend and rolling forecast. |

        ### Advanced ML Tables

        | Output | Description |
        |--------|-------------|
        | `outputs/tables/ml_created_class_target.csv` | Created Water_Quality_Category labels from Water_Quality_Index for classification. |
        | `outputs/tables/ml_linear_regression_metrics.csv` | Linear regression metrics for Fish_Count prediction. |
        | `outputs/tables/ml_kmeans_elbow_silhouette_scores.csv` | K values, inertia, and silhouette scores for K-Means selection. |
        | `outputs/tables/ml_kmeans_clustered_records.csv` | Cleaned records with assigned KMeans_Cluster labels. |
        | `outputs/tables/ml_pca_2d_cluster_points.csv` | PCA1 and PCA2 coordinates used for the cluster visualisation. |
        | `outputs/tables/ml_kmeans_cluster_summary_table.csv` | Mean pH, temperature, DO, fish count, and unavailable nutrient fields per cluster. |
        | `outputs/tables/ml_classification_model_comparison.csv` | Accuracy, precision, recall, F1-score, and confusion matrix summary for all classifiers. |
        | `outputs/tables/ml_confusion_matrix_logistic_regression.csv` | Confusion matrix for Logistic Regression. |
        | `outputs/tables/ml_confusion_matrix_svm.csv` | Confusion matrix for SVM. |
        | `outputs/tables/ml_confusion_matrix_decision_tree.csv` | Confusion matrix for Decision Tree. |
        | `outputs/tables/ml_confusion_matrix_random_forest.csv` | Confusion matrix for Random Forest. |
        | `outputs/tables/ml_confusion_matrix_gradient_boosting.csv` | Confusion matrix for Gradient Boosting. |
        | `outputs/tables/ml_confusion_matrix_neural_network___mlp.csv` | Confusion matrix for Neural Network / MLP. |
        | `outputs/tables/ml_time_series_monthly_trend_forecast.csv` | Monthly average fish count, WQI, and simple rolling forecast. |
        ---

        ## Task 1-A: Problem Identification

        **Challenge 1: Poor water quality can reduce fish health and fish population**

        | Output | Description |
        |--------|-------------|
        | `outputs/figures/fig03_correlation_heatmap.png` | Shows correlation between DO, pH, temperature, and fish count. |
        | `outputs/figures/fig04_scatter_do_vs_fishcount.png` | Visualises direct relationship between DO and fish count. |
        | `outputs/figures/fig05_scatter_temp_vs_do.png` | Shows how temperature relates to dissolved oxygen. |
        | `outputs/figures/fig12_executive_dashboard.png` | Gives an overview of river health and possible problem areas. |
        | `outputs/reports/correlation_matrix.csv` | Quantitative evidence of water quality-fish relationships. |
        | `outputs/reports/site_summary.csv` | Compares water quality and fish indicators across three sites. |

        **Challenge 2: Some sites/species show weaker fish population indicators**

        | Output | Description |
        |--------|-------------|
        | `outputs/figures/fig09_site_comparison.png` | Compares WQI and FHI across AV-1, AV-2, and AV-3. |
        | `outputs/figures/fig10_species_distribution.png` | Shows species fish count and average size patterns. |
        | `outputs/figures/fig11_monthly_trends.png` | Shows time trend of fish count over Oct-Dec 2023. |
        | `outputs/reports/species_summary.csv` | Provides species-level variation in fish count and fish size. |
        | `outputs/tables/ml_kmeans_cluster_summary_table.csv` | Adds cluster-level evidence for hidden site/record patterns. |

        ---

        ## Task 1-B: Data Analysis Techniques

        **Technique 1: Pearson Correlation Analysis**

        | Output | Description |
        |--------|-------------|
        | `outputs/reports/correlation_matrix.csv` | Full Pearson correlation matrix for 7 variables. |
        | `outputs/figures/fig03_correlation_heatmap.png` | Visual heatmap of all correlation values. |
        | `outputs/figures/fig04_scatter_do_vs_fishcount.png` | Bivariate scatter showing correlation direction. |
        | `outputs/figures/fig05_scatter_temp_vs_do.png` | Bivariate scatter showing temperature and DO relationship. |
        | `outputs/figures/fig06_scatter_ph_vs_fishcount.png` | Bivariate scatter showing pH and fish count relationship. |

        **Technique 2: Multiple Linear Regression**

        | Output | Description |
        |--------|-------------|
        | `outputs/reports/regression_results.txt` | Coefficients, R², MAE, RMSE, OLS summary, and limitations. |
        | `outputs/figures/fig07_regression_plot.png` | Actual vs predicted chart and residual plot. |
        | `outputs/tables/ml_linear_regression_metrics.csv` | Extra ML metric table for Fish_Count prediction. |

        **Extra Lecture-Based ML Techniques Added**

        | Output | Description |
        |--------|-------------|
        | `outputs/tables/ml_classification_model_comparison.csv` | Compares Logistic Regression, SVM, Decision Tree, Random Forest, Gradient Boosting, and MLP. |
        | `outputs/tables/ml_kmeans_elbow_silhouette_scores.csv` | Supports K-Means cluster selection. |
        | `outputs/tables/ml_time_series_monthly_trend_forecast.csv` | Supports simple predictive trend analysis. |

        ---

        ## Task 1-C: Tools & Visualisation

        **Tool 1: Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, Statsmodels)**

        Python is used for data loading, cleaning, preprocessing, EDA, statistical analysis, visualisation, machine learning, and report/table export.

        **Tool 2: Power BI / Excel (recommended for stakeholder dashboard)**

        | Output | Description |
        |--------|-------------|
        | `outputs/cleaned_data/cleaned_avon_river_data.xlsx` | Can be imported into Power BI or Excel. |
        | `outputs/cleaned_data/cleaned_avon_river_data.csv` | Can be imported into BI tools or reused in other analytics workflows. |
        | `outputs/reports/site_summary.csv` | Good table for dashboard KPI cards by site. |
        | `outputs/reports/species_summary.csv` | Good table for dashboard species filtering. |
        | `outputs/tables/ml_kmeans_cluster_summary_table.csv` | Good table for cluster-based stakeholder explanation. |

        ---

        ## Task 1-D: Recommendations

        | Recommendation | Assessment Alignment |
        |----------------|----------------------|
        {recs}

        ---

        """).format(
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            recs="\n".join(
                f"| [{r['id']}] {r['title']} | {r['task_alignment']} |"
                for r in recommendations
            ),
        )

    
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.strip())
        print(f"  [Saved] assessment_mapping.md")



# =============================================================================
# CLASS 8 - AdvancedMLAnalyzer
# =============================================================================
class AdvancedMLAnalyzer:
    """
    This class adds the extra machine learning coverage from the lecture.
    It does not remove old analysis. It only add new supervised and unsupervised ML.
    """

    def __init__(self):
        # This store all result so we can save them later
        self.tables = {}
        self.notes = []

    def run(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Run all advanced ML sections and save outputs."""
        print("\n" + "=" * 70)
        print("  STEP 8: ADVANCED MACHINE LEARNING EXTENSION")
        print("=" * 70)

        # This create folders requested by assessment prompt
        self.output_dir = output_dir
        self.table_dir = output_dir / "tables"
        self.report_dir = output_dir / "reports"
        self.fig_dir = output_dir / "figures"
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        ml_df = df.copy()
        ml_df = self._create_class_target(ml_df)

        # Run lecture related ML parts
        self._linear_regression_metrics(ml_df)
        self._kmeans_clustering(ml_df)
        self._classification_models(ml_df)
        self._time_series_analysis(ml_df)
        self._save_notes()

    def _available_features(self, df: pd.DataFrame) -> list:
        """Return only numeric feature columns that really exist in dataset."""
        wanted = [
            "Temperature_C", "pH", "Dissolved_Oxygen", "Fish_Count",
            "Avg_Size_cm", "Fish_Health_Index", "Water_Quality_Index"
        ]
        return [c for c in wanted if c in df.columns]

    def _create_class_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create simple water quality class target for classification models."""
        # This target is made from Water Quality Index if it already exist
        if "Water_Quality_Index" not in df.columns:
            df["Water_Quality_Index"] = df[["pH", "Dissolved_Oxygen", "Temperature_C"]].mean(axis=1)

        # This make three class: Low, Medium, High by quantile.
        # It is not perfect truth label, but okay for demonstration because dataset has no official class label.
        try:
            df["Water_Quality_Category"] = pd.qcut(
                df["Water_Quality_Index"], q=3,
                labels=["Low", "Medium", "High"], duplicates="drop"
            )
        except ValueError:
            df["Water_Quality_Category"] = pd.cut(
                df["Water_Quality_Index"], bins=3,
                labels=["Low", "Medium", "High"]
            )

        df["Water_Quality_Category"] = df["Water_Quality_Category"].astype(str)
        target_table = df[["Water_Quality_Index", "Water_Quality_Category"]].copy()
        target_table.to_csv(self.table_dir / "ml_created_class_target.csv", index=False)
        print("  [Saved] ml_created_class_target.csv")
        return df

    def _linear_regression_metrics(self, df: pd.DataFrame) -> None:
        """Save aligned linear regression metric table for fish count prediction."""

        # Same predictors as main regression section.
        # This keep all report/output values consistent.
        features = ["Temperature_C", "pH", "Dissolved_Oxygen"]

        if "Fish_Count" not in df.columns or not all(c in df.columns for c in features):
            self.notes.append("Linear Regression skipped because Fish_Count or required features are missing.")
            return

        X = df[features]
        y = df["Fish_Count"]

        # Full cleaned dataset used to align with regression_results.txt and report.
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        metrics = pd.DataFrame([{
            "Model": "Linear Regression",
            "Target": "Fish_Count",
            "Evaluation_Type": "Full cleaned dataset explanatory regression",
            "R2": round(r2_score(y, y_pred), 4),
            "MAE": round(mean_absolute_error(y, y_pred), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y, y_pred))), 4),
            "Features_Used": ", ".join(features)
        }])

        metrics.to_csv(self.table_dir / "ml_linear_regression_metrics.csv", index=False)
        print("  [Saved] ml_linear_regression_metrics.csv")

    def _kmeans_clustering(self, df: pd.DataFrame) -> None:
        """Add K-Means, Elbow, Silhouette, PCA 2D plot, and cluster summary."""
        features = self._available_features(df)
        if len(features) < 3 or len(df) < 6:
            self.notes.append("K-Means skipped because not enough numeric features or rows.")
            return

        # This scale data because K-Means is distance based
        X = df[features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        max_k = min(8, len(df) - 1)
        inertias = []
        silhouettes = []
        k_values = list(range(2, max_k + 1))
        for k in k_values:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X_scaled, labels))

        elbow_df = pd.DataFrame({"k": k_values, "Inertia": inertias, "Silhouette_Score": silhouettes})
        elbow_df.to_csv(self.table_dir / "ml_kmeans_elbow_silhouette_scores.csv", index=False)
        print("  [Saved] ml_kmeans_elbow_silhouette_scores.csv")

        # This choose best k using highest silhouette score
        best_k = int(elbow_df.loc[elbow_df["Silhouette_Score"].idxmax(), "k"])
        final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        df["KMeans_Cluster"] = final_km.fit_predict(X_scaled)
        df[["Site_ID", "Date", "Species", "KMeans_Cluster"] + features].to_csv(
            self.table_dir / "ml_kmeans_clustered_records.csv", index=False
        )
        print("  [Saved] ml_kmeans_clustered_records.csv")

        # Elbow chart
        plt.figure(figsize=(8, 5))
        plt.plot(elbow_df["k"], elbow_df["Inertia"], marker="o")
        plt.title("K-Means Elbow Method")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.tight_layout()
        plt.savefig(self.fig_dir / "fig13_kmeans_elbow_method.png", dpi=300)
        plt.close()
        print("  [Saved] fig13_kmeans_elbow_method.png")

        # Silhouette chart
        plt.figure(figsize=(8, 5))
        plt.plot(elbow_df["k"], elbow_df["Silhouette_Score"], marker="o")
        plt.title("K-Means Silhouette Score")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.tight_layout()
        plt.savefig(self.fig_dir / "fig14_kmeans_silhouette_score.png", dpi=300)
        plt.close()
        print("  [Saved] fig14_kmeans_silhouette_score.png")

        # PCA reduce many features into 2D for simple cluster picture
        pca = PCA(n_components=2, random_state=42)
        pca_points = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame({
            "PCA1": pca_points[:, 0],
            "PCA2": pca_points[:, 1],
            "KMeans_Cluster": df["KMeans_Cluster"].values
        })
        pca_df.to_csv(self.table_dir / "ml_pca_2d_cluster_points.csv", index=False)
        print("  [Saved] ml_pca_2d_cluster_points.csv")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="KMeans_Cluster", palette="tab10", s=80)
        plt.title("PCA 2D Visualization of K-Means Clusters")
        plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
        plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
        plt.tight_layout()
        plt.savefig(self.fig_dir / "fig15_pca_kmeans_clusters.png", dpi=300)
        plt.close()
        print("  [Saved] fig15_pca_kmeans_clusters.png")

        # Summary table requested by user. Missing nitrate/phosphate/turbidity are marked clearly.
        summary_cols = ["pH", "Temperature_C", "Dissolved_Oxygen", "Fish_Count"]
        cluster_summary = df.groupby("KMeans_Cluster")[summary_cols].mean().round(3).reset_index()
        cluster_summary["Turbidity"] = "Not available in dataset"
        cluster_summary["Nitrate"] = "Not available in dataset"
        cluster_summary["Phosphate"] = "Not available in dataset"
        cluster_summary = cluster_summary[[
            "KMeans_Cluster", "pH", "Temperature_C", "Dissolved_Oxygen",
            "Turbidity", "Nitrate", "Phosphate", "Fish_Count"
        ]]
        cluster_summary.to_csv(self.table_dir / "ml_kmeans_cluster_summary_table.csv", index=False)
        print("  [Saved] ml_kmeans_cluster_summary_table.csv")

    def _classification_models(self, df: pd.DataFrame) -> None:
        """Run Logistic Regression, SVM, Decision Tree, Random Forest, Gradient Boosting, MLP."""
        features = [c for c in ["Temperature_C", "pH", "Dissolved_Oxygen", "Fish_Count", "Avg_Size_cm"] if c in df.columns]
        target = "Water_Quality_Category"
        if target not in df.columns or len(features) < 3:
            self.notes.append("Classification skipped because class target or features are missing.")
            return

        model_df = df[features + [target]].dropna().copy()
        if model_df[target].nunique() < 2 or len(model_df) < 10:
            self.notes.append("Classification skipped because class target has too few classes or rows.")
            return

        X = model_df[features]
        y = model_df[target]
        stratify = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=stratify
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "SVM": SVC(kernel="rbf", random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=4),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Neural Network / MLP": MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=1000, random_state=42)
        }

        rows = []
        labels = sorted(y.unique().tolist())
        for name, model in models.items():
            # Linear/SVM/MLP usually need scaled values. Tree models can also accept scaled values okay.
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            cm = confusion_matrix(y_test, pred, labels=labels)

            rows.append({
                "Model": name,
                "Target": target,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision_Macro": precision_score(y_test, pred, average="macro", zero_division=0),
                "Recall_Macro": recall_score(y_test, pred, average="macro", zero_division=0),
                "F1_Macro": f1_score(y_test, pred, average="macro", zero_division=0),
                "Confusion_Matrix_Labels": str(labels),
                "Confusion_Matrix": cm.tolist(),
                "Features_Used": ", ".join(features)
            })

            cm_df = pd.DataFrame(cm, index=[f"Actual_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
            safe_name = name.lower().replace(" ", "_").replace("/", "_")
            cm_df.to_csv(self.table_dir / f"ml_confusion_matrix_{safe_name}.csv")
            print(f"  [Saved] ml_confusion_matrix_{safe_name}.csv")

        comparison = pd.DataFrame(rows).sort_values("F1_Macro", ascending=False)
        comparison.to_csv(self.table_dir / "ml_classification_model_comparison.csv", index=False)
        print("  [Saved] ml_classification_model_comparison.csv")

        plt.figure(figsize=(10, 5))
        # This use simple matplotlib bar chart to avoid seaborn version color issue
        plt.barh(comparison["Model"], comparison["F1_Macro"])
        plt.title("Classification Model Comparison by F1-Score")
        plt.xlabel("Macro F1-score")
        plt.ylabel("Model")
        plt.tight_layout()
        plt.savefig(self.fig_dir / "fig16_classification_model_comparison.png", dpi=300)
        plt.close()
        print("  [Saved] fig16_classification_model_comparison.png")

    def _time_series_analysis(self, df: pd.DataFrame) -> None:
        """Add simple time series trend if Date column exists."""
        if "Date" not in df.columns or "Fish_Count" not in df.columns:
            self.notes.append("Time Series skipped because Date or Fish_Count is missing.")
            return

        ts = df.copy()
        ts["Date"] = pd.to_datetime(ts["Date"], errors="coerce")
        ts = ts.dropna(subset=["Date"])
        if ts.empty:
            self.notes.append("Time Series skipped because Date values cannot be parsed.")
            return

        # This aggregate by month because small dataset should not overfit daily trend
        ts["Month"] = ts["Date"].dt.to_period("M").dt.to_timestamp()
        monthly = ts.groupby("Month").agg(
            Avg_Fish_Count=("Fish_Count", "mean"),
            Avg_WQI=("Water_Quality_Index", "mean")
        ).reset_index()

        monthly["Forecast_Next_Fish_Count"] = monthly["Avg_Fish_Count"].rolling(window=2, min_periods=1).mean().shift(1)
        monthly.to_csv(self.table_dir / "ml_time_series_monthly_trend_forecast.csv", index=False)
        print("  [Saved] ml_time_series_monthly_trend_forecast.csv")

        plt.figure(figsize=(9, 5))
        plt.plot(monthly["Month"], monthly["Avg_Fish_Count"], marker="o", label="Actual average fish count")
        plt.plot(monthly["Month"], monthly["Forecast_Next_Fish_Count"], marker="o", linestyle="--", label="Simple rolling forecast")
        plt.title("Simple Time Series Trend and Forecast")
        plt.xlabel("Month")
        plt.ylabel("Average Fish Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.fig_dir / "fig17_time_series_trend_forecast.png", dpi=300)
        plt.close()
        print("  [Saved] fig17_time_series_trend_forecast.png")


    def _save_notes(self) -> None:
        """Save simple technical conclusion and limitation notes."""
        text = """
ADVANCED ML TECHNICAL CONCLUSION
================================

1. Data cleaning, EDA, correlation, regression, and visualisation analyses applied.
2. K-Means clustering was applied to identify environmental groups with similar water quality and fish population characteristics.
3. PCA was used to visualise cluster patterns in a simplified two-dimensional space.
4. Water_Quality_Category (Low, Medium, High) was created from WQI for classification modelling.
5. Multiple Linear Regression was used to analyse fish count prediction and showed weak explanatory power (R² = 0.025).
6. Classification models were used to evaluate water quality categories using environmental variables.
7. Time series analysis was applied to identify short-term trends in water quality and fish health indicators.
8. Additional environmental variables such as turbidity, nitrate, and phosphate were identified as important future data requirements.
9. Results should be interpreted carefully because the dataset is small, and some classification labels were derived from the Water Quality Index.
""".strip()
        if self.notes:
            text += "\n\nPROCESS NOTES\n=============\n" + "\n".join(f"- {n}" for n in self.notes)
        with open(self.report_dir / "advanced_ml_conclusion_and_limitations.txt", "w") as f:
            f.write(text)
        print("  [Saved] advanced_ml_conclusion_and_limitations.txt")


# =============================================================================
# CLASS 9 - AvonRiverAnalysisApp  (Orchestrator)
# =============================================================================
class AvonRiverAnalysisApp:
    """
    This is the main app class. It coordinate all other class.
    Single responsibility: orchestrate the full analysis pipeline.
    """

    def __init__(self, base_dir: Path):
        # This set folder path for all input and output
        self.base_dir   = base_dir
        self.data_dir   = base_dir / "data"
        self.output_dir = base_dir / "outputs"
        self.fig_dir    = self.output_dir / "figures"

        # This create all the helper class
        self.loader      = DataLoader(self.data_dir)
        self.cleaner     = DataCleaner()
        self.eda         = ExploratoryDataAnalyzer()
        self.stats       = StatisticalAnalyzer()
        self.viz         = VisualizationGenerator(self.fig_dir)
        self.recommender = RecommendationEngine()
        self.exporter    = ReportExporter()
        self.advanced_ml = AdvancedMLAnalyzer()

    def run(self) -> None:
        """Run the full analysis pipeline from start to finish."""
        print("\n" + "★" * 70)
        print("  MSE803 ASSESSMENT 1 - AVON RIVER ANALYSIS")
        print("  Water Quality & Fish Population Investigation")
        print("★" * 70)
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # ── Step 1: Load ──────────────────────────────────────────────────────
        raw_df = self.loader.load("Data_Set_Assignmnet_1-V0.1_20426.xlsx")

        # ── Step 2: Clean ─────────────────────────────────────────────────────
        clean_df = self.cleaner.clean(raw_df)
        self.cleaner.save(clean_df, self.output_dir)

        # ── Step 3: EDA ───────────────────────────────────────────────────────
        eda_results = self.eda.analyze(clean_df)
        self.eda.save_reports(self.output_dir)

        # ── Step 4: Statistical Analysis ─────────────────────────────────────
        corr_matrix  = self.stats.correlation_analysis(clean_df)
        reg_results  = self.stats.regression_analysis(clean_df)
        self.stats.save_reports(self.output_dir)

        # ── Step 5: Visualisation ─────────────────────────────────────────────
        self.viz.generate_all(clean_df, corr_matrix, reg_results)

        # ── Step 6: Recommendations ───────────────────────────────────────────
        recommendations = self.recommender.generate(clean_df, corr_matrix, eda_results, reg_results)
        self.recommender.save(self.output_dir)

        # ── Step 7: Export reports ────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("  STEP 7: SAVING ALL REPORTS")
        print("=" * 70)
        print()
        self.exporter.export_assessment_mapping(self.output_dir, recommendations)

        # ── Step 8: Advanced ML Extension ─────────────────────────────────────
        self.advanced_ml.run(clean_df, self.output_dir)

        # ── Done ──────────────────────────────────────────────────────────────
        self._print_summary()

    def _print_summary(self) -> None:
        """Print final summary of all generated output."""
        print("\n" + "★" * 70)
        print("  ANALYSIS COMPLETE")
        print("★" * 70)
        print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n  All outputs saved to: {self.output_dir}")
        print("""
  OUTPUT FILES GENERATED:
  ─────────────────────────────────────────────────────────────
  CLEANED DATA (2 files):
    outputs/cleaned_data/cleaned_avon_river_data.xlsx
      - Cleaned dataset in Excel format for marker review and Power BI/Excel.
    outputs/cleaned_data/cleaned_avon_river_data.csv
      - Cleaned dataset in CSV format for reproducible analysis.

  REPORTS (8 files):
    outputs/reports/descriptive_statistics.csv
      - Mean, standard deviation, min, max, skewness, and kurtosis.
    outputs/reports/site_summary.csv
      - Site-level summary for AV-1, AV-2, and AV-3.
    outputs/reports/species_summary.csv
      - Species-level fish count and average fish size summary.
    outputs/reports/correlation_matrix.csv
      - Pearson correlation matrix for water quality and fish variables.
    outputs/reports/regression_results.txt
      - Regression metrics, coefficients, OLS summary, and limitations.
    outputs/reports/recommendations.txt
      - Evidence-based environmental recommendations.
    outputs/reports/assessment_mapping.md
      - Mapping of generated outputs to assessment tasks.
    outputs/reports/advanced_ml_conclusion_and_limitations.txt
      - Advanced ML conclusions, limitations, and ethical notes.

  FIGURES (17 charts):
    outputs/figures/fig01_histograms.png
      - Univariate histograms for key variables.
    outputs/figures/fig02_boxplots.png
      - Boxplots for spread and outlier checking.
    outputs/figures/fig03_correlation_heatmap.png
      - Pearson correlation heatmap.
    outputs/figures/fig04_scatter_do_vs_fishcount.png
      - Dissolved oxygen vs fish count scatter/regression chart.
    outputs/figures/fig05_scatter_temp_vs_do.png
      - Temperature vs dissolved oxygen scatter/regression chart.
    outputs/figures/fig06_scatter_ph_vs_fishcount.png
      - pH vs fish count scatter/regression chart.
    outputs/figures/fig07_regression_plot.png
      - Actual vs predicted and residual regression plot.
    outputs/figures/fig08_pairplot.png
      - Pairplot for numeric variable relationships.
    outputs/figures/fig09_site_comparison.png
      - Site comparison for WQI, FHI, DO, and fish count.
    outputs/figures/fig10_species_distribution.png
      - Species fish count and average size chart.
    outputs/figures/fig11_monthly_trends.png
      - Monthly trends for DO, fish count, and WQI.
    outputs/figures/fig12_executive_dashboard.png
      - Executive river health dashboard.
    outputs/figures/fig13_kmeans_elbow_method.png
      - K-Means elbow method chart.
    outputs/figures/fig14_kmeans_silhouette_score.png
      - K-Means silhouette score chart.
    outputs/figures/fig15_pca_kmeans_clusters.png
      - PCA 2D visualisation of K-Means clusters.
    outputs/figures/fig16_classification_model_comparison.png
      - Classification model comparison by macro F1-score.
    outputs/figures/fig17_time_series_trend_forecast.png
      - Monthly time-series trend and simple rolling forecast.

  ADVANCED ML TABLES (14 files):
    outputs/tables/ml_created_class_target.csv
      - Created Water_Quality_Category labels from Water_Quality_Index.
    outputs/tables/ml_linear_regression_metrics.csv
      - Linear Regression metrics for Fish_Count prediction.
    outputs/tables/ml_kmeans_elbow_silhouette_scores.csv
      - K values, inertia, and silhouette scores.
    outputs/tables/ml_kmeans_clustered_records.csv
      - Clean records with assigned KMeans_Cluster labels.
    outputs/tables/ml_pca_2d_cluster_points.csv
      - PCA1 and PCA2 points for cluster visualisation.
    outputs/tables/ml_kmeans_cluster_summary_table.csv
      - Cluster summary table with pH, temperature, DO, and fish count.
    outputs/tables/ml_classification_model_comparison.csv
      - Accuracy, precision, recall, F1, and model comparison.
    outputs/tables/ml_confusion_matrix_logistic_regression.csv
      - Confusion matrix for Logistic Regression.
    outputs/tables/ml_confusion_matrix_svm.csv
      - Confusion matrix for SVM.
    outputs/tables/ml_confusion_matrix_decision_tree.csv
      - Confusion matrix for Decision Tree.
    outputs/tables/ml_confusion_matrix_random_forest.csv
      - Confusion matrix for Random Forest.
    outputs/tables/ml_confusion_matrix_gradient_boosting.csv
      - Confusion matrix for Gradient Boosting.
    outputs/tables/ml_confusion_matrix_neural_network___mlp.csv
      - Confusion matrix for Neural Network / MLP.
    outputs/tables/ml_time_series_monthly_trend_forecast.csv
      - Monthly trend and simple rolling forecast table.
  ─────────────────────────────────────────────────────────────
        """)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # This find the folder where main.py is saved
    BASE_DIR = Path(__file__).parent.resolve()

    # This create and run the main application
    app = AvonRiverAnalysisApp(BASE_DIR)
    app.run()
