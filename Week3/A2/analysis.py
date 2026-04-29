# -------------------------------------------------------
# Week 3 - Activity 2: Data Cleaning and Data Visualization
#                        with Correlation Heatmap (Pearson)
# Author : Benjelyn Reves Patiag
# Date   : 28 Apr 2026
# Description: cleans messy CSV data,
#              computes Pearson correlation between Age & Salary,
#              detects outliers (IQR + Z-score), and produces
#              5 publication-quality charts.
# -------------------------------------------------------

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════
#  CLASS 1: DataLoader
#  Responsibility: read the raw CSV from disk and return a DataFrame
# ═══════════════════════════════════════════════════════════════════
class DataLoader:
    """Loads raw CSV data and prints a quick summary."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.raw_df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """Read CSV and store as raw_df.  Returns the DataFrame."""
        self.raw_df = pd.read_csv(self.filepath)
        print("=" * 55)
        print("  RAW DATA")
        print("=" * 55)
        print(self.raw_df.to_string())
        print(f"\n  Shape : {self.raw_df.shape}")
        print(f"  Nulls :\n{self.raw_df.isnull().sum().to_string()}")
        print("=" * 55)
        return self.raw_df


# ═══════════════════════════════════════════════════════════════════
#  CLASS 2: DataCleaner
#  Responsibility: fix every quality problem in the raw DataFrame
# ═══════════════════════════════════════════════════════════════════
class DataCleaner:
    """
    Fixes: text-encoded numbers, duplicates, missing values,
           invalid dates, missing IDs.
    """

    # words-to-number lookup tables
    AGE_MAP    = {"thirty-eight": 38}
    SALARY_MAP = {"sixty five thousand": 65000}

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    #──── private helpers ─────────────────────────────────────────────

    def _fix_text_numbers(self) -> None:
        """Convert word-numbers ('thirty-eight') to real integers."""
        self.df['Age']    = self.df['Age'].replace(self.AGE_MAP)
        self.df['Age']    = pd.to_numeric(self.df['Age'],    errors='coerce')
        self.df['Salary'] = self.df['Salary'].replace(self.SALARY_MAP)
        self.df['Salary'] = pd.to_numeric(self.df['Salary'], errors='coerce')
        self.df['ID']     = pd.to_numeric(self.df['ID'],     errors='coerce')

    def _remove_duplicates(self) -> None:
        """Drop duplicate rows, keeping the first occurrence per Name."""
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=['Name'], keep='first')
        print(f"  [Cleaner] Removed {before - len(self.df)} duplicate row(s).")

    @staticmethod
    def _parse_date(value) -> pd.Timestamp:
        """Try multiple date formats; return NaT if none match."""
        if pd.isnull(value):
            return pd.NaT
        for fmt in ('%d/%m/%Y', '%Y-%m-%d', '%Y-%d-%m'):
            try:
                return pd.to_datetime(value, format=fmt)
            except ValueError:
                pass
        return pd.NaT

    def _fix_dates(self) -> None:
        """Parse the Join Date column, silently fixing invalid entries."""
        self.df['Join Date'] = self.df['Join Date'].apply(self._parse_date)

    def _impute_missing(self) -> None:
        """Fill numeric gaps with median; fill categorical gaps with mode."""
        self.df['Age']     = self.df['Age'].fillna(self.df['Age'].median())
        self.df['Salary']  = self.df['Salary'].fillna(self.df['Salary'].median())
        self.df['Country'] = self.df['Country'].fillna(self.df['Country'].mode()[0])
        self.df['Name']    = self.df['Name'].fillna('Unknown')
        max_id = self.df['ID'].max()
        self.df['ID'] = (self.df['ID']
                         .fillna(max_id + 1)
                         .astype(int))

    #──── public method ────────────────────────────────────────────────

    def clean(self) -> pd.DataFrame:
        """Run all cleaning steps in order and return the clean DataFrame."""
        print("\n  Running DataCleaner …")
        self._fix_text_numbers()
        self._remove_duplicates()
        self._fix_dates()
        self._impute_missing()
        print("\n" + "=" * 55)
        print("  CLEANED DATA")
        print("=" * 55)
        print(self.df.to_string())
        print("=" * 55)
        return self.df


# ═══════════════════════════════════════════════════════════════════
#  CLASS 3: CorrelationAnalyzer
#  Responsibility: compute Pearson r, p-value, regression line
# ═══════════════════════════════════════════════════════════════════
class CorrelationAnalyzer:
    """
    Computes Pearson correlation between two numeric columns and
    stores regression parameters for downstream use.
    """

    def __init__(self, df: pd.DataFrame,
                 col_x: str = 'Age', col_y: str = 'Salary'):
        self.df    = df
        self.col_x = col_x
        self.col_y = col_y
        # results (populated by .analyze())
        self.pearson_r: float = 0.0
        self.p_value:   float = 0.0
        self.slope:     float = 0.0
        self.intercept: float = 0.0
        self.corr_matrix: pd.DataFrame | None = None

    def analyze(self) -> dict:
        """
        Run Pearson correlation + linear regression.
        Returns a dict with r, p, slope, intercept.
        """
        numeric = self.df[[self.col_x, self.col_y]].dropna()
        self.pearson_r, self.p_value = stats.pearsonr(
            numeric[self.col_x], numeric[self.col_y]
        )
        self.slope, self.intercept, *_ = stats.linregress(
            numeric[self.col_x], numeric[self.col_y]
        )
        self.corr_matrix = self.df[[self.col_x, self.col_y]].corr(method='pearson')

        print("\n" + "=" * 55)
        print("  CORRELATION ANALYSIS  (Pearson)")
        print("=" * 55)
        print(f"  Pearson r: {self.pearson_r:.4f}")
        print(f"  p-value  : {self.p_value:.4f}")
        print(f"  Slope    : {self.slope:.2f}")
        print(f"  Intercept: {self.intercept:.2f}")
        print(f"\n  Correlation matrix:\n{self.corr_matrix.to_string()}")
        print("=" * 55)

        return {
            'r': self.pearson_r,
            'p': self.p_value,
            'slope': self.slope,
            'intercept': self.intercept,
        }


# ═══════════════════════════════════════════════════════════════════
#  CLASS 4: OutlierDetector
#  Responsibility: identify outliers via IQR and Z-score
# ═══════════════════════════════════════════════════════════════════
class OutlierDetector:
    """Detects outliers using IQR fences and Z-score threshold."""

    def __init__(self, df: pd.DataFrame,
                 columns: list[str] | None = None,
                 z_threshold: float = 2.0):
        self.df          = df
        self.columns     = columns or ['Age', 'Salary']
        self.z_threshold = z_threshold
        self.iqr_results: dict = {}
        self.z_df: pd.DataFrame | None = None

    def detect_iqr(self) -> dict:
        """Compute IQR fences and flag outliers for each column."""
        print("\n" + "=" * 55)
        print("  OUTLIER DETECTION  (IQR)")
        print("=" * 55)
        for col in self.columns:
            series = self.df[col].dropna()
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = series[(series < lower) | (series > upper)]
            self.iqr_results[col] = {
                'Q1': q1, 'Q3': q3, 'IQR': iqr,
                'lower': lower, 'upper': upper,
                'outliers': outliers,
            }
            print(f"  {col}: Q1={q1}, Q3={q3}, IQR={iqr}")
            print(f"         Fences [{lower:.1f}, {upper:.1f}]")
            print(f"         Outliers → {list(outliers.values)}")
        print("=" * 55)
        return self.iqr_results

    def detect_zscore(self) -> pd.DataFrame:
        """Flag rows where any column's |z-score| exceeds threshold."""
        numeric = self.df[self.columns].dropna().copy()
        z_abs   = np.abs(stats.zscore(numeric))
        numeric['z_max']   = z_abs.max(axis=1)
        numeric['outlier'] = numeric['z_max'] > self.z_threshold
        self.z_df = numeric

        n_out = numeric['outlier'].sum()
        print(f"\n  Z-score outliers (|z|>{self.z_threshold}): {n_out} found.")
        return self.z_df


# ═══════════════════════════════════════════════════════════════════
#  CLASS 5: ChartBuilder
#  Responsibility: produce and save all 5 visualisations
# ═══════════════════════════════════════════════════════════════════
class ChartBuilder:
    """Generates all charts and saves them to an output directory."""

    PALETTE = ["#028090", "#F96167", "#F9E795", "#2F3C7E", "#02C39A"]

    def __init__(self, raw_df: pd.DataFrame, clean_df: pd.DataFrame,
                 analyzer: CorrelationAnalyzer,
                 detector: OutlierDetector,
                 output_dir: str = "charts"):
        self.raw     = raw_df
        self.clean   = clean_df
        self.ana     = analyzer
        self.det     = detector
        self.out_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'axes.spines.top':   False,
            'axes.spines.right': False,
        })

    def _save(self, name: str) -> str:
        path = os.path.join(self.out_dir, name)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [Chart] Saved → {path}")
        return path

    #──── Chart 1 ─────────────────────────────────────────────────────
    def chart1_missing_heatmap(self) -> str:
        """Side-by-side heatmap showing missing values before and after cleaning."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, data, title in zip(
            axes,
            [self.raw, self.clean],
            ['Raw (Before Cleaning)', 'Cleaned (After)']
        ):
            sns.heatmap(
                data.isnull().astype(int), ax=ax,
                cbar=False, cmap=['#ECEFF4', '#F96167'],
                linewidths=0.5, linecolor='white',
                yticklabels=False, xticklabels=data.columns,
            )
            ax.set_title(title, fontweight='bold', fontsize=12, color='#1E2761')
            ax.tick_params(axis='x', labelsize=9, rotation=30)
        fig.suptitle('Missing Value Map: Before vs After Cleaning',
                     fontsize=14, fontweight='bold', color='#1E2761', y=1.02)
        plt.tight_layout()
        return self._save('chart1_missing.png')

    #──── Chart 2 ─────────────────────────────────────────────────────
    def chart2_scatter_regression(self) -> str:
        """Scatter plot of Age vs Salary with Pearson regression line."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(self.clean['Age'], self.clean['Salary'],
                   color=self.PALETTE[0], s=100, alpha=0.85,
                   edgecolors='white', linewidth=1.5, zorder=3)
        x_line = np.linspace(self.clean['Age'].min(), self.clean['Age'].max(), 100)
        ax.plot(x_line,
                self.ana.slope * x_line + self.ana.intercept,
                color=self.PALETTE[1], linewidth=2.5,
                label=f'Trend  (r = {self.ana.pearson_r:.2f})')
        for _, row in self.clean.iterrows():
            ax.annotate(row['Name'], (row['Age'], row['Salary']),
                        textcoords='offset points', xytext=(5, 5),
                        fontsize=8, color='#36454F')
        ax.set_xlabel('Age (years)', fontsize=12, color='#36454F')
        ax.set_ylabel('Salary (NZD)', fontsize=12, color='#36454F')
        ax.set_title('Age vs Salary — Scatter & Regression',
                     fontsize=14, fontweight='bold', color='#1E2761')
        ax.legend(fontsize=11)
        ax.set_facecolor('#F8FAFC')
        ax.grid(axis='y', color='#E2E8F0', linewidth=0.8)
        plt.tight_layout()
        return self._save('chart2_scatter.png')

    #──── Chart 3 ─────────────────────────────────────────────────────
    def chart3_pearson_heatmap(self) -> str:
        """Pearson correlation heatmap for Age & Salary."""
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            self.ana.corr_matrix, ax=ax,
            annot=True, fmt='.3f', cmap='coolwarm',
            vmin=-1, vmax=1, linewidths=2, linecolor='white',
            annot_kws={'size': 14, 'weight': 'bold'}, square=True,
        )
        ax.set_title('Pearson Correlation Heatmap\n(Age & Salary)',
                     fontsize=12, fontweight='bold', color='#1E2761')
        ax.tick_params(labelsize=11)
        plt.tight_layout()
        return self._save('chart3_heatmap.png')

    #──── Chart 4 ─────────────────────────────────────────────────────
    def chart4_boxplots(self) -> str:
        """Box-and-whisker plots for outlier detection (IQR)."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, col, color in zip(axes, ['Age', 'Salary'],
                                  [self.PALETTE[0], self.PALETTE[3]]):
            ax.boxplot(
                self.clean[col].dropna(), patch_artist=True,
                medianprops=dict(color='white', linewidth=2.5),
                boxprops=dict(facecolor=color, alpha=0.8),
                whiskerprops=dict(color=color, linewidth=1.5),
                capprops=dict(color=color, linewidth=2),
                flierprops=dict(marker='o', color=self.PALETTE[1],
                                markersize=10, markerfacecolor=self.PALETTE[1]),
            )
            ax.set_title(f'{col} Distribution & Outliers',
                         fontsize=12, fontweight='bold', color='#1E2761')
            ax.set_ylabel(col, fontsize=11, color='#36454F')
            ax.set_facecolor('#F8FAFC')
            ax.grid(axis='y', color='#E2E8F0', linewidth=0.8)
            ax.set_xticks([])
        fig.suptitle('Outlier Detection via IQR (Box-and-Whisker)',
                     fontsize=13, fontweight='bold', color='#1E2761')
        plt.tight_layout()
        return self._save('chart4_boxplots.png')

    #──── Chart 5 ─────────────────────────────────────────────────────
    def chart5_country_zscore(self) -> str:
        """Average salary per country (bar) + Z-score outlier scatter."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Bar chart
        avg_sal = (self.clean.groupby('Country')['Salary']
                   .mean().sort_values(ascending=False))
        bars = axes[0].bar(avg_sal.index, avg_sal.values,
                           color=self.PALETTE[:3],
                           edgecolor='white', linewidth=1.2)
        for bar in bars:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 400,
                f'{bar.get_height():,.0f}',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#36454F',
            )
        axes[0].set_title('Average Salary by Country',
                          fontsize=12, fontweight='bold', color='#1E2761')
        axes[0].set_ylabel('Avg Salary (NZD)', fontsize=11)
        axes[0].set_facecolor('#F8FAFC')
        axes[0].grid(axis='y', color='#E2E8F0', linewidth=0.8)

        # Z-score scatter
        z_df = self.det.z_df
        normal  = z_df[~z_df['outlier']]
        outlier = z_df[ z_df['outlier']]
        axes[1].scatter(normal['Age'],  normal['Salary'],
                        color=self.PALETTE[0], s=90, label='Normal', alpha=0.85)
        axes[1].scatter(outlier['Age'], outlier['Salary'],
                        color=self.PALETTE[1], s=130, marker='*',
                        label='Outlier (|z|>2)', zorder=5)
        axes[1].set_title('Z-Score Outlier Detection\n(Age vs Salary)',
                          fontsize=12, fontweight='bold', color='#1E2761')
        axes[1].set_xlabel('Age', fontsize=11)
        axes[1].set_ylabel('Salary (NZD)', fontsize=11)
        axes[1].legend(fontsize=10)
        axes[1].set_facecolor('#F8FAFC')
        axes[1].grid(color='#E2E8F0', linewidth=0.8)

        plt.tight_layout()
        return self._save('chart5_country_zscore.png')

    def build_all(self) -> list[str]:
        """Run all chart methods and return a list of saved file paths."""
        print("\n  Building charts …")
        return [
            self.chart1_missing_heatmap(),
            self.chart2_scatter_regression(),
            self.chart3_pearson_heatmap(),
            self.chart4_boxplots(),
            self.chart5_country_zscore(),
        ]


# ═══════════════════════════════════════════════════════════════════
#  CLASS 6: AnalysisPipeline
#  Responsibility: orchestrate the full end-to-end workflow
# ═══════════════════════════════════════════════════════════════════
class AnalysisPipeline:
    """
    Top-level orchestrator.
    Call  .run()  to execute the full pipeline end-to-end.
    """

    def __init__(self, csv_path: str, charts_dir: str = "charts"):
        self.csv_path  = csv_path
        self.charts_dir = charts_dir

    def run(self) -> None:
        """Execute all pipeline stages in order."""

        # Stage 1: Load
        loader  = DataLoader(self.csv_path)
        raw_df  = loader.load()

        # Stage 2: Clean
        cleaner  = DataCleaner(raw_df)
        clean_df = cleaner.clean()

        # Stage 3: Correlation
        analyzer = CorrelationAnalyzer(clean_df)
        analyzer.analyze()

        # Stage 4: Outlier detection
        detector = OutlierDetector(clean_df)
        detector.detect_iqr()
        detector.detect_zscore()

        # Stage 5: Charts
        builder = ChartBuilder(raw_df, clean_df, analyzer, detector,
                               output_dir=self.charts_dir)
        paths = builder.build_all()

        # Final summary
        print("\n" + "=" * 55)
        print("  PIPELINE COMPLETE")
        print("=" * 55)
        print(f"  Records (raw)    : {len(raw_df)}")
        print(f"  Records (cleaned): {len(clean_df)}")
        print(f"  Pearson r        : {analyzer.pearson_r:.4f}")
        print(f"  p-value          : {analyzer.p_value:.4f}")
        print(f"  IQR outliers Age : {list(detector.iqr_results['Age']['outliers'].values)}")
        print(f"  IQR outliers Sal : {list(detector.iqr_results['Salary']['outliers'].values)}")
        print(f"  Charts saved     : {len(paths)}")
        for p in paths:
            print(f"    • {p}")
        print("=" * 55)


#──── Entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = AnalysisPipeline(
        csv_path   = "messy_dataset_Mukesh.csv",
        charts_dir = "charts",
    )
    pipeline.run()
