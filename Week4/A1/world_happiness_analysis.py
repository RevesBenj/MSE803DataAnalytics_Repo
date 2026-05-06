
# -------------------------------------------------------
# Week 4 - Activity 1: World Happiness Dataset: Data Cleaning & Visualization
# Author : Benjelyn Reves Patiag
# Date   : 6 May 2026
# Description: Perform data visualization using the dataset provided in Week 4 on Blackboard. Before creating  visualizations, ensure that the conduct proper data cleaning, including handling missing values and detecting/removing outliers.
# -------------------------------------------------------

"""
========================================================
Outputs:
  chart1_happiness_by_country.png : Happiness Score ranking
  chart2_gdp_vs_happiness.png     : GDP vs Happiness scatter
  chart3_feature_comparison.png   : Top 5 vs Bottom 5 feature comparison
  chart4_correlation_heatmap.png  : Pearson correlation heatmap
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats


# ══════════════════════════════════════════════════════════════════
# CLASS 1 : load and clean the data
# this class is responsible for read csv and make it clean
# ══════════════════════════════════════════════════════════════════

class DataCleaner:
    """
    This class for load data from csv file
    and do the cleaning process before analysis.
    """

    def __init__(self, filepath: str):
        # save the path where csv file is living
        self.filepath = filepath

        # this will store dataframe after loading
        self.df = None

        # this store only numeric column name, not include Country
        self.numeric_cols = []

        # here we save outlier result for each column
        self.outlier_summary = {}

    def load(self) -> "DataCleaner":
        """
        read csv file and put into dataframe.
        if file not found, it will make error and stop.
        """
        # try open the file and read it
        self.df = pd.read_csv(self.filepath)
        print(f"Loaded: {self.df.shape[0]} rows x {self.df.shape[1]} columns")

        # find column that is number type, skip Country column
        self.numeric_cols = [c for c in self.df.columns if c != "Country"]

        # return self so we can chain method like .load().clean()
        return self

    def remove_duplicate(self) -> "DataCleaner":
        """
        check and remove row that is exactly same with other row.
        duplicate row is bad for analysis, must delete it.
        """
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)

        # tell how many row was deleted
        removed = before - len(self.df)
        print(f"Duplicate removed: {removed}")
        return self

    def strip_whitespace(self) -> "DataCleaner":
        """
        for all text column, remove space in front and back.
        this important so 'Canada ' and 'Canada' is same thing.
        """
        for col in self.df.select_dtypes(include="object").columns:
            # apply strip to every cell in this column
            self.df[col] = self.df[col].str.strip()
        return self

    def check_missing(self) -> "DataCleaner":
        """
        count how many empty cell in each column.
        empty value is problem, we must know if exist.
        """
        missing = self.df.isnull().sum()
        print("\nMissing value each column:")

        # if any missing found, print it, if not print nice message
        if missing.any():
            print(missing[missing > 0])
        else:
            print("  None found. All cell is complete.")
        return self

    def detect_outlier(self) -> "DataCleaner":
        """
        use IQR method (1.5x rule) to find value that is too far
        from normal range. we only flag it, not delete the row.

        IQR rule:
          lower bound = Q1 - (1.5 * IQR)
          upper bound = Q3 + (1.5 * IQR)
          value outside this range = outlier
        """
        print("\nOutlier detection (IQR 1.5x rule):")

        for col in self.numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1

            # calculate the boundary for normal value
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # find row where value is outside boundary
            outlier_rows = self.df[
                (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            ]

            # save result to dictionary for later use
            self.outlier_summary[col] = outlier_rows["Country"].tolist()

            if outlier_rows.empty:
                print(f"  {col}: no outlier found")
            else:
                print(f"  {col}: outlier found -> {outlier_rows['Country'].tolist()}")

        return self

    def clean(self) -> "DataCleaner":
        """
        run all cleaning step one by one in correct order.
        this is shortcut method so user no need call each step manual.
        """
        # step 1: remove same row
        self.remove_duplicate()

        # step 2: fix whitespace in text
        self.strip_whitespace()

        # step 3: find empty cell
        self.check_missing()

        # step 4: find outlier using IQR
        self.detect_outlier()

        print(f"\nFinal clean data: {self.df.shape[0]} rows x {self.df.shape[1]} columns")
        return self

    def get_dataframe(self) -> pd.DataFrame:
        """
        give back the clean dataframe to other class.
        """
        # make copy so original dataframe is safe
        return self.df.copy()


# ══════════════════════════════════════════════════════════════════
# CLASS 2 : compute the correlation
# this class calculate pearson correlation between features
# ══════════════════════════════════════════════════════════════════

class CorrelationAnalyzer:
    """
    This class take clean dataframe and compute Pearson correlation.
    Pearson method measure linear relationship between two number column.
    Value is between -1 (opposite) and +1 (same direction).
    """

    def __init__(self, df: pd.DataFrame, numeric_cols: list):
        # store the dataframe we will analyze
        self.df = df

        # store list of numeric column name
        self.numeric_cols = numeric_cols

        # this will hold full correlation matrix
        self.corr_matrix = None

    def compute(self) -> "CorrelationAnalyzer":
        """
        calculate full pearson correlation matrix for all numeric column.
        result is square table, each cell show correlation between two column.
        """
        # pearson is default method in pandas corr()
        self.corr_matrix = self.df[self.numeric_cols].corr(method="pearson")

        # print correlation with happiness score sorted from high to low
        print("\nPearson correlation with Happiness_Score:")
        result = (
            self.corr_matrix["Happiness_Score"]
            .drop("Happiness_Score")    # remove self correlation (always 1.0)
            .sort_values(ascending=False)
        )
        print(result.to_string())
        return self

    def get_matrix(self) -> pd.DataFrame:
        """
        return the full correlation matrix dataframe.
        """
        return self.corr_matrix


# ══════════════════════════════════════════════════════════════════
# CLASS 3 : make all the chart
# this class draw and save 4 different visualization
# ══════════════════════════════════════════════════════════════════

class Visualizer:
    """
    This class create all 4 chart from clean data.
    All chart use white background and save as PNG file.
    """

    # color constant that we reuse in all chart
    TITLE_COLOR = "#1a1a2e"
    GRID_COLOR  = "#dddddd"
    COLOR_HIGH  = "#2A9D8F"   # green color for happy country
    COLOR_MID   = "#F4A261"   # orange color for medium country
    COLOR_LOW   = "#E63946"   # red color for sad country

    def __init__(self, df: pd.DataFrame, corr_matrix: pd.DataFrame):
        # store clean dataframe for plotting
        self.df = df

        # store correlation matrix for heatmap
        self.corr_matrix = corr_matrix

    def _make_bar_color(self, score: float) -> str:
        """
        private helper method decide color based on happiness score value.
        green = high, orange = medium, red = low.
        """
        if score >= 6.0:
            return self.COLOR_HIGH
        elif score >= 4.5:
            return self.COLOR_MID
        else:
            return self.COLOR_LOW

    def chart1_happiness_ranking(self) -> "Visualizer":
        """
        draw horizontal bar chart that rank all country by happiness score.
        bar color show which category each country belong.
        """
        # sort from lowest to highest so top country is at top of chart
        df_sorted = self.df.sort_values("Happiness_Score", ascending=True)

        # pick color for each bar based on score value
        colors = [self._make_bar_color(s) for s in df_sorted["Happiness_Score"]]

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # draw the horizontal bar
        bars = ax.barh(
            df_sorted["Country"], df_sorted["Happiness_Score"],
            color=colors, height=0.65, edgecolor="none"
        )

        # add number label at end of each bar
        for bar, val in zip(bars, df_sorted["Happiness_Score"]):
            ax.text(
                bar.get_width() + 0.05,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left",
                fontsize=9, color="#333333"
            )

        # set axis label and title
        ax.set_xlabel("Happiness Score", fontsize=12, color="#333333")
        ax.set_title(
            "Happiness Score by Country", fontsize=15,
            fontweight="bold", color=self.TITLE_COLOR, pad=14
        )
        ax.set_xlim(0, 8.5)

        # remove border that not needed
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#cccccc")
        ax.tick_params(axis="y", labelsize=9.5, colors="#333333")
        ax.tick_params(axis="x", colors="#888888")
        ax.xaxis.grid(True, linestyle="--", alpha=0.5, color=self.GRID_COLOR)
        ax.set_axisbelow(True)

        # make legend to explain color meaning
        legend_elements = [
            mpatches.Patch(color=self.COLOR_HIGH, label="High (>= 6.0)"),
            mpatches.Patch(color=self.COLOR_MID,  label="Medium (4.5 - 6.0)"),
            mpatches.Patch(color=self.COLOR_LOW,  label="Low (< 4.5)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.8)

        plt.tight_layout()
        plt.savefig("chart1_happiness_by_country.png", dpi=150,
                    bbox_inches="tight", facecolor="white")
        plt.close()
        print("Saved: chart1_happiness_by_country.png")
        return self

    def chart2_gdp_scatter(self) -> "Visualizer":
        """
        draw scatter plot for GDP per capita versus happiness score.
        also add regression line and color point by life expectancy value.
        regression line use linear equation to show trend direction.
        """
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # draw scatter point, color show life expectancy value
        scatter = ax.scatter(
            self.df["GDP_per_Capita"], self.df["Happiness_Score"],
            c=self.df["Healthy_Life_Expectancy"], cmap="YlOrRd",
            s=100, edgecolors="#555555", linewidths=0.5, zorder=5
        )

        # add color bar on side to show what color mean
        cb = plt.colorbar(scatter, ax=ax)
        cb.set_label("Healthy Life Expectancy", fontsize=10, color="#333333")

        # calculate regression line using scipy pearson method
        m, b, r, _p, _se = stats.linregress(
            self.df["GDP_per_Capita"], self.df["Happiness_Score"]
        )

        # make x value for drawing regression line
        x_line = np.linspace(
            self.df["GDP_per_Capita"].min(),
            self.df["GDP_per_Capita"].max(),
            100
        )

        # draw the regression line as dashed
        ax.plot(
            x_line, m * x_line + b, "--", color="#264653",
            linewidth=1.8, label=f"Regression  r={r:.2f}"
        )

        # add country name label next to each point
        for _, row in self.df.iterrows():
            ax.annotate(
                row["Country"],
                (row["GDP_per_Capita"], row["Happiness_Score"]),
                fontsize=7.5, textcoords="offset points",
                xytext=(5, 2), color="#444444"
            )

        # label and style the chart
        ax.set_xlabel("GDP per Capita", fontsize=12, color="#333333")
        ax.set_ylabel("Happiness Score", fontsize=12, color="#333333")
        ax.set_title(
            "GDP per Capita vs Happiness Score", fontsize=15,
            fontweight="bold", color=self.TITLE_COLOR, pad=14
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cccccc")
        ax.tick_params(colors="#888888")
        ax.xaxis.grid(True, linestyle="--", alpha=0.4, color=self.GRID_COLOR)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, color=self.GRID_COLOR)
        ax.set_axisbelow(True)
        ax.legend(fontsize=10, framealpha=0.8)

        plt.tight_layout()
        plt.savefig("chart2_gdp_vs_happiness.png", dpi=150,
                    bbox_inches="tight", facecolor="white")
        plt.close()
        print("Saved: chart2_gdp_vs_happiness.png")
        return self

    def chart3_feature_comparison(self) -> "Visualizer":
        """
        draw grouped bar chart compare feature average between
        top 5 happiest country and bottom 5 happiest country.
        good for see which feature make big difference between group.
        """
        # get top 5 and bottom 5 country by happiness score
        top5 = self.df.nlargest(5, "Happiness_Score")
        bot5 = self.df.nsmallest(5, "Happiness_Score")

        # list of feature column we want to compare
        features = [
            "GDP_per_Capita", "Social_Support", "Healthy_Life_Expectancy",
            "Freedom_to_Make_Choices", "Generosity", "Perceptions_of_Corruption"
        ]

        # short label for x axis so it not too long
        feat_labels = [
            "GDP/Capita", "Social\nSupport", "Life\nExpectancy",
            "Freedom", "Generosity", "Low\nCorruption"
        ]

        # calculate mean value of each feature for both group
        top_means = top5[features].mean().values
        bot_means = bot5[features].mean().values

        # x position for each group of bar
        x = np.arange(len(features))
        w = 0.35   # width of single bar

        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # draw two bar group side by side
        b1 = ax.bar(x - w / 2, top_means, w,
                    label="Top 5 Happiest", color=self.COLOR_HIGH, edgecolor="none")
        b2 = ax.bar(x + w / 2, bot_means, w,
                    label="Bottom 5 Happiest", color=self.COLOR_LOW, edgecolor="none")

        # add value label on top of each bar for easy reading
        for bar in b1:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=8,
                color=self.COLOR_HIGH, fontweight="bold"
            )
        for bar in b2:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=8,
                color=self.COLOR_LOW, fontweight="bold"
            )

        # set x tick to use short label
        ax.set_xticks(x)
        ax.set_xticklabels(feat_labels, fontsize=10.5, color="#333333")
        ax.set_ylabel("Average Score", fontsize=12, color="#333333")
        ax.set_title(
            "Feature Comparison: Top 5 vs Bottom 5 Happiest Countries",
            fontsize=14, fontweight="bold", color=self.TITLE_COLOR, pad=14
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cccccc")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, color=self.GRID_COLOR)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", colors="#888888")
        ax.legend(fontsize=11, framealpha=0.9)

        plt.tight_layout()
        plt.savefig("chart3_feature_comparison.png", dpi=150,
                    bbox_inches="tight", facecolor="white")
        plt.close()
        print("Saved: chart3_feature_comparison.png")
        return self

    def chart4_correlation_heatmap(self) -> "Visualizer":
        """
        draw heatmap to show pearson correlation between all numeric column.
        green color mean positive correlation, red mean negative correlation.
        number inside cell is the correlation value, close to 1 or -1 is strong.
        """
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # draw heatmap using seaborn, put number annotation in each cell
        sns.heatmap(
            self.corr_matrix,
            annot=True,           # show number inside cell
            fmt=".2f",            # round to 2 decimal
            cmap="RdYlGn",        # red=bad, yellow=neutral, green=good
            linewidths=0.6,
            linecolor="white",
            square=True,
            ax=ax,
            vmin=-1, vmax=1,      # fix color scale from -1 to 1
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 9}
        )

        ax.set_title(
            "Pearson Correlation Heatmap", fontsize=15,
            fontweight="bold", color=self.TITLE_COLOR, pad=14
        )

        # rotate x label so it not overlap each other
        ax.tick_params(axis="x", rotation=35, labelsize=9, colors="#333333")
        ax.tick_params(axis="y", rotation=0,  labelsize=9, colors="#333333")

        plt.tight_layout()
        plt.savefig("chart4_correlation_heatmap.png", dpi=150,
                    bbox_inches="tight", facecolor="white")
        plt.close()
        print("Saved: chart4_correlation_heatmap.png")
        return self

    def run_all(self) -> None:
        """
        run all 4 chart in correct order.
        shortcut so user only need call one method to get everything.
        """
        # call each chart method one by one using method chain
        (
            self
            .chart1_happiness_ranking()
            .chart2_gdp_scatter()
            .chart3_feature_comparison()
            .chart4_correlation_heatmap()
        )
        print("\nAll chart generated successfully.")


# ══════════════════════════════════════════════════════════════════
# CLASS 4 : the main runner / pipeline
# this class connect everything together and run the full pipeline
# ══════════════════════════════════════════════════════════════════

class HappinessPipeline:
    """
    This is main class that control whole analysis pipeline.
    It create the other class and run them in correct order.

    Usage:
        pipeline = HappinessPipeline("world_happiness_dataset.csv")
        pipeline.run()
    """

    def __init__(self, filepath: str):
        # save path to csv file
        self.filepath = filepath

        # these will be fill later when pipeline run
        self.cleaner    = None
        self.analyzer   = None
        self.visualizer = None

    def run(self) -> None:
        """
        run full pipeline from load data until save all chart.
        step 1: load and clean data
        step 2: compute pearson correlation
        step 3: draw and save all 4 chart
        """
        print("=" * 55)
        print("  World Happiness Analysis Pipeline Start")
        print("=" * 55)

        # --- step 1: data cleaning ---
        # create cleaner object, load file, then run all cleaning
        self.cleaner = DataCleaner(self.filepath)
        self.cleaner.load().clean()

        # get clean dataframe and numeric column list for next step
        df_clean     = self.cleaner.get_dataframe()
        numeric_cols = self.cleaner.numeric_cols

        # --- step 2: correlation analysis ---
        # create analyzer and compute pearson correlation matrix
        self.analyzer = CorrelationAnalyzer(df_clean, numeric_cols)
        self.analyzer.compute()

        # get correlation matrix for visualization
        corr_matrix = self.analyzer.get_matrix()

        # --- step 3: visualization ---
        # create visualizer and run all 4 chart
        self.visualizer = Visualizer(df_clean, corr_matrix)
        self.visualizer.run_all()

        print("=" * 55)
        print("  Pipeline finish. All output file is saved.")
        print("=" * 55)


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # create pipeline with path to csv file and run it
    pipeline = HappinessPipeline("world_happiness_dataset.csv")
    pipeline.run()
