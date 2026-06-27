from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

class VisualizationGenerator:
    """Create visualizations."""

    def __init__(self, figures_dir):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def selected_series(self, df, series_id):
        sdf = df[df["Series_ID"] == series_id].sort_values("Year")
        if sdf.empty:
            return
        plt.figure(figsize=(9,5))
        plt.plot(sdf["Year"], sdf["Value"], marker="o")
        plt.title("Selected NZ Wellbeing Time Series")
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "selected_series_trend.png", dpi=160)
        plt.close()

    def model_comparison(self, comparison):
        plt.figure(figsize=(9,5))
        plt.bar(comparison["Model"], comparison["RMSE"])
        plt.title("Model Comparison by RMSE")
        plt.xlabel("Model")
        plt.ylabel("RMSE")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "model_comparison_rmse.png", dpi=160)
        plt.close()

    def forecast_sample(self, forecast, n=15):
        sample = forecast.head(n)
        labels = sample["Topic"].str[:30] + " | " + sample["Group"].str[:18]
        plt.figure(figsize=(10,6))
        plt.barh(labels, sample["Forecast_Value"])
        plt.title("Sample Next-Year Forecast")
        plt.xlabel("Forecast Value")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "sample_next_year_forecast.png", dpi=160)
        plt.close()
