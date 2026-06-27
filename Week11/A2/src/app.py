from pathlib import Path
from src.data_loader import WellbeingDataLoader
from src.preprocessing import WellbeingPreprocessor
from src.models import ForecastModelTrainer
from src.visualization import VisualizationGenerator

class NZWellbeingForecastingApp:
    """Main controller."""

    def __init__(self, data_path, output_dir="output", models_dir="models", next_year=None, selected_series=None):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.models_dir = Path(models_dir)
        self.next_year = next_year
        self.selected_series = selected_series
        (self.output_dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        print("1. Loading attached Excel dataset...")
        sheets = WellbeingDataLoader(self.data_path).load()
        print(f"Loaded sheets: {len(sheets)}")

        print("2. Preprocessing complex workbook...")
        prepared = WellbeingPreprocessor().transform(sheets)
        print(f"Clean long records: {len(prepared.long_data)}")
        print(f"Model-ready records: {len(prepared.model_data)}")

        print("3. Training and comparing models...")
        trainer = ForecastModelTrainer(self.output_dir, self.models_dir)
        results = trainer.train_all(prepared.model_data, self.selected_series, self.next_year)

        print("4. Saving outputs...")
        prepared.long_data.to_csv(self.output_dir / "tables" / "clean_long_wellbeing_data.csv", index=False)
        prepared.model_data.to_csv(self.output_dir / "tables" / "model_ready_series_data.csv", index=False)
        prepared.quality_report.to_csv(self.output_dir / "tables" / "data_quality_report.csv", index=False)
        results["comparison"].to_csv(self.output_dir / "tables" / "model_comparison_metrics.csv", index=False)
        results["forecast"].to_csv(self.output_dir / "tables" / "next_year_forecast.csv", index=False)

        viz = VisualizationGenerator(self.output_dir / "figures")
        viz.selected_series(prepared.model_data, results["selected_arima_series"])
        viz.model_comparison(results["comparison"])
        viz.forecast_sample(results["forecast"])

        with open(self.output_dir / "activity_summary.txt", "w", encoding="utf-8") as f:
            f.write("NZ Wellbeing Forecasting Activity Summary\\n")
            f.write("=" * 45 + "\\n\\n")
            f.write(f"Best model by RMSE: {results['best_model']}\\n")
            f.write(f"Selected ARIMA series: {results['selected_arima_series']}\\n\\n")
            f.write(results["comparison"].to_string(index=False))
            f.write("\\n\\nForecast sample:\\n")
            f.write(results["forecast"].head(10).to_string(index=False))

        print("Done.")
        print(f"Best model: {results['best_model']}")
