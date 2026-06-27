from pathlib import Path
import pandas as pd

class WellbeingDataLoader:
    """Load the attached NZ wellbeing Excel workbook."""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load(self) -> dict[str, pd.DataFrame]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        excel = pd.ExcelFile(self.data_path)
        sheets = {}
        for sheet in excel.sheet_names:
            if sheet.strip().lower() == "contents":
                continue
            sheets[sheet] = pd.read_excel(
                self.data_path,
                sheet_name=sheet,
                header=None,
                dtype=object
            )

        if not sheets:
            raise ValueError("No wellbeing data sheets were loaded.")
        return sheets
