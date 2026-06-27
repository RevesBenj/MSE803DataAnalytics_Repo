import re
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

@dataclass
class PreparedData:
    long_data: pd.DataFrame
    model_data: pd.DataFrame
    quality_report: pd.DataFrame

class WellbeingPreprocessor:
    """
    Convert the complex Excel workbook into clean time-series data.

    Raw workbook problem:
    - many sheets
    - multiple header rows
    - blank separator columns
    - estimate, ASE, and change columns mixed together
    - demographic groups repeated in blocks
    """

    def __init__(self):
        self.missing_tokens = {"…", "...", "..", "S", "C", "np", "", "NA", "N/A"}

    def clean_text(self, x: Any) -> str | None:
        if pd.isna(x):
            return None
        s = str(x).strip()
        s = re.sub(r"\\s+", " ", s)
        s = re.sub(r"\\(\\d+\\)", "", s).strip()
        return s if s else None

    def to_number(self, x: Any) -> float | None:
        if pd.isna(x):
            return None
        if isinstance(x, str):
            s = x.strip().replace(",", "")
            if s in self.missing_tokens:
                return None
            try:
                return float(s)
            except ValueError:
                return None
        try:
            return float(x)
        except Exception:
            return None

    def is_year(self, x: Any) -> bool:
        try:
            y = int(float(x))
            return 2010 <= y <= 2035
        except Exception:
            return False

    def sheet_topic(self, df: pd.DataFrame, sheet_name: str) -> str:
        for r in range(min(5, len(df))):
            for v in df.iloc[r].tolist():
                t = self.clean_text(v)
                if t and not t.lower().startswith("table") and "april" not in t.lower():
                    return t
        return sheet_name

    def column_labels(self, df: pd.DataFrame) -> dict[int, dict[str, str]]:
        labels = {}
        for c in range(df.shape[1]):
            parts = []
            for r in range(4, min(11, df.shape[0])):
                t = self.clean_text(df.iat[r, c])
                if t:
                    parts.append(t)
            joined = " | ".join(dict.fromkeys(parts))
            if joined:
                labels[c] = {
                    "label": joined,
                    "metric": "ase" if "ASE" in joined.upper() else "estimate"
                }

        # forward fill labels for separated estimate columns
        last = None
        for c in range(df.shape[1]):
            if c in labels:
                last = labels[c]
            elif last:
                labels[c] = last
        return labels

    def make_series_id(self, *parts: str) -> str:
        raw = "__".join(str(p) for p in parts if p is not None)
        raw = re.sub(r"\\s+", "_", raw)
        raw = re.sub(r"[^A-Za-z0-9_\\-|.]+", "", raw)
        return raw[:220]

    def parse_sheet(self, sheet_name: str, df: pd.DataFrame) -> list[dict]:
        topic = self.sheet_topic(df, sheet_name)
        labels = self.column_labels(df)
        records = []

        group_type = "Total"
        group = "Total population"

        broad_groups = {
            "life stage", "sex", "ethnic group", "labour force status",
            "family type", "highest qualification", "region",
            "total personal income", "neighbourhood deprivation",
            "housing tenure", "migrant status", "disability status"
        }

        for r in range(df.shape[0]):
            year_cell = df.iat[r, 1] if df.shape[1] > 1 else None

            if not self.is_year(year_cell):
                texts = [self.clean_text(v) for v in df.iloc[r, 2:5].tolist()]
                texts = [t for t in texts if t]
                if texts:
                    t = texts[0]
                    if t.lower() in broad_groups:
                        group_type = t
                    elif not t.lower().startswith("change"):
                        group = t
                continue

            year = int(float(year_cell))
            for c in range(2, df.shape[1]):
                value = self.to_number(df.iat[r, c])
                if value is None:
                    continue

                info = labels.get(c, {"label": f"Column {c}", "metric": "estimate"})
                if info["metric"] != "estimate":
                    continue

                measure = info["label"]
                sid = self.make_series_id(sheet_name, topic, group_type, group, measure)

                records.append({
                    "Sheet": sheet_name,
                    "Topic": topic,
                    "Year": year,
                    "Group_Type": group_type,
                    "Group": group,
                    "Measure": measure,
                    "Value": value,
                    "Series_ID": sid
                })
        return records

    def transform(self, sheets: dict[str, pd.DataFrame]) -> PreparedData:
        records = []
        quality = []

        for name, raw in sheets.items():
            parsed = self.parse_sheet(name, raw)
            records.extend(parsed)
            quality.append({
                "Sheet": name,
                "Raw_Rows": raw.shape[0],
                "Raw_Columns": raw.shape[1],
                "Parsed_Records": len(parsed)
            })

        long_df = pd.DataFrame(records)
        if long_df.empty:
            raise ValueError("No numeric estimate time-series records found.")

        long_df = long_df.drop_duplicates()
        long_df["Year"] = long_df["Year"].astype(int)
        long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
        long_df = long_df.dropna(subset=["Year", "Value", "Series_ID"])

        year_count = long_df.groupby("Series_ID")["Year"].nunique().rename("Year_Count")
        model_df = long_df.merge(year_count, on="Series_ID", how="left")
        model_df = model_df[model_df["Year_Count"] >= 2].copy()
        model_df = model_df.sort_values(["Series_ID", "Year"]).reset_index(drop=True)

        model_df["Year_Index"] = model_df["Year"] - model_df["Year"].min()
        model_df["Is_Total_Population"] = model_df["Group"].str.contains("Total population", case=False, na=False).astype(int)

        model_df["Lag_1"] = model_df.groupby("Series_ID")["Value"].shift(1)
        model_df["Lag_2"] = model_df.groupby("Series_ID")["Value"].shift(2)
        model_df["Rolling_Mean_2"] = model_df.groupby("Series_ID")["Value"].transform(
            lambda s: s.shift(1).rolling(2, min_periods=1).mean()
        )

        for col in ["Lag_1", "Lag_2", "Rolling_Mean_2"]:
            model_df[col] = model_df.groupby("Series_ID")[col].transform(lambda s: s.fillna(s.mean()))
            model_df[col] = model_df[col].fillna(model_df["Value"].mean())

        quality_df = pd.DataFrame(quality)
        quality_df.loc[len(quality_df)] = ["TOTAL", quality_df["Raw_Rows"].sum(), "", len(long_df)]

        return PreparedData(long_df, model_df, quality_df)
