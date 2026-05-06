# World Happiness Dataset: Analysis

A data cleaning and visualization project exploring happiness scores and contributing factors across 20 countries.

---

## Dataset Overview

| Property | Value |
|----------|-------|
| File | `world_happiness_dataset.csv` |
| Rows | 20 countries |
| Columns | 8 features |
| Missing values | None |
| Outliers detected | None (IQR method) |

### Columns

| Column | Description |
|--------|-------------|
| `Country` | Country name |
| `Happiness_Score` | Overall happiness score (0–10) |
| `GDP_per_Capita` | Economic output per person |
| `Social_Support` | Perceived social support level |
| `Healthy_Life_Expectancy` | Expected healthy years of life |
| `Freedom_to_Make_Choices` | Perceived freedom in life decisions |
| `Generosity` | Charitable giving level |
| `Perceptions_of_Corruption` | Perceived public/government corruption |

---

## Data Cleaning Steps

1. **Duplicate removal**: Checked and removed exact duplicate rows (none found).
2. **Whitespace stripping**: Trimmed leading/trailing spaces from string columns.
3. **Missing value check**: Confirmed zero missing values across all columns.
4. **Outlier detection (IQR method)**: Applied the 1.5 × IQR rule to every numeric column. No outliers were detected, meaning all values fall within statistically expected ranges for this 20-country sample.

---

## Visualizations

### Chart 1: Happiness Score by Country
Horizontal bar chart ranking all 20 countries by happiness score.
- **Green** = High (≥ 6.0): Canada, Brazil, Finland, Netherlands, Japan, Norway, China
- **Orange** = Medium (4.5 – 6.0): South Korea, UK, Australia
- **Red** = Low (< 4.5): Iceland, Switzerland, India, USA, France, New Zealand, Sweden, Denmark, Germany, South Africa

### Chart 2: GDP per Capita vs Happiness Score
Scatter plot with a Pearson regression line. Each point is coloured by Healthy Life Expectancy.
- The regression line (r ≈ 0.01) shows almost no linear relationship between GDP and happiness in this dataset.
- Countries with higher life expectancy (darker dots) tend to cluster in the middle happiness range.

### Chart 3: Feature Comparison: Top 5 vs Bottom 5
Grouped bar chart comparing average feature values between the five happiest and five least happy countries.
- Top-5 countries score higher on Life Expectancy and Social Support.
- Bottom-5 countries show higher Perceptions of Corruption on average.

### Chart 4: Pearson Correlation Heatmap
Full correlation matrix using Pearson's r.

| Feature | r with Happiness |
|---------|-----------------|
| Healthy Life Expectancy | +0.16 |
| Freedom to Make Choices | +0.08 |
| Social Support | +0.02 |
| GDP per Capita | +0.01 |
| Generosity | −0.15 |
| Perceptions of Corruption | −0.34 |

The strongest (negative) correlation is between Corruption Perceptions and Happiness Score.

---

## Key Findings

- **No outliers** were found in any column: the data is clean and consistent.
- **Canada** has the highest happiness score (7.34); **South Africa** the lowest (3.53).
- **GDP does not strongly predict happiness** in this sample (r = 0.01).
- **Higher corruption perception is the strongest predictor of lower happiness** (r = −0.34).
- **Healthy Life Expectancy** shows the strongest positive association with happiness (r = +0.16).
- Top-5 happiest countries consistently outperform bottom-5 on Life Expectancy and Social Support.

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scipy

# Place world_happiness_dataset.csv in the same folder, then:
python world_happiness_analysis.py
```

Four PNG chart files will be saved in the working directory.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scipy
```
