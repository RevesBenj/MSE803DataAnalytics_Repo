Data Aggregation using World Happiness Dataset

**Prepared by:** Benjelyn Reves Patiag  
**Subject:** MSE803 Data Analytics

---

## Overview

This activity use the World Happiness dataset to perform SQL data aggregation and comparison. The analysis is focus on two question:

1. **GDP and Happiness**: Can we see if richer country tend to be happier? We group country by GDP level, calculate average happiness per group, and rank country inside each group.
2. **Corruption and Happiness**: Do country with high corruption perception have lower happiness? We split country into two group based on corruption score and compare many happiness indicator between them.

---

## Dataset

The dataset have country-level happiness and social/economic indicator:

| Column | Description |
|---|---|
| `Country` | Name of country |
| `Happiness_Score` | Overall happiness score (higher = more happy) |
| `GDP_per_Capita` | Economic output per person (normalised 0–2 scale) |
| `Social_Support` | Feeling of having someone to depend on |
| `Healthy_Life_Expectancy` | Expected number of healthy life year |
| `Freedom_to_Make_Choices` | Freedom score (0–1 scale) |
| `Generosity` | Charitable giving score |
| `Perceptions_of_Corruption` | How much people feel corruption exist in society |

---

## Script Architecture

The Python script is organise into **seven classes**, each with single responsibility:

```
WorldHappinessAnalysis          ← Boss class, coordinate everything
├── Config                      ← Hold all path and setting
├── WorldHappinessQueries       ← Hold the two SQL query string
├── DatabaseManager             ← Load CSV into SQLite, run query
├── ResultExporter              ← Save result as CSV and PNG image
└── ConsolePrinter              ← Print formatted output to terminal
```

---

## Query 1: GDP Categories, Average Happiness, and Country Ranking

### Purpose

We want to understand if economic wealth (measured by GDP per capita) is connected to happiness level. To do this properly, we need to:

1. Classify every country into one of three GDP tier
2. See the average happiness for each tier (not just individual country)
3. Rank country within each tier so we can identify the top performer

### GDP Category Boundaries

| Category | GDP per Capita Range | Interpretation |
|---|---|---|
| Low GDP | Below 0.80 | Lower-income country in this dataset |
| Medium GDP | 0.80 to below 1.20 | Middle-income country |
| High GDP | 1.20 and above | Higher-income country |

These boundary are chosen based on the distribution of normalised GDP value in this dataset. The values are not real dollar: they are normalised score from the World Happiness Report.

---

### SQL Query

```sql
WITH gdp_grouped AS (
    SELECT
        Country,
        Happiness_Score,
        GDP_per_Capita,
        CASE
            WHEN GDP_per_Capita < 0.80 THEN 'Low GDP'
            WHEN GDP_per_Capita >= 0.80 AND GDP_per_Capita < 1.20 THEN 'Medium GDP'
            ELSE 'High GDP'
        END AS GDP_Category
    FROM world_happiness
),
category_average AS (
    SELECT
        GDP_Category,
        ROUND(AVG(Happiness_Score), 2) AS Average_Happiness
    FROM gdp_grouped
    GROUP BY GDP_Category
),
ranked_countries AS (
    SELECT
        Country,
        GDP_Category,
        GDP_per_Capita,
        Happiness_Score,
        RANK() OVER (
            PARTITION BY GDP_Category
            ORDER BY Happiness_Score DESC
        ) AS Happiness_Rank
    FROM gdp_grouped
)
SELECT
    r.GDP_Category,
    r.Country,
    r.GDP_per_Capita,
    r.Happiness_Score,
    a.Average_Happiness,
    r.Happiness_Rank
FROM ranked_countries r
JOIN category_average a
    ON r.GDP_Category = a.GDP_Category
ORDER BY
    CASE r.GDP_Category
        WHEN 'Low GDP' THEN 1
        WHEN 'Medium GDP' THEN 2
        WHEN 'High GDP' THEN 3
    END,
    r.Happiness_Rank;
```

---

### Explanation of Every Step in Detail

#### What is CTE (WITH clause)?

CTE stand for **Common Table Expression**. It is a way to create temporary named result set that only exist during the execution of one query. We use CTE because:

- It break a complicated query into smaller, readable piece
- Each CTE can reference the one before it
- It avoid repeating the same sub-query many time
- It make the logic flow from top to bottom in natural way

---

#### CTE 1: `gdp_grouped`: Label Every Country with GDP Category

**Purpose:** Add a new column called `GDP_Category` to every row using conditional logic.

**How CASE works:**

```sql
CASE
    WHEN GDP_per_Capita < 0.80  THEN 'Low GDP'
    WHEN GDP_per_Capita >= 0.80
     AND GDP_per_Capita <  1.20  THEN 'Medium GDP'
    ELSE                              'High GDP'
END AS GDP_Category
```

`CASE` work like an `if / else if / else` statement in programming.

**Result:** Every row in original table now has an extra column `GDP_Category`. If a country like Cambodia has GDP = 0.574, it get label `'Low GDP'`. If Australia has GDP = 1.372, it get `'High GDP'`.

**Why we need this CTE:** We cannot GROUP BY a column that does not exist yet in the table. By creating `gdp_grouped` first, we have the `GDP_Category` column ready for the next CTE to use.

---

#### CTE 2: `category_average`: Calculate Average Happiness per Category

**Purpose:** Collapse all country rows into one summary row per GDP category, showing the average happiness for that group.

```sql
SELECT
    GDP_Category,
    ROUND(AVG(Happiness_Score), 2) AS Average_Happiness
FROM gdp_grouped
GROUP BY GDP_Category
```

### GROUP BY + AVG 

- `GROUP BY GDP_Category` → group data into:
  - Low GDP  
  - Medium GDP  
  - High GDP  

- `AVG(Happiness_Score)` → get average happiness inside each group  

Example (Low GDP):
(4.5 + 5.0 + 4.8 + 5.2 + 4.9 + 5.5 + 5.4) / 7 = 5.04  

- `ROUND(..., 2)` → show only 2 decimal (for clean output)

### Result

- only 3 rows output  
- each row = one GDP group + average happiness  

### Why use this CTE

- compute average once only  
- reuse for all country rows  
- faster and easier to read  

---

#### CTE 3: `ranked_countries`: Rank Countries Within Each Category

**Purpose:** Give each country a rank number that represent its position in terms of happiness within its own GDP category.

```sql
RANK() OVER (
    PARTITION BY GDP_Category
    ORDER BY Happiness_Score DESC
) AS Happiness_Rank
```
**RANK() (Window Function)**

RANK() works on a group of rows (window), not like GROUP BY.

`OVER (...)` → define the window.

`PARTITION BY GDP_Category` → split data into groups  
→ ranking restart from 1 per group  

Example:
- Low GDP → rank 1 = happiest in Low GDP  
- High GDP → rank 1 = happiest in High GDP  

`ORDER BY Happiness_Score DESC` → highest score = rank 1

**RANK() vs ROW_NUMBER():**

| Function | Behaviour with Tie |
|---|---|
| `RANK()` | Two tied countries both get same rank, next rank is skipped (e.g. 1, 1, 3) |
| `ROW_NUMBER()` | Always assign unique number, even to tie (e.g. 1, 2, 3: arbitrary tiebreak) |
| `DENSE_RANK()` | Same rank for tie, but next rank is NOT skipped (e.g. 1, 1, 2) |

We use `RANK()` because if two countries have exactly same happiness score, they deserve the same rank position. The skipped rank number signals that a tie occurred.

**Result:** Same rows as `gdp_grouped` plus new `Happiness_Rank` column. Each country now know its rank within its GDP tier.

---

#### Final SELECT: Combine Everything

**Purpose:** Merge the country-level ranked data with the category-level average, then present in readable order.

```sql
FROM ranked_countries r
JOIN category_average a
    ON r.GDP_Category = a.GDP_Category
```

**How JOIN works here:**

**JOIN**

JOIN = combine 2 tables using matching column.

Here:
- match by `GDP_Category`  
- attach `Average_Happiness` to each country  

Example:
Canada (Low GDP) → get Low GDP average  

---

**ORDER BY with CASE**

Default order = alphabetical (High → Low → Medium)  

Use CASE to control order:
Low → Medium → High  

So output is logical, not alphabetical.

```sql
ORDER BY
    CASE r.GDP_Category
        WHEN 'Low GDP'    THEN 1
        WHEN 'Medium GDP' THEN 2
        WHEN 'High GDP'   THEN 3
    END,
    r.Happiness_Rank
```

We use `CASE` inside `ORDER BY` to assign a sort number to each category. SQLite then sort by these numbers (1, 2, 3) instead of alphabetically. Within each category (same sort number), it then sort by `Happiness_Rank` so rank 1 country appear first.

---

### Result Summary

| GDP Category | Average Happiness | Highest Ranked Country |
|---|---:|---|
| Low GDP | 5.22 | Canada |
| Medium GDP | 5.00 | Netherlands |
| High GDP | 5.30 | Brazil |

**Key Insight**

High GDP → higher average happiness.  
But inside each GDP group → still big variation.  
Meaning: GDP helps, but not enough to explain happiness.
---

## Query 2: High vs Low Corruption Perception Comparison

### Purpose

Goal is to check if corruption perception connect to happiness.

Countries split into 2 groups:

    high corruption perception
    low corruption perception

Then compare many wellbeing indicators between groups.

Use dataset average as benchmark (not fixed value).
This means threshold auto change when data change.

---

### SQL Query

```sql
WITH corruption_groups AS (
    SELECT
        Country,
        Happiness_Score,
        GDP_per_Capita,
        Social_Support,
        Healthy_Life_Expectancy,
        Freedom_to_Make_Choices,
        Generosity,
        Perceptions_of_Corruption,
        CASE
            WHEN Perceptions_of_Corruption >= (
                SELECT AVG(Perceptions_of_Corruption)
                FROM world_happiness
            ) THEN 'High Corruption Perception'
            ELSE 'Low Corruption Perception'
        END AS Corruption_Group
    FROM world_happiness
)
SELECT
    Corruption_Group,
    COUNT(*) AS Country_Count,
    ROUND(AVG(Happiness_Score), 2) AS Avg_Happiness,
    ROUND(AVG(GDP_per_Capita), 2) AS Avg_GDP,
    ROUND(AVG(Social_Support), 2) AS Avg_Social_Support,
    ROUND(AVG(Healthy_Life_Expectancy), 2) AS Avg_Life_Expectancy,
    ROUND(AVG(Freedom_to_Make_Choices), 2) AS Avg_Freedom,
    ROUND(AVG(Generosity), 2) AS Avg_Generosity,
    ROUND(AVG(Perceptions_of_Corruption), 2) AS Avg_Corruption_Perception,
    ROUND(
        AVG(Happiness_Score) - (
            SELECT AVG(Happiness_Score)
            FROM world_happiness
        ), 2
    ) AS Difference_From_Overall_Happiness
FROM corruption_groups
GROUP BY Corruption_Group
ORDER BY Avg_Happiness DESC;
```

---

### Deep Explanation: Every Step in Detail

#### CTE 1: `corruption_groups`: Label Each Country as High or Low Corruption Perception

**Purpose:** Create a new column `Corruption_Group` by comparing each country's corruption score against the overall dataset average.

**The scalar subquery as dynamic benchmark:**

```sql
CASE
    WHEN Perceptions_of_Corruption >= (
        SELECT AVG(Perceptions_of_Corruption)
        FROM world_happiness
    ) THEN 'High Corruption Perception'
    ELSE   'Low Corruption Perception'
END AS Corruption_Group
```

Scalar subquery:

`SELECT AVG(Perceptions_of_Corruption)` → return one value only (overall average).  

This value used as threshold in CASE.

---

Why use subquery:

If use fixed value (ex: 0.11) → not change when data change.  

Result can become wrong.

Using `AVG()` → auto adjust based on data.  

Meaning: threshold follow the data (data-driven).

**How the labelling logic works:**

- A country with `Perceptions_of_Corruption = 0.18` and average = `0.12` → `0.18 >= 0.12` → `'High Corruption Perception'`
- A country with `Perceptions_of_Corruption = 0.07` and average = `0.12` → `0.07 < 0.12` → `'Low Corruption Perception'`

**Important Note**

`Perceptions_of_Corruption` = people feeling about corruption.  

High score → people feel more corruption  
Low score → people trust more  

Note: not actual corruption, only perception.

**Result:** Same rows as original table plus `Corruption_Group` column. Every country is now labelled as either `'High Corruption Perception'` or `'Low Corruption Perception'`.

---

#### Final SELECT: Multiple Averages with GROUP BY and Subquery Comparison

**Purpose:** Summarise both corruption groups into two rows, calculating average for every happiness indicator, and showing how each group compare to the overall dataset average.

**How GROUP BY with multiple AVG() work:**

```sql
GROUP BY Corruption_Group
```

This create two "buckets": one for `High Corruption Perception` and one for `Low Corruption Perception`. Every country falls into one bucket.

Then `AVG()` is applied independently to each bucket. For example, `AVG(GDP_per_Capita)` calculate the mean GDP only among countries in the `Low Corruption Perception` bucket, and separately calculate the mean only among countries in the `High Corruption Perception` bucket.

We calculate eight different averages to give a complete picture:

| Column | What it tells us |
|---|---|
| `Country_Count` | Sample size of each group (is comparison fair?) |
| `Avg_Happiness` | Main outcome: is one group happier? |
| `Avg_GDP` | Do the groups differ in economic wealth? |
| `Avg_Social_Support` | Do the groups differ in community support? |
| `Avg_Life_Expectancy` | Do the groups differ in health outcomes? |
| `Avg_Freedom` | Do the groups differ in personal freedom? |
| `Avg_Generosity` | Do the groups differ in charitable behaviour? |
| `Avg_Corruption_Perception` | Verify that groups are correctly split |

**The `Difference_From_Overall_Happiness` column: another scalar subquery:**

```sql
ROUND(
    AVG(Happiness_Score) - (
        SELECT AVG(Happiness_Score)
        FROM world_happiness
    ), 2
) AS Difference_From_Overall_Happiness
```

Subtract overall average happiness from each group.

Result meaning:
- Positive → group more happy than overall  
- Negative → group less happy  
- Zero → same as overall  

Note:
CTE subquery → use for corruption threshold  
SELECT subquery → use for happiness comparison  

Different purpose.

**How ORDER BY DESC affect interpretation:**

```sql
ORDER BY Avg_Happiness DESC
```

Sort by happiness (DESC).
Top row = happier group.  
Easy to compare:  
gap between row 1 and row 2 = happiness difference.

---

### Result Summary

| Corruption Group | Country Count | Avg Happiness | Difference from Overall Happiness |
|---|---:|---:|---:|
| Low Corruption Perception | 9 | 5.63 | +0.46 |
| High Corruption Perception | 11 | 4.80 | -0.37 |

**Key Insight**

Low corruption perception → higher happiness.  
Difference ≈ 0.83 higher than high corruption group.  
Low corruption group: +0.46 above overall average  
High corruption group: -0.37 below overall average  

Meaning: more trust and good governance → more happiness.

---

## Overall Findings

1. High GDP → highest happiness  
2. Same GDP group → still big variation (not only GDP)  
3. Low corruption → higher happiness  
4. Low corruption → +0.46 above average  
5. High corruption → -0.37 below average  
6. GDP and corruption related to happiness, but no proof of cause

---


## How to Run

```bash
python sql_analysis.py
```

The script will:

1. Load the CSV file into SQLite database (using `DatabaseManager` class)
2. Write both SQL queries to `.sql` file (using `SqlFileWriter` class)
3. Run Query 1: GDP categories and ranking (using `DatabaseManager`)
4. Run Query 2: Corruption perception comparison (using `DatabaseManager`)
5. Export both result as CSV files (using `ResultExporter` class)
6. Save screenshot-style PNG image of both results (using `ResultExporter` class)
7. Print summary and findings to terminal (using `ConsolePrinter` class)

---

## Files Included

| File | Description |
|---|---|
| `world_happiness_dataset.csv` | The raw dataset |
| `sql_analysis.py` | Main Python script (OOP structure) |
| `world_happiness_queries.sql` | Both SQL queries with detailed comments |
| `outputs/query1_gdp_category_ranking_results.csv` | Query 1 result as CSV |
| `outputs/query2_corruption_group_comparison_results.csv` | Query 2 result as CSV |
| `outputs/query1_gdp_category_ranking_screenshot.png` | Query 1 result as PNG image |
| `outputs/query2_corruption_group_comparison_screenshot.png` | Query 2 result as PNG image |

---
