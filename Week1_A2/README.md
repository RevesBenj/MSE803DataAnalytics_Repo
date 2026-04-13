# Housing Price Analysis (Descriptive Analytics)

## Objective
Develop an initial story behind the Housing Prices Dataset using descriptive analytics.
Understand what factors: size, rooms, amenities, and extras - affect house prices.
Present the findings in a 5-slide presentation (3-minute delivery).

---

## Tools Used
- **Python**:  main scripting language
- **Pandas**: data loading, grouping, and aggregation
- **Matplotlib**:  data visualisation / chart generation
- **python-pptx**:  PowerPoint slide creation

---

## Dataset: Housing.csv
545 records, 13 columns.

| Column            | Type      | What it tells us                              |
|-------------------|-----------|-----------------------------------------------|
| price             | Number    | Selling price of the house                    |
| area              | Number    | Total floor size in square feet               |
| bedrooms          | Number    | Number of bedrooms (1 – 6)                    |
| bathrooms         | Number    | Number of bathrooms (1 – 4)                   |
| stories           | Number    | Number of floors / levels (1 – 4)             |
| parking           | Number    | Number of car parking spaces (0 – 3)          |
| mainroad          | Yes / No  | Is the house on a main road?                  |
| guestroom         | Yes / No  | Does it have a guest room?                    |
| basement          | Yes / No  | Does it have a basement?                      |
| hotwaterheating   | Yes / No  | Is there a hot water heating system?          |
| airconditioning   | Yes / No  | Does it have air conditioning?                |
| prefarea          | Yes / No  | Is it in a preferred / premium area?          |
| furnishingstatus  | Category  | Furnished, Semi-Furnished, or Unfurnished     |

---

## How to Run

```bash
# 1. Generate all charts (run from same folder as Housing.csv)
python analysis_W1A2.py

```

## Key Findings (Story Behind the Data)

### Finding 1: Size and Structure Drive Price
- **Area** is the strongest single driver of price. The scatter plot shows a clear upward trend: bigger house = higher price.
- **Bedrooms** matter: price rises from ~2.7M (1 bed) to ~5.8M (5 beds).
- **Bathrooms** matter even more: 4-bathroom homes average ~12.2M vs 4.2M for 1-bathroom homes.
- **Stories**: 4-story homes average ~7.2M vs ~4.2M for single-story homes.

### Finding 2: Amenities and Extras Add Value
- **Air conditioning** has the biggest amenity impact (+1.8M avg over no AC).
- **Main road access** adds ~+1.6M on average.
- **Preferred area** adds ~+1.5M on average.
- **Guest room** adds ~+1.2M; **basement** adds ~+0.7M; **hot water heating** adds ~+0.8M.
- **Parking**: 3 spaces averages ~5.9M vs 4.1M with no parking (+1.8M).
- **Furnishing**: Furnished homes (5.5M avg) cost 37% more than unfurnished (4.0M avg).

### Overall Conclusion
House price is driven by a combination of **size**, **room count**, **amenities**, and **extras**.
No single feature tells the whole story — the most expensive homes score well across all 13 factors.

---

## Author
- **Name:** Benjelyn Reves Patiag
- **Activity:** Week 1 - Activity 2
- **Date:** 13 Apr 2026
