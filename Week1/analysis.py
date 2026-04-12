# -------------------------------------------------------
# Week1- Activity 1: Develop the descriptive analysis using data aggregation
# Author: Benjelyn Reves Patiag
# Date Created: 12 Apr 2026
# Description: Load the attached file and perform an initial analysis to identify the story behind the data using data aggregation techniques only. Then share your GitHub repository link and include a README file with your description and a screenshot of the final outcome.
# -------------------------------------------------------



import pandas as pd
import matplotlib.pyplot as plt

# load the excel file
file_path = "data/Data_set_w1A1.xlsx"

# read the excel file
df = pd.read_excel(file_path)

# show the first rows
print("Raw data preview:")
print(df.head())

# descriptive analysis using data aggregation only
# group by category and calculate summary values
summary = df.groupby("category").agg(
    sales_sum=("sales", "sum"),
    sales_mean=("sales", "mean"),
    sales_count=("sales", "count"),
    quantity_sum=("quantity", "sum"),
    quantity_mean=("quantity", "mean")
).reset_index()

# round decimal values for better display
summary["sales_mean"] = summary["sales_mean"].round(2)
summary["quantity_mean"] = summary["quantity_mean"].round(2)

print("\nAggregated Summary:")
print(summary)

# save result to csv
summary.to_csv("aggregated_summary.csv", index=False)

# create a simple bar chart for total sales
plt.figure(figsize=(8, 5))
plt.bar(summary["category"], summary["sales_sum"])
plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales Sum")
plt.tight_layout()
plt.savefig("screenshots/final_output.png")
plt.show()