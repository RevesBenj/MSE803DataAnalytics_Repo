# -------------------------------------------------------
# Week 3 - Activity 1 : Calculate the correlation between two features
# Author: Benjelyn Reves Patiag
# Date Created: 26 Apr 2026
# Description: Develop code to calculate and visualize the correlation between two selected features from the attached file to demonstrate the relationship between the data. Explain the outcome of analysis.
## -------------------------------------------------------



import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load dataset
df = pd.read_csv('age_networth.csv')

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Calculate correlation
r, p_value = pearsonr(df['age'], df['net_worth'])

print("Pearson r:", r)
print("P-value:", p_value)

# Visualization
plt.figure()
plt.scatter(df['age'], df['net_worth'])
plt.xlabel('Age')
plt.ylabel('Net Worth')
plt.title('Age vs Net Worth (Scipy)')
plt.show()