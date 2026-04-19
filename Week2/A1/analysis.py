# -------------------------------------------------------
# Week 2 - Activity 1 : statistical analysis using  Beijing Multi-Site Air Quality.
# Author: Benjelyn Reves Patiag
# Date Created: 19 Apr 2026
# Description:  See link: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
# Describe the data structure and do just Task-1 including (share your GitHub link & readme file with the screenshot of your outcome ) :
#  Display the first 5 rows
# Identify column names and data types
# Count total rows and columns
# -------------------------------------------------------
import pandas as pd
import glob

# STEP 1: Load ALL datasets (multiple CSV files)
files = glob.glob("data/*.csv")   # put all csv inside /data folder

df_list = []

for file in files:
    temp_df = pd.read_csv(file)
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

# STEP 2: Display first 5 rows
print("First 5 rows:")
print(df.head())

# STEP 3: Column names and data types
print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

# STEP 4: Count rows and columns
print("\nShape of dataset:")
print(df.shape)

# STEP 5: Identify missing values
print("\nMissing values:")
print(df.isnull().sum())

# STEP 6: Replace missing values (numerical → mean)
df.fillna(df.mean(numeric_only=True), inplace=True)

# STEP 7: Remove rows if still missing
df.dropna(inplace=True)

# STEP 8: Basic Statistics
print("\nMean:")
print(df.mean(numeric_only=True))

print("\nMedian:")
print(df.median(numeric_only=True))

print("\nMinimum:")
print(df.min(numeric_only=True))

print("\nMaximum:")
print(df.max(numeric_only=True))

print("\nStandard Deviation:")
print(df.std(numeric_only=True))