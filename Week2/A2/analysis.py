# -------------------------------------------------------
# Week 2 - Activity 2 : statistical analysis using Beijing Multi-Site Air Quality.
# Author: Benjelyn Reves Patiag
# Date Created: 21 Apr 2026
# Description: See link: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
# Task 1 to Task 6 - load, clean, analyse, filter, visualize, and correlate data
# -------------------------------------------------------

import pandas as pd
import glob
import matplotlib
matplotlib.use('Agg')  # use no-display backend, no need window
import matplotlib.pyplot as plt
import seaborn as sns
import os

# make folder for save all picture output
os.makedirs("output", exist_ok=True)

# -------------------------------------------------------
# TASK 1: Load data and first look
# -------------------------------------------------------

# find all csv file inside data folder
files = glob.glob("data/*.csv")

df_list = []

# loop every file and read it, put in list
for file in files:
    temp_df = pd.read_csv(file)
    df_list.append(temp_df)

# combine all file into one big table
df = pd.concat(df_list, ignore_index=True)

# show first 5 row to see what data look like
print("=" * 60)
print("TASK 1: First look at data")
print("=" * 60)
print("\nFirst 5 rows:")
print(df.head())

# show all column name and what type they are
print("\nColumn names and data types:")
print(df.dtypes)

# count how many row and column total
print("\nShape of dataset (rows, columns):")
print(df.shape)


# -------------------------------------------------------
# TASK 2: Data Cleaning - fix the messy data
# -------------------------------------------------------

print("\n" + "=" * 60)
print("TASK 2: Data Cleaning")
print("=" * 60)

# check how many missing value in each column
print("\nMissing values per column:")
print(df.isnull().sum())

# replace missing number value using mean of that column
# mean is middle average value, better than just delete
df.fillna(df.mean(numeric_only=True), inplace=True)

# if still got missing value (like text column), remove that row
df.dropna(inplace=True)

# check again after clean, should be zero missing now
print("\nMissing values after cleaning (should be 0 or small):")
print(df.isnull().sum())

# confirm how many row left after cleaning
print(f"\nRows remaining after cleaning: {len(df)}")


# -------------------------------------------------------
# TASK 3: Basic Statistical Analysis - study the numbers
# -------------------------------------------------------

print("\n" + "=" * 60)
print("TASK 3: Basic Statistical Analysis")
print("=" * 60)

# pick only pollution and weather number column for calculation
numeric_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

# mean = average value of each column
print("\nMean (average):")
print(df[numeric_cols].mean().round(2))

# median = middle value when you sort all number
print("\nMedian (middle value):")
print(df[numeric_cols].median().round(2))

# minimum = smallest number in column
print("\nMinimum (smallest value):")
print(df[numeric_cols].min().round(2))

# maximum = biggest number in column
print("\nMaximum (biggest value):")
print(df[numeric_cols].max().round(2))

# standard deviation = how much value spread away from average
print("\nStandard Deviation (how much spread):")
print(df[numeric_cols].std().round(2))


# -------------------------------------------------------
# TASK 4: Data Filtering - look at each station separate
# -------------------------------------------------------

print("\n" + "=" * 60)
print("TASK 4: Data Filtering by Station")
print("=" * 60)

# show all station name in the data
print("\nAll station names:")
print(df['station'].unique())

# filter data for one specific station - Dongsi
station_name = 'Dongsi'
df_station = df[df['station'] == station_name]
print(f"\nFiltered data for station: {station_name}")
print(f"Total rows for this station: {len(df_station)}")

# calculate average pollution for each station
# groupby station then take mean of pollution column
print("\nAverage PM2.5 per station (from high to low):")
avg_pollution = df.groupby('station')[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].mean().round(2)
avg_pollution_sorted = avg_pollution.sort_values('PM2.5', ascending=False)
print(avg_pollution_sorted)

# save this table to csv so easy to check later
avg_pollution_sorted.to_csv("output/avg_pollution_by_station.csv")
print("\nSaved average pollution table to output/avg_pollution_by_station.csv")


# -------------------------------------------------------
# TASK 5: Data Visualization - make picture from data
# -------------------------------------------------------

print("\n" + "=" * 60)
print("TASK 5: Data Visualization")
print("=" * 60)

# --- Plot 1: Histogram of PM2.5 ---
# histogram show how many time each pollution level happen
plt.figure(figsize=(10, 5))
plt.hist(df['PM2.5'].dropna(), bins=50, color='steelblue', edgecolor='white', alpha=0.85)
plt.title('Distribution of PM2.5 Levels in Beijing', fontsize=14, fontweight='bold')
plt.xlabel('PM2.5 Concentration (µg/m³)', fontsize=12)
plt.ylabel('Frequency (count)', fontsize=12)
plt.axvline(df['PM2.5'].mean(), color='red', linestyle='--', linewidth=1.5, label=f"Mean = {df['PM2.5'].mean():.1f}")
plt.legend()
plt.tight_layout()
plt.savefig('output/histogram_pm25.png', dpi=150)
plt.close()
print("Saved histogram_pm25.png")

# --- Plot 2: Line plot PM2.5 over time ---
# line plot show how pollution change by month and year
# first make year-month column for group
df['year_month'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
monthly_avg = df.groupby('year_month')['PM2.5'].mean().reset_index()

plt.figure(figsize=(12, 5))
plt.plot(monthly_avg['year_month'], monthly_avg['PM2.5'], color='steelblue', linewidth=1.5)
plt.fill_between(monthly_avg['year_month'], monthly_avg['PM2.5'], alpha=0.2, color='steelblue')
plt.title('Monthly Average PM2.5 Levels Over Time (All Stations)', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average PM2.5 (µg/m³)', fontsize=12)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('output/lineplot_pm25_overtime.png', dpi=150)
plt.close()
print("Saved lineplot_pm25_overtime.png")

# --- Plot 3: Boxplot of pollutants ---
# boxplot show spread and outlier of each pollutant together
# need to normalise because each have different scale
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3']
plt.figure(figsize=(10, 6))
df_box = df[pollutants].copy()
# sample for speed, too many row make slow
df_sample = df_box.sample(n=min(10000, len(df_box)), random_state=42)
df_sample.boxplot(column=pollutants, grid=False,
                  boxprops=dict(color='steelblue'),
                  medianprops=dict(color='red', linewidth=2),
                  whiskerprops=dict(color='steelblue'),
                  capprops=dict(color='steelblue'),
                  flierprops=dict(marker='o', color='gray', alpha=0.2, markersize=3))
plt.title('Boxplot of Air Pollutants (PM2.5, PM10, SO2, NO2, O3)', fontsize=13, fontweight='bold')
plt.xlabel('Pollutant', fontsize=12)
plt.ylabel('Concentration (µg/m³)', fontsize=12)
plt.tight_layout()
plt.savefig('output/boxplot_pollutants.png', dpi=150)
plt.close()
print("Saved boxplot_pollutants.png")


# -------------------------------------------------------
# TASK 6: Correlation Analysis - find connection between variable
# -------------------------------------------------------

print("\n" + "=" * 60)
print("TASK 6: Correlation Analysis")
print("=" * 60)

# calculate correlation of all numeric column
# correlation number between -1 and 1
# close to 1 = strong positive relation
# close to -1 = strong negative relation
# close to 0 = no relation
corr_matrix = df[numeric_cols].corr()

# find which variable most correlated with PM2.5
pm25_corr = corr_matrix['PM2.5'].drop('PM2.5').sort_values(ascending=False)
print("\nCorrelation with PM2.5 (most to least):")
print(pm25_corr.round(3))

# top positive and negative correlation
print(f"\nMost positively correlated with PM2.5: {pm25_corr.idxmax()} ({pm25_corr.max():.3f})")
print(f"Most negatively correlated with PM2.5: {pm25_corr.idxmin()} ({pm25_corr.min():.3f})")

# check if temperature affect pollution
temp_pm25_corr = df['TEMP'].corr(df['PM2.5'])
print(f"\nCorrelation between TEMP and PM2.5: {temp_pm25_corr:.3f}")
if temp_pm25_corr < -0.2:
    print("Temperature have negative relation with PM2.5 - when temperature go up, pollution go down a bit")
elif temp_pm25_corr > 0.2:
    print("Temperature have positive relation with PM2.5 - when temperature go up, pollution go up too")
else:
    print("Temperature not have strong direct relation with PM2.5")

# --- Heatmap: show all correlation in colour map ---
# dark colour = strong correlation, light colour = weak
plt.figure(figsize=(10, 8))
mask = None
sns.heatmap(corr_matrix,
            annot=True,        # show number inside box
            fmt='.2f',         # 2 decimal place
            cmap='coolwarm',   # red = positive, blue = negative
            center=0,
            linewidths=0.5,
            square=True)
plt.title('Correlation Heatmap of All Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/heatmap_correlation.png', dpi=150)
plt.close()
print("Saved heatmap_correlation.png")

# --- Scatter plot: Temperature vs PM2.5 ---
# scatter show dots, each dot is one data point
plt.figure(figsize=(8, 5))
# sample data because too many point make slow
df_scatter = df[['TEMP', 'PM2.5']].dropna().sample(n=min(5000, len(df)), random_state=42)
plt.scatter(df_scatter['TEMP'], df_scatter['PM2.5'],
            alpha=0.2, color='steelblue', s=10)
# add trend line to see direction
z = df_scatter['TEMP'].values
m, b = pd.Series(df_scatter['PM2.5'].values).values, None
import numpy as np
coeffs = np.polyfit(df_scatter['TEMP'], df_scatter['PM2.5'], 1)
trendline = np.poly1d(coeffs)
temp_range = np.linspace(df_scatter['TEMP'].min(), df_scatter['TEMP'].max(), 100)
plt.plot(temp_range, trendline(temp_range), color='red', linewidth=2,
         label=f'Trend (r = {temp_pm25_corr:.2f})')
plt.title('Temperature vs PM2.5 (Does Weather Affect Pollution?)', fontsize=13, fontweight='bold')
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('PM2.5 (µg/m³)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('output/scatter_temp_vs_pm25.png', dpi=150)
plt.close()
print("Saved scatter_temp_vs_pm25.png")

print("\n" + "=" * 60)
print("ALL DONE! Check output/ folder for all saved charts.")
print("=" * 60)
