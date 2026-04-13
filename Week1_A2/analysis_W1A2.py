
# -------------------------------------------------------
# Week 1 - Activity 2 : Housing Prices Dataset
# Author: Benjelyn Reves Patiag
# Date Created: 12 Apr 2026
# Description: Develop an initial story behind the Housing Prices Dataset.
#              Covers ALL 13 columns. Creates 7 charts saved to screenshots/.
# -------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Setup ────────────────────────────────────────────────────────────────────
df = pd.read_csv("Housing.csv")
os.makedirs("screenshots", exist_ok=True)

# Shared style
BLUE       = "#1B4F72"
LIGHT_BLUE = "#2186C6"
GRAY       = "#7F8C8D"
BAR_COLOR  = "#1B4F72"
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.titlesize":  13,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

def fmt_price(val, _=None):
    """Format y-axis labels as e.g. 4.5M"""
    return f"{val/1_000_000:.1f}M"

# ── Chart 1: Area vs Price (scatter) ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df["area"], df["price"], alpha=0.45, color=LIGHT_BLUE, edgecolors="none", s=25)
ax.set_title("House Size (Area) vs Price", fontweight="bold", color=BLUE)
ax.set_xlabel("Area (sq ft)")
ax.set_ylabel("Price")
ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_price))
plt.tight_layout()
plt.savefig("screenshots/area_vs_price.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Chart 2: Average Price by Bedrooms ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 3.5))
data = df.groupby("bedrooms")["price"].mean()
bars = ax.bar(data.index.astype(str), data.values, color=BAR_COLOR, width=0.55)
ax.set_title("Average Price by Bedrooms", fontweight="bold", color=BLUE)
ax.set_xlabel("Number of Bedrooms")
ax.set_ylabel("Average Price")
ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_price))
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50000,
            fmt_price(bar.get_height()), ha="center", va="bottom", fontsize=8, color=GRAY)
plt.tight_layout()
plt.savefig("screenshots/bedrooms_price.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Chart 3: Average Price by Bathrooms ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 3.5))
data = df.groupby("bathrooms")["price"].mean()
bars = ax.bar(data.index.astype(str), data.values, color=BAR_COLOR, width=0.45)
ax.set_title("Average Price by Bathrooms", fontweight="bold", color=BLUE)
ax.set_xlabel("Number of Bathrooms")
ax.set_ylabel("Average Price")
ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_price))
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50000,
            fmt_price(bar.get_height()), ha="center", va="bottom", fontsize=8, color=GRAY)
plt.tight_layout()
plt.savefig("screenshots/bathrooms_price.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Chart 4: Average Price by Stories ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 3.5))
data = df.groupby("stories")["price"].mean()
bars = ax.bar(data.index.astype(str), data.values, color=BAR_COLOR, width=0.45)
ax.set_title("Average Price by Stories (Floors)", fontweight="bold", color=BLUE)
ax.set_xlabel("Number of Stories")
ax.set_ylabel("Average Price")
ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_price))
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50000,
            fmt_price(bar.get_height()), ha="center", va="bottom", fontsize=8, color=GRAY)
plt.tight_layout()
plt.savefig("screenshots/stories_price.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Chart 5: Amenities (Yes/No features) ────────────────────────────────────
amenity_cols = {
    "mainroad":        "Main Road",
    "guestroom":       "Guest Room",
    "basement":        "Basement",
    "hotwaterheating": "Hot Water Heating",
    "airconditioning": "Air Conditioning",
    "prefarea":        "Preferred Area",
}
yes_prices = [df[df[col] == "yes"]["price"].mean() for col in amenity_cols]
no_prices  = [df[df[col] == "no"]["price"].mean()  for col in amenity_cols]
labels     = list(amenity_cols.values())

x = range(len(labels))
w = 0.38
fig, ax = plt.subplots(figsize=(7, 4))
b1 = ax.bar([i - w/2 for i in x], yes_prices, width=w, color=BLUE,       label="Has Feature")
b2 = ax.bar([i + w/2 for i in x], no_prices,  width=w, color=LIGHT_BLUE, label="No Feature",  alpha=0.65)
ax.set_title("Average Price: Has Feature vs No Feature", fontweight="bold", color=BLUE)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
ax.set_ylabel("Average Price")
ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_price))
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("screenshots/amenities_price.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Chart 6: Average Price by Parking Spaces ────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 3.5))
data = df.groupby("parking")["price"].mean()
bars = ax.bar(data.index.astype(str), data.values, color=BAR_COLOR, width=0.45)
ax.set_title("Average Price by Parking Spaces", fontweight="bold", color=BLUE)
ax.set_xlabel("Number of Parking Spaces")
ax.set_ylabel("Average Price")
ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_price))
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50000,
            fmt_price(bar.get_height()), ha="center", va="bottom", fontsize=8, color=GRAY)
plt.tight_layout()
plt.savefig("screenshots/parking_price.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Chart 7: Average Price by Furnishing Status ──────────────────────────────
fig, ax = plt.subplots(figsize=(5, 3.5))
data = df.groupby("furnishingstatus")["price"].mean()
order = ["furnished", "semi-furnished", "unfurnished"]
data  = data.reindex(order)
labels_f = ["Furnished", "Semi-Furnished", "Unfurnished"]
bars = ax.bar(labels_f, data.values, color=[BLUE, LIGHT_BLUE, GRAY], width=0.5)
ax.set_title("Average Price by Furnishing Status", fontweight="bold", color=BLUE)
ax.set_xlabel("Furnishing Status")
ax.set_ylabel("Average Price")
ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_price))
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50000,
            fmt_price(bar.get_height()), ha="center", va="bottom", fontsize=8, color=GRAY)
plt.tight_layout()
plt.savefig("screenshots/furnishing_price.png", dpi=150, bbox_inches="tight")
plt.close()

print("  All 7 charts saved to screenshots/")
print("    area_vs_price.png")
print("    bedrooms_price.png")
print("    bathrooms_price.png")
print("    stories_price.png")
print("    amenities_price.png")
print("    parking_price.png")
print("    furnishing_price.png")
