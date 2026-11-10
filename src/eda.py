
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "restaurant_data.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("=== RESTAURANT PROFIT OPTIMIZATION ===")
print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

# Calculate total profit across the four channels
profit_columns = [
    "InStoreNetProfit",
    "UberEatsNetProfit",
    "DoorDashNetProfit",
    "SelfDeliveryNetProfit",
]

df["TotalNetProfit"] = df[profit_columns].sum(axis=1)

print("\n=== OVERALL PROFIT SUMMARY ===")
print(df["TotalNetProfit"].describe().round(2))

print("\n=== AVERAGE PROFIT BY CUISINE ===")
cuisine_profit = (
    df.groupby("CuisineType")["TotalNetProfit"]
    .mean()
    .sort_values(ascending=False)
    .round(2)
)
print(cuisine_profit.to_string())

print("\n=== TOTAL PROFIT BY CHANNEL ===")
channel_profit = df[profit_columns].sum().sort_values(ascending=False)
print(channel_profit.round(2).to_string())

# Save reports
cuisine_profit.to_csv(REPORTS_DIR / "profit_by_cuisine.csv")
channel_profit.to_csv(REPORTS_DIR / "profit_by_channel.csv")
df.to_csv(PROCESSED_DIR / "restaurant_analysis.csv", index=False)

# Create a chart
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df,
    x="CuisineType",
    y="TotalNetProfit",
    errorbar=None,
)
plt.title("Average Total Net Profit by Cuisine")
plt.xlabel("Cuisine Type")
plt.ylabel("Total Net Profit")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "profit_by_cuisine.png", dpi=150)
plt.close()

print("\nEDA completed successfully!")
print("Processed data saved in data/processed/")
print("Reports and chart saved in reports/")
