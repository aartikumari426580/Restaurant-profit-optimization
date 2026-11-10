
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "restaurant_analysis.csv"
REPORTS_DIR = BASE_DIR / "reports"

df = pd.read_csv(DATA_PATH)

profit_columns = [
    "InStoreNetProfit",
    "UberEatsNetProfit",
    "DoorDashNetProfit",
    "SelfDeliveryNetProfit",
]

target = "TotalNetProfit"

# Inputs that would leak the answer into the model
leakage_columns = profit_columns + [target]

# Potential predictors: exclude IDs, names, target and direct profit components
excluded_columns = leakage_columns + ["RestaurantID", "RestaurantName"]

candidate_features = [
    col for col in df.columns
    if col not in excluded_columns
]

print("Dataset shape:", df.shape)
print("\nPrediction target:", target)
print("\nExcluded columns:")
print(excluded_columns)

print("\nCandidate features:")
print(candidate_features)

print("\nCandidate feature data types:")
print(df[candidate_features].dtypes.to_string())

print("\nNumeric feature summary:")
print(df[candidate_features].select_dtypes(include="number").describe().round(2))

print("\nTarget summary:")
print(df[target].describe().round(2))

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

pd.DataFrame({
    "CandidateFeature": candidate_features,
    "DataType": [str(df[col].dtype) for col in candidate_features],
}).to_csv(REPORTS_DIR / "candidate_features.csv", index=False)

print("\nFeature inspection completed.")
print("Saved: reports/candidate_features.csv")
