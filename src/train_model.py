
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "restaurant_analysis.csv"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load processed dataset
df = pd.read_csv(DATA_PATH)

target = "TotalNetProfit"

# Remove rows where target is missing
df = df.dropna(subset=[target]).copy()

# Do not use IDs, names, or profit columns as predictors
excluded_columns = [
    target,
    "RestaurantID",
    "RestaurantName",
    "InStoreNetProfit",
    "UberEatsNetProfit",
    "DoorDashNetProfit",
    "SelfDeliveryNetProfit",
]

X = df.drop(columns=excluded_columns, errors="ignore")
y = df[target]

# Identify categorical and numeric features
categorical_features = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

print("Dataset rows:", len(df))
print("Target:", target)
print("Number of input features:", X.shape[1])
print("Categorical features:", categorical_features)
print("Numeric features:", len(numeric_features))

# Preprocess numeric and categorical columns
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# Hold out test data for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
}

results = []
trained_models = {}

for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

    trained_models[name] = pipeline

    print(f"\n{name}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")

# Select model with the lowest test RMSE
results_df = pd.DataFrame(results).sort_values("RMSE")
results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

joblib.dump(best_model, MODEL_DIR / "best_profit_model.joblib")

# Save the exact input columns used by the model
joblib.dump(X.columns.tolist(), MODEL_DIR / "feature_columns.joblib")

print("\n=== MODEL COMPARISON ===")
print(results_df.round(4).to_string(index=False))
print("\nBest model by test RMSE:", best_name)
print("Saved model: models/best_profit_model.joblib")
print("Saved feature list: models/feature_columns.joblib")
print("Saved metrics: reports/model_comparison.csv")
