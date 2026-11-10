
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "restaurant_analysis.csv"
MODEL_PATH = BASE_DIR / "models" / "best_profit_model.joblib"
FEATURES_PATH = BASE_DIR / "models" / "feature_columns.joblib"
COMPARISON_PATH = BASE_DIR / "reports" / "model_comparison.csv"

st.set_page_config(
    page_title="Restaurant Profit Optimization",
    page_icon="🍽️",
    layout="wide",
)

st.title("🍽️ Restaurant Profit Optimization")
st.caption("Profit analysis, channel comparison and machine learning estimates")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features

try:
    df = load_data()
    model, feature_columns = load_model()
except FileNotFoundError as exc:
    st.error(f"Required project file not found: {exc}")
    st.info("Run EDA and model training scripts before starting the dashboard.")
    st.stop()

profit_columns = [
    "InStoreNetProfit",
    "UberEatsNetProfit",
    "DoorDashNetProfit",
    "SelfDeliveryNetProfit",
]

if "TotalNetProfit" not in df.columns:
    df["TotalNetProfit"] = df[profit_columns].sum(axis=1)

# Overview
st.subheader("Business Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Restaurants", f"{len(df):,}")
col2.metric("Average Total Net Profit", f"{df['TotalNetProfit'].mean():,.2f}")
col3.metric("Highest Total Net Profit", f"{df['TotalNetProfit'].max():,.2f}")
col4.metric("Lowest Total Net Profit", f"{df['TotalNetProfit'].min():,.2f}")

st.divider()

# Historical dataset analysis
left, right = st.columns(2)

with left:
    st.subheader("Average Profit by Cuisine")
    cuisine = (
        df.groupby("CuisineType", as_index=False)["TotalNetProfit"]
        .mean()
        .sort_values("TotalNetProfit", ascending=False)
    )
    fig = px.bar(
        cuisine,
        x="CuisineType",
        y="TotalNetProfit",
        title="Average Total Net Profit",
    )
    fig.update_layout(xaxis_title="Cuisine", yaxis_title="Average Profit")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Profit by Sales Channel")
    channel_names = {
        "InStoreNetProfit": "In-Store",
        "UberEatsNetProfit": "Uber Eats",
        "DoorDashNetProfit": "DoorDash",
        "SelfDeliveryNetProfit": "Self Delivery",
    }
    channel_data = pd.DataFrame({
        "Channel": [channel_names[c] for c in profit_columns],
        "Total Profit": [df[c].sum() for c in profit_columns],
    })
    fig = px.bar(
        channel_data,
        x="Channel",
        y="Total Profit",
        title="Total Profit by Channel",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Model-based estimate
st.subheader("Estimate Restaurant Total Net Profit")
st.write(
    "Choose a restaurant as a starting point, adjust its operating inputs, "
    "and submit to estimate total net profit."
)

restaurant_labels = (
    df["RestaurantName"].astype(str)
    + " (ID: "
    + df["RestaurantID"].astype(str)
    + ")"
)
selected_label = st.selectbox("Choose a sample restaurant", restaurant_labels)
selected_index = restaurant_labels[restaurant_labels == selected_label].index[0]
base_row = df.loc[selected_index]

with st.form("profit_prediction_form"):
    st.write("Adjust the inputs below. Values initially come from the selected record.")

    input_values = {}
    columns = st.columns(2)

    for i, feature in enumerate(feature_columns):
        with columns[i % 2]:
            if feature not in df.columns:
                st.error(f"Missing model feature in dataset: {feature}")
                st.stop()

            if pd.api.types.is_numeric_dtype(df[feature]):
                minimum = float(df[feature].min())
                maximum = float(df[feature].max())
                default = float(base_row[feature])

                if pd.api.types.is_integer_dtype(df[feature]):
                    input_values[feature] = st.number_input(
                        feature,
                        min_value=int(minimum),
                        max_value=int(maximum),
                        value=int(default),
                        step=1,
                        key=f"input_{feature}",
                    )
                else:
                    input_values[feature] = st.number_input(
                        feature,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=0.01,
                        format="%.2f",
                        key=f"input_{feature}",
                    )
            else:
                options = sorted(df[feature].dropna().astype(str).unique().tolist())
                current = str(base_row[feature])
                default_index = options.index(current) if current in options else 0

                input_values[feature] = st.selectbox(
                    feature,
                    options=options,
                    index=default_index,
                    key=f"input_{feature}",
                )

    submitted = st.form_submit_button("Estimate Total Net Profit")

if submitted:
    input_df = pd.DataFrame([input_values], columns=feature_columns)
    prediction = float(model.predict(input_df)[0])

    st.subheader("Estimated Result")
    st.metric("Estimated Total Net Profit", f"{prediction:,.2f}")
    st.caption(
        "This is a model estimate based on the entered operating data, "
        "not a guarantee of actual profit or a verified future forecast."
    )

# Model comparison
if COMPARISON_PATH.exists():
    st.divider()
    st.subheader("Machine Learning Model Comparison")
    comparison = pd.read_csv(COMPARISON_PATH)
    st.dataframe(comparison.round(4), use_container_width=True)

st.divider()
st.caption("Restaurant Profit Optimization | Machine Learning Dashboard")
