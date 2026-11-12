
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="AgriSmart Dashboard",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 AgriSmart")
st.subheader("Smart Agriculture: Crop Yield and Farmer Profitability")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("india_rainfall_crop_yield_2014_2024(2).csv")

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("Dashboard Menu")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard Overview",
        "Crop Analysis",
        "Crop Yield Prediction",
        "Farmer Profitability Prediction"
    ]
)

# ---------------- COMMON FEATURES ----------------
categorical_features = [
    "State", "Season", "Crop_Type",
    "Soil_Type", "Irrigation_Type"
]

numeric_features = [
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Humidity_percent",
    "Fertilizer_Used_kg",
    "Pesticide_Used_kg",
    "Cultivated_Area_hectares"
]

# ---------------- DASHBOARD OVERVIEW ----------------
if page == "Dashboard Overview":

    st.header("Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{len(df):,}")
    col2.metric(
        "Average Crop Yield",
        f"{df['Yield_ton_per_hectare'].mean():.2f} tons/ha"
    )
    col3.metric(
        "Average Farmer Profit",
        f"₹{df['Farmer_Profit_INR'].mean():,.0f}"
    )
    col4.metric(
        "Years Covered",
        f"{df['Year'].min()}–{df['Year'].max()}"
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Crop-wise Average Yield")
        crop_yield = df.groupby("Crop_Type")[
            "Yield_ton_per_hectare"
        ].mean().sort_values(ascending=False)

        st.bar_chart(crop_yield)

    with right:
        st.subheader("Year-wise Crop Yield")
        yearly_yield = df.groupby("Year")[
            "Yield_ton_per_hectare"
        ].mean()

        st.line_chart(yearly_yield)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ---------------- CROP ANALYSIS ----------------
elif page == "Crop Analysis":

    st.header("Crop Performance Analysis")

    crop_list = ["All"] + sorted(df["Crop_Type"].dropna().unique().tolist())
    selected_crop = st.selectbox("Select Crop", crop_list)

    filtered = df.copy()

    if selected_crop != "All":
        filtered = filtered[filtered["Crop_Type"] == selected_crop]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("State-wise Average Yield")
        state_yield = filtered.groupby("State")[
            "Yield_ton_per_hectare"
        ].mean().sort_values(ascending=False)

        st.bar_chart(state_yield)

    with col2:
        st.subheader("Season-wise Average Yield")
        season_yield = filtered.groupby("Season")[
            "Yield_ton_per_hectare"
        ].mean()

        st.bar_chart(season_yield)

    st.subheader("Crop Data")
    st.dataframe(filtered, use_container_width=True)

# ---------------- MODEL TRAINING FUNCTION ----------------
@st.cache_resource
def train_model(target):

    features = categorical_features + numeric_features

    if target == "Farmer_Profit_INR":
        features = features + ["Yield_ton_per_hectare"]

    X = df[features]
    y = df[target]

    categorical = [c for c in features if c in categorical_features]
    numerical = [c for c in features if c not in categorical_features]

    preprocessor = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),

        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numerical)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ])

    model.fit(X, y)

    return model, features


# ---------------- PREDICTION INPUTS ----------------
def prediction_inputs(include_yield=False):

    values = {}

    values["State"] = st.selectbox(
        "State",
        sorted(df["State"].dropna().unique())
    )

    values["Season"] = st.selectbox(
        "Season",
        sorted(df["Season"].dropna().unique())
    )

    values["Crop_Type"] = st.selectbox(
        "Crop",
        sorted(df["Crop_Type"].dropna().unique())
    )

    values["Soil_Type"] = st.selectbox(
        "Soil Type",
        sorted(df["Soil_Type"].dropna().unique())
    )

    values["Irrigation_Type"] = st.selectbox(
        "Irrigation Type",
        sorted(df["Irrigation_Type"].dropna().unique())
    )

    col1, col2 = st.columns(2)

    with col1:
        values["Rainfall_mm"] = st.number_input(
            "Rainfall (mm)", min_value=0.0, value=800.0
        )
        values["Avg_Temperature_C"] = st.number_input(
            "Temperature (°C)", value=25.0
        )
        values["Humidity_percent"] = st.number_input(
            "Humidity (%)", min_value=0.0, max_value=100.0, value=60.0
        )
        values["Fertilizer_Used_kg"] = st.number_input(
            "Fertilizer (kg)", min_value=0.0, value=200.0
        )

    with col2:
        values["Pesticide_Used_kg"] = st.number_input(
            "Pesticide (kg)", min_value=0.0, value=10.0
        )
        values["Cultivated_Area_hectares"] = st.number_input(
            "Cultivated Area (hectares)",
            min_value=0.01,
            value=2.0
        )

        if include_yield:
            values["Yield_ton_per_hectare"] = st.number_input(
                "Expected Yield (tons/hectare)",
                min_value=0.0,
                value=4.8
            )

    return values


# ---------------- CROP YIELD PREDICTION ----------------
elif page == "Crop Yield Prediction":

    st.header("🌱 Crop Yield Prediction")
    st.write("Estimate crop yield based on weather and farming conditions.")

    model, features = train_model("Yield_ton_per_hectare")

    with st.form("yield_form"):
        inputs = prediction_inputs()
        submitted = st.form_submit_button("Predict Crop Yield")

    if submitted:
        input_df = pd.DataFrame([inputs])[features]
        prediction = model.predict(input_df)[0]

        st.success(f"Predicted Crop Yield: {prediction:.2f} tons/hectare")

# ---------------- FARMER PROFITABILITY ----------------
elif page == "Farmer Profitability Prediction":

    st.header("💰 Farmer Profitability Prediction")
    st.write("Estimate farmer profit based on crop and farming conditions.")

    model, features = train_model("Farmer_Profit_INR")

    with st.form("profit_form"):
        inputs = prediction_inputs(include_yield=True)
        submitted = st.form_submit_button("Predict Farmer Profit")

    if submitted:
        input_df = pd.DataFrame([inputs])[features]
        prediction = model.predict(input_df)[0]

        st.success(f"Estimated Farmer Profit: ₹{prediction:,.2f}")

st.sidebar.divider()
st.sidebar.caption("AgriSmart | Smart Agriculture Dashboard")
