
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AgriSmart Dashboard",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 AgriSmart")
st.subheader("Smart Agriculture: Crop Yield and Farmer Profitability")

st.markdown("---")


# ==================================================
# 2. LOAD DATASET
# ==================================================


import os
import streamlit as st
import pandas as pd

@st.cache_data
def load_data():

    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "india_rainfall_crop_yield_2014_2024(2).csv"
    )

    if not os.path.exists(file_path):
        st.error(
            "Dataset not found! Please upload "
            "india_rainfall_crop_yield_2014_2024(2).csv "
            "to the same GitHub folder as app.py."
        )
        st.stop()

    return pd.read_csv(file_path)

df = load_data()


# Clean column names
df.columns = df.columns.str.strip()

# Required columns
required_columns = [
    "State",
    "Season",
    "Crop_Type",
    "Soil_Type",
    "Irrigation_Type",
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Humidity_percent",
    "Fertilizer_Used_kg",
    "Pesticide_Used_kg",
    "Cultivated_Area_hectares",
    "Yield_ton_per_hectare",
    "Farmer_Profit_INR",
    "Year"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing dataset columns: {missing_columns}")
    st.stop()


# ==================================================
# 3. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("🌾 AgriSmart Menu")

page = st.sidebar.radio(
    "Select Dashboard",
    [
        "Dashboard Overview",
        "Crop Analysis",
        "Crop Yield Prediction",
        "Farmer Profitability Prediction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("AgriSmart | Smart Agriculture")


# ==================================================
# 4. FEATURES FOR MACHINE LEARNING
# ==================================================

categorical_features = [
    "State",
    "Season",
    "Crop_Type",
    "Soil_Type",
    "Irrigation_Type"
]

numeric_features = [
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Humidity_percent",
    "Fertilizer_Used_kg",
    "Pesticide_Used_kg",
    "Cultivated_Area_hectares"
]


# ==================================================
# 5. MODEL TRAINING FUNCTION
# ==================================================

@st.cache_resource
def train_model(target):

    features = categorical_features + numeric_features

    if target == "Farmer_Profit_INR":
        features = features + ["Yield_ton_per_hectare"]

    X = df[features]
    y = df[target]

    categorical = [
        col for col in features
        if col in categorical_features
    ]

    numerical = [
        col for col in features
        if col not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore")
                    )
                ]),
                categorical
            ),
            (
                "numerical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]),
                numerical
            )
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ])

    model.fit(X, y)

    return model, features


# ==================================================
# 6. PREDICTION INPUT FUNCTION
# ==================================================

def prediction_inputs(include_yield=False):

    values = {}

    st.markdown("### Farming Information")

    col1, col2 = st.columns(2)

    with col1:

        values["State"] = st.selectbox(
            "Select State",
            sorted(df["State"].dropna().unique())
        )

        values["Crop_Type"] = st.selectbox(
            "Select Crop",
            sorted(df["Crop_Type"].dropna().unique())
        )

        values["Season"] = st.selectbox(
            "Select Season",
            sorted(df["Season"].dropna().unique())
        )

        values["Soil_Type"] = st.selectbox(
            "Select Soil Type",
            sorted(df["Soil_Type"].dropna().unique())
        )

        values["Irrigation_Type"] = st.selectbox(
            "Irrigation Type",
            sorted(df["Irrigation_Type"].dropna().unique())
        )

    with col2:

        values["Rainfall_mm"] = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=800.0
        )

        values["Avg_Temperature_C"] = st.number_input(
            "Temperature (°C)",
            value=25.0
        )

        values["Humidity_percent"] = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0
        )

        values["Fertilizer_Used_kg"] = st.number_input(
            "Fertilizer Used (kg)",
            min_value=0.0,
            value=200.0
        )

        values["Pesticide_Used_kg"] = st.number_input(
            "Pesticide Used (kg)",
            min_value=0.0,
            value=10.0
        )

        values["Cultivated_Area_hectares"] = st.number_input(
            "Cultivated Area (hectares)",
            min_value=0.01,
            value=2.0
        )

        if include_yield:

            values["Yield_ton_per_hectare"] = st.number_input(
                "Expected Crop Yield (tons/hectare)",
                min_value=0.0,
                value=4.8
            )

    return values


# ==================================================
# 7. DASHBOARD OVERVIEW
# ==================================================

if page == "Dashboard Overview":

    st.header("📊 Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        f"{len(df):,}"
    )

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
        f"{df['Year'].min():.0f}–{df['Year'].max():.0f}"
    )

    st.markdown("---")

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

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ==================================================
# 8. CROP ANALYSIS
# ==================================================

elif page == "Crop Analysis":

    st.header("🌱 Crop Performance Analysis")

    crop_options = [
        "All Crops"
    ] + sorted(df["Crop_Type"].dropna().unique().tolist())

    selected_crop = st.selectbox(
        "Select Crop",
        crop_options
    )

    filtered_df = df.copy()

    if selected_crop != "All Crops":
        filtered_df = filtered_df[
            filtered_df["Crop_Type"] == selected_crop
        ]

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("State-wise Average Yield")

        state_yield = filtered_df.groupby("State")[
            "Yield_ton_per_hectare"
        ].mean().sort_values(ascending=False)

        st.bar_chart(state_yield)

    with col2:

        st.subheader("Season-wise Average Yield")

        season_yield = filtered_df.groupby("Season")[
            "Yield_ton_per_hectare"
        ].mean()

        st.bar_chart(season_yield)

    st.markdown("---")

    st.subheader("Crop Records")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )


# ==================================================
# 9. CROP YIELD PREDICTION
# ==================================================

elif page == "Crop Yield Prediction":

    st.header("🌾 Crop Yield Prediction")

    st.write(
        "Predict crop yield using weather conditions "
        "and farming information."
    )

    model, features = train_model(
        "Yield_ton_per_hectare"
    )

    with st.form("yield_prediction_form"):

        inputs = prediction_inputs()

        submitted = st.form_submit_button(
            "Predict Crop Yield"
        )

    if submitted:

        input_df = pd.DataFrame([inputs])[features]

        prediction = model.predict(input_df)[0]

        st.markdown("---")

        st.subheader("Prediction Result")

        st.success(
            f"Estimated Crop Yield: {prediction:.2f} tons/hectare"
        )


# ==================================================
# 10. FARMER PROFITABILITY PREDICTION
# ==================================================

elif page == "Farmer Profitability Prediction":

    st.header("💰 Farmer Profitability Prediction")

    st.write(
        "Estimate farmer profitability using crop, "
        "weather and farming conditions."
    )

    model, features = train_model(
        "Farmer_Profit_INR"
    )

    with st.form("profit_prediction_form"):

        inputs = prediction_inputs(
            include_yield=True
        )

        submitted = st.form_submit_button(
            "Predict Farmer Profit"
        )

    if submitted:

        input_df = pd.DataFrame([inputs])[features]

        prediction = model.predict(input_df)[0]

        st.markdown("---")

        st.subheader("Prediction Result")

        st.success(
            f"Estimated Farmer Profit: ₹{prediction:,.2f}"
        )


# ==================================================
# 11. FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "AgriSmart | Crop Yield Forecasting "
    "and Farmer Profitability Prediction"
)
