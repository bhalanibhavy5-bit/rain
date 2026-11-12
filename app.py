
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AgriSmart",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 AgriSmart")
st.markdown(
    "### Smart Agriculture Dashboard"
)
st.caption(
    "Crop Yield Forecasting | Farmer Profitability Prediction"
)

st.markdown("---")


# =====================================================
# 2. AUTOMATIC DATASET DETECTION
# =====================================================

@st.cache_data
def load_data():

    base_folder = Path(__file__).resolve().parent

    # Search project folder and its subfolders
    csv_files = list(base_folder.rglob("*.csv"))

    if not csv_files:
        st.error("No CSV dataset found in the project.")
        st.info(
            "Upload your crop yield CSV file to GitHub "
            "in the same folder as app.py or inside a data folder."
        )
        st.stop()

    # Identify the required dataset by filename
    matching_files = [
        file for file in csv_files
        if (
            "rainfall" in file.name.lower()
            or "crop" in file.name.lower()
            or "yield" in file.name.lower()
        )
    ]

    if len(matching_files) == 1:
        selected_file = matching_files[0]

    elif len(matching_files) > 1:
        preferred = [
            file for file in matching_files
            if "rainfall" in file.name.lower()
            and "crop" in file.name.lower()
            and "yield" in file.name.lower()
        ]

        if len(preferred) == 1:
            selected_file = preferred[0]
        else:
            st.error("Multiple possible crop datasets were found.")
            selected_file = st.selectbox(
                "Select your dataset",
                matching_files,
                format_func=lambda x: str(x.relative_to(base_folder))
            )

    elif len(csv_files) == 1:
        selected_file = csv_files[0]

    else:
        st.error("Unable to identify the crop yield dataset.")
        st.write("Available CSV files:")
        for file in csv_files:
            st.write(str(file.relative_to(base_folder)))
        st.stop()

    data = pd.read_csv(selected_file)
    return data, str(selected_file.name)


df, dataset_name = load_data()

df.columns = df.columns.str.strip()

# Remove completely empty rows
df = df.dropna(how="all")

st.sidebar.success(f"Dataset loaded: {dataset_name}")


# =====================================================
# 3. CHECK REQUIRED COLUMNS
# =====================================================

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
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        "The dataset is missing these columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


# =====================================================
# 4. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🌾 AgriSmart")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Crop Analysis",
        "Crop Yield Prediction",
        "Farmer Profitability Prediction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Smart Agriculture System")


# =====================================================
# 5. MACHINE LEARNING FEATURES
# =====================================================

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


# =====================================================
# 6. MACHINE LEARNING MODEL
# =====================================================

@st.cache_resource
def train_model(target):

    features = categorical_features + numeric_features

    if target == "Farmer_Profit_INR":
        features.append("Yield_ton_per_hectare")

    X = df[features].copy()
    y = pd.to_numeric(df[target], errors="coerce")

    # Remove rows with missing target values
    valid_rows = y.notna()

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    # Ensure numerical columns are numeric
    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if target == "Farmer_Profit_INR":
        X["Yield_ton_per_hectare"] = pd.to_numeric(
            X["Yield_ton_per_hectare"],
            errors="coerce"
        )

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


# =====================================================
# 7. PREDICTION INPUT FUNCTION
# =====================================================

def prediction_inputs(include_yield=False):

    values = {}

    st.subheader("Farming Information")

    col1, col2 = st.columns(2)

    with col1:

        values["State"] = st.selectbox(
            "State",
            sorted(df["State"].dropna().unique())
        )

        values["Crop_Type"] = st.selectbox(
            "Crop",
            sorted(df["Crop_Type"].dropna().unique())
        )

        values["Season"] = st.selectbox(
            "Season",
            sorted(df["Season"].dropna().unique())
        )

        values["Soil_Type"] = st.selectbox(
            "Soil Type",
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
            "Fertilizer (kg)",
            min_value=0.0,
            value=200.0
        )

        values["Pesticide_Used_kg"] = st.number_input(
            "Pesticide (kg)",
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
                "Expected Yield (tons/hectare)",
                min_value=0.0,
                value=4.8
            )

    return values


# =====================================================
# 8. DASHBOARD OVERVIEW
# =====================================================

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
        f"{df['Year'].min():.0f} - {df['Year'].max():.0f}"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Crop-wise Average Yield")

        crop_yield = df.groupby("Crop_Type")[
            "Yield_ton_per_hectare"
        ].mean().sort_values(ascending=False)

        st.bar_chart(crop_yield)

    with col2:

        st.subheader("Year-wise Crop Yield")

        yearly_yield = df.groupby("Year")[
            "Yield_ton_per_hectare"
        ].mean()

        st.line_chart(yearly_yield)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("State-wise Farmer Profit")

        state_profit = df.groupby("State")[
            "Farmer_Profit_INR"
        ].mean().sort_values(ascending=False)

        st.bar_chart(state_profit)

    with col2:

        st.subheader("Season-wise Crop Yield")

        season_yield = df.groupby("Season")[
            "Yield_ton_per_hectare"
        ].mean()

        st.bar_chart(season_yield)

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# =====================================================
# 9. CROP ANALYSIS
# =====================================================

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

    st.subheader("Crop Data")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )


# =====================================================
# 10. CROP YIELD PREDICTION
# =====================================================

elif page == "Crop Yield Prediction":

    st.header("🌾 Crop Yield Prediction")

    st.write(
        "Estimate crop yield using weather conditions "
        "and farming information."
    )

    model, features = train_model(
        "Yield_ton_per_hectare"
    )

    with st.form("yield_form"):

        inputs = prediction_inputs()

        submitted = st.form_submit_button(
            "Predict Crop Yield",
            use_container_width=True
        )

    if submitted:

        input_df = pd.DataFrame([inputs])[features]

        prediction = model.predict(input_df)[0]

        st.markdown("---")
        st.subheader("Prediction Result")

        st.success(
            f"Estimated Crop Yield: {prediction:.2f} tons/hectare"
        )


# =====================================================
# 11. FARMER PROFITABILITY PREDICTION
# =====================================================

elif page == "Farmer Profitability Prediction":

    st.header("💰 Farmer Profitability Prediction")

    st.write(
        "Estimate farmer profitability using crop, "
        "weather and farming conditions."
    )

    model, features = train_model(
        "Farmer_Profit_INR"
    )

    with st.form("profit_form"):

        inputs = prediction_inputs(
            include_yield=True
        )

        submitted = st.form_submit_button(
            "Predict Farmer Profit",
            use_container_width=True
        )

    if submitted:

        input_df = pd.DataFrame([inputs])[features]

        prediction = model.predict(input_df)[0]

        st.markdown("---")
        st.subheader("Prediction Result")

        st.success(
            f"Estimated Farmer Profit: ₹{prediction:,.2f}"
        )


# =====================================================
# 12. FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "AgriSmart | Smart Agriculture Dashboard | "
    "Crop Yield and Farmer Profitability"
)
