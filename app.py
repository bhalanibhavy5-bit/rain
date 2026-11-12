
import streamlit as st
import pandas as pd
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
    page_title="AgriSmart Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. DATASET LOADING
# =====================================================

@st.cache_data
def load_data():

    folder = Path(__file__).resolve().parent

    csv_files = list(folder.rglob("*.csv"))

    if not csv_files:
        return None, "No CSV file found in the project folder."

    required = {
        "State", "Season", "Crop_Type", "Soil_Type",
        "Irrigation_Type", "Rainfall_mm",
        "Avg_Temperature_C", "Humidity_percent",
        "Fertilizer_Used_kg", "Pesticide_Used_kg",
        "Cultivated_Area_hectares",
        "Yield_ton_per_hectare", "Farmer_Profit_INR",
        "Year"
    }

    for file in csv_files:

        try:
            data = pd.read_csv(file)
            data.columns = data.columns.str.strip()

            if required.issubset(set(data.columns)):
                return data, file.name

        except Exception:
            continue

    return None, "No matching crop yield dataset was found."


df, dataset_name = load_data()

if df is None:
    st.error(dataset_name)
    st.info(
        "Upload your crop yield CSV file to the same GitHub "
        "repository as app.py, or into a subfolder."
    )
    st.stop()


# =====================================================
# 3. DATA PREPARATION
# =====================================================

df = df.dropna(subset=[
    "Yield_ton_per_hectare",
    "Farmer_Profit_INR"
]).copy()

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
# 4. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🌾 AgriSmart")
st.sidebar.write("Smart Agriculture Dashboard")

st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Choose a Dashboard Page",
    [
        "Dashboard Overview",
        "Crop Analysis",
        "Crop Yield Forecasting",
        "Farmer Profitability Prediction"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.success("Dataset successfully loaded")
st.sidebar.caption(dataset_name)


# =====================================================
# 5. APPLICATION HEADER
# =====================================================

st.title("🌾 AgriSmart")
st.caption(
    "Smart Crop Yield Forecasting and Farmer Profitability System"
)

st.markdown("---")


# =====================================================
# 6. MACHINE LEARNING MODEL
# =====================================================

def train_model(target):

    features = categorical_features + numeric_features

    if target == "Farmer_Profit_INR":
        features.append("Yield_ton_per_hectare")

    X = df[features].copy()
    y = pd.to_numeric(df[target], errors="coerce")

    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if target == "Farmer_Profit_INR":
        X["Yield_ton_per_hectare"] = pd.to_numeric(
            X["Yield_ton_per_hectare"],
            errors="coerce"
        )

    valid = y.notna()

    X = X.loc[valid]
    y = y.loc[valid]

    categorical = [
        col for col in features
        if col in categorical_features
    ]

    numerical = [
        col for col in features
        if col not in categorical_features
    ]

    preprocessing = ColumnTransformer([
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical
        ),
        (
            "numerical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numerical
        )
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        (
            "model",
            RandomForestRegressor(
                n_estimators=150,
                random_state=42,
                min_samples_leaf=2
            )
        )
    ])

    model.fit(X, y)

    return model, features


# =====================================================
# 7. INPUT FORM FOR PREDICTIONS
# =====================================================

def get_prediction_inputs(include_yield=False):

    values = {}

    st.subheader("Crop and Farming Details")

    col1, col2 = st.columns(2)

    with col1:

        values["State"] = st.selectbox(
            "Select State",
            sorted(df["State"].dropna().unique()),
            key="state_input"
        )

        values["Crop_Type"] = st.selectbox(
            "Select Crop",
            sorted(df["Crop_Type"].dropna().unique()),
            key="crop_input"
        )

        values["Season"] = st.selectbox(
            "Select Season",
            sorted(df["Season"].dropna().unique()),
            key="season_input"
        )

        values["Soil_Type"] = st.selectbox(
            "Select Soil Type",
            sorted(df["Soil_Type"].dropna().unique()),
            key="soil_input"
        )

        values["Irrigation_Type"] = st.selectbox(
            "Select Irrigation Type",
            sorted(df["Irrigation_Type"].dropna().unique()),
            key="irrigation_input"
        )

    with col2:

        values["Rainfall_mm"] = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=800.0,
            key="rainfall_input"
        )

        values["Avg_Temperature_C"] = st.number_input(
            "Average Temperature (°C)",
            value=25.0,
            key="temperature_input"
        )

        values["Humidity_percent"] = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            key="humidity_input"
        )

        values["Fertilizer_Used_kg"] = st.number_input(
            "Fertilizer Used (kg)",
            min_value=0.0,
            value=200.0,
            key="fertilizer_input"
        )

        values["Pesticide_Used_kg"] = st.number_input(
            "Pesticide Used (kg)",
            min_value=0.0,
            value=10.0,
            key="pesticide_input"
        )

        values["Cultivated_Area_hectares"] = st.number_input(
            "Cultivated Area (hectares)",
            min_value=0.01,
            value=2.0,
            key="area_input"
        )

        if include_yield:

            values["Yield_ton_per_hectare"] = st.number_input(
                "Expected Yield (tons/hectare)",
                min_value=0.0,
                value=4.8,
                key="expected_yield_input"
            )

    return values


# =====================================================
# 8. DASHBOARD OVERVIEW
# =====================================================

if page == "Dashboard Overview":

    st.header("📊 Dashboard Overview")

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
        f"{int(df['Year'].min())}–{int(df['Year'].max())}"
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
        st.subheader("Year-wise Average Yield")

        yearly_yield = df.groupby("Year")[
            "Yield_ton_per_hectare"
        ].mean()

        st.line_chart(yearly_yield)

    left, right = st.columns(2)

    with left:
        st.subheader("State-wise Average Profit")

        state_profit = df.groupby("State")[
            "Farmer_Profit_INR"
        ].mean().sort_values(ascending=False)

        st.bar_chart(state_profit)

    with right:
        st.subheader("Season-wise Average Yield")

        season_yield = df.groupby("Season")[
            "Yield_ton_per_hectare"
        ].mean()

        st.bar_chart(season_yield)

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)


# =====================================================
# 9. CROP ANALYSIS
# =====================================================

elif page == "Crop Analysis":

    st.header("🌱 Crop Analysis")

    crop_options = [
        "All Crops"
    ] + sorted(df["Crop_Type"].dropna().unique().tolist())

    selected_crop = st.selectbox(
        "Choose Crop",
        crop_options,
        key="analysis_crop"
    )

    filtered = df.copy()

    if selected_crop != "All Crops":
        filtered = filtered[
            filtered["Crop_Type"] == selected_crop
        ]

    left, right = st.columns(2)

    with left:
        st.subheader("State-wise Crop Yield")

        state_yield = filtered.groupby("State")[
            "Yield_ton_per_hectare"
        ].mean().sort_values(ascending=False)

        st.bar_chart(state_yield)

    with right:
        st.subheader("Season-wise Crop Yield")

        season_yield = filtered.groupby("Season")[
            "Yield_ton_per_hectare"
        ].mean()

        st.bar_chart(season_yield)

    st.subheader("Filtered Crop Records")

    st.dataframe(filtered, use_container_width=True)


# =====================================================
# 10. CROP YIELD FORECASTING
# =====================================================

elif page == "Crop Yield Forecasting":

    st.header("🌾 Crop Yield Forecasting")

    st.write(
        "Enter the farming and weather details to estimate "
        "crop yield in tons per hectare."
    )

    with st.form("yield_forecast_form"):

        inputs = get_prediction_inputs(
            include_yield=False
        )

        submitted = st.form_submit_button(
            "Forecast Crop Yield",
            use_container_width=True
        )

    if submitted:

        with st.spinner("Calculating crop yield..."):

            model, features = train_model(
                "Yield_ton_per_hectare"
            )

            input_data = pd.DataFrame(
                [inputs],
                columns=features
            )

            prediction = model.predict(input_data)[0]

        st.markdown("---")
        st.subheader("Crop Yield Forecast Result")

        st.metric(
            "Predicted Crop Yield",
            f"{prediction:.2f} tons/hectare"
        )

        st.success("Crop yield forecast generated successfully.")


# =====================================================
# 11. FARMER PROFITABILITY PREDICTION
# =====================================================

elif page == "Farmer Profitability Prediction":

    st.header("💰 Farmer Profitability Prediction")

    st.write(
        "Enter crop, farming, weather and expected yield "
        "details to estimate farmer profit."
    )

    with st.form("profit_forecast_form"):

        inputs = get_prediction_inputs(
            include_yield=True
        )

        submitted = st.form_submit_button(
            "Predict Farmer Profit",
            use_container_width=True
        )

    if submitted:

        with st.spinner("Calculating estimated farmer profit..."):

            model, features = train_model(
                "Farmer_Profit_INR"
            )

            input_data = pd.DataFrame(
                [inputs],
                columns=features
            )

            prediction = model.predict(input_data)[0]

        st.markdown("---")
        st.subheader("Farmer Profitability Result")

        st.metric(
            "Estimated Farmer Profit",
            f"₹{prediction:,.2f}"
        )

        st.success("Farmer profitability prediction generated successfully.")


# =====================================================
# 12. FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "AgriSmart | Smart Agriculture Dashboard | "
    "Crop Yield Forecasting and Farmer Profitability"
)
