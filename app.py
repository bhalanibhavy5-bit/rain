
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
# 2. PROFESSIONAL UI STYLING
# =====================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F5F8F5;
}

.block-container {
    max-width: 1500px;
    padding: 2rem 2.5rem 3rem 2.5rem;
}

/* Main headings */
h1 {
    color: #17452D !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
}

h2 {
    color: #205B38 !important;
    font-size: 1.65rem !important;
    font-weight: 750 !important;
}

h3 {
    color: #285C3C !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
}

p {
    color: #35483A;
    line-height: 1.6;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #173F2C;
    min-width: 270px;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] hr {
    border-color: #65836E !important;
}

/* Sidebar radio buttons */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    font-size: 1rem !important;
    font-weight: 600 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DCE7DD;
    border-top: 4px solid #328451;
    border-radius: 12px;
    padding: 22px 20px;
    min-height: 125px;
    box-shadow: 0 2px 8px rgba(25,65,40,0.05);
}

[data-testid="stMetricLabel"] {
    color: #4B6251 !important;
    font-size: 0.95rem !important;
    font-weight: 650 !important;
}

[data-testid="stMetricValue"] {
    color: #17452D !important;
    font-size: 1.65rem !important;
    font-weight: 800 !important;
}

/* Prediction forms */
[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #DCE7DD;
    border-radius: 12px;
    padding: 28px;
    box-shadow: 0 2px 8px rgba(25,65,40,0.04);
}

/* Input labels */
.stSelectbox label,
.stNumberInput label {
    color: #254B31 !important;
    font-size: 0.98rem !important;
    font-weight: 650 !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background-color: #246B42;
    color: #FFFFFF !important;
    border: 1px solid #246B42;
    border-radius: 8px;
    min-height: 48px;
    font-size: 1rem;
    font-weight: 700;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #17452D;
    border-color: #17452D;
}

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #DCE7DD;
    border-radius: 9px;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 9px;
}

/* Dividers */
hr {
    border-color: #DCE5DD !important;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
}

/* Responsive */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }

    [data-testid="stForm"] {
        padding: 15px;
    }

    h1 {
        font-size: 1.7rem !important;
    }
}

/* Footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. DATASET LOADING
# =====================================================

@st.cache_data
def load_data():

    folder = Path(__file__).resolve().parent
    csv_files = list(folder.rglob("*.csv"))

    if not csv_files:
        return None, "No CSV dataset found."

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

    return None, "No matching crop yield dataset found."


df, dataset_name = load_data()

if df is None:
    st.error(dataset_name)
    st.info(
        "Upload your crop yield CSV file to the GitHub "
        "repository containing app.py."
    )
    st.stop()


# =====================================================
# 4. DATA PREPARATION
# =====================================================

df = df.dropna(
    subset=[
        "Yield_ton_per_hectare",
        "Farmer_Profit_INR"
    ]
).copy()

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
# 5. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🌾 AgriSmart")

st.sidebar.caption("SMART AGRICULTURE SYSTEM")

st.sidebar.divider()

st.sidebar.subheader("Navigation")

page = st.sidebar.radio(
    "Select Dashboard",
    [
        "Dashboard Overview",
        "Crop Analysis",
        "Crop Yield Forecasting",
        "Farmer Profitability Prediction"
    ],
    index=0
)

st.sidebar.divider()

st.sidebar.subheader("Dataset Information")

st.sidebar.success("Dataset Loaded Successfully")

st.sidebar.caption(f"File: {dataset_name}")

st.sidebar.caption(
    f"Total Records: {len(df):,}"
)

st.sidebar.divider()

st.sidebar.caption(
    "AgriSmart | Smart Agriculture"
)


# =====================================================
# 6. MAIN HEADER - NATIVE STREAMLIT
# =====================================================

st.title("🌾 AgriSmart")

st.subheader(
    "Smart Agriculture Intelligence Dashboard"
)

st.caption(
    "Crop Yield Forecasting | Farmer Profitability Prediction"
)

st.divider()


# =====================================================
# 7. MACHINE LEARNING MODEL
# =====================================================

@st.cache_resource
def train_model(target):

    features = categorical_features + numeric_features

    if target == "Farmer_Profit_INR":
        features.append("Yield_ton_per_hectare")

    X = df[features].copy()
    y = pd.to_numeric(df[target], errors="coerce")

    for col in numeric_features:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

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
# 8. PREDICTION INPUT FUNCTION
# =====================================================

def get_prediction_inputs(include_yield=False):

    values = {}

    st.markdown("### Crop and Farming Information")

    st.write(
        "Enter the following details to generate your prediction."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Crop Details")

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

        st.markdown("#### Weather and Farming Inputs")

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
# 9. DASHBOARD OVERVIEW
# =====================================================

if page == "Dashboard Overview":

    st.header("Dashboard Overview")

    st.write(
        "Summary of agricultural records, crop production "
        "and farmer profitability."
    )

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
        f"{int(df['Year'].min())}–{int(df['Year'].max())}"
    )

    st.divider()

    st.subheader("Crop-wise Average Yield")

    crop_yield = df.groupby("Crop_Type")[
        "Yield_ton_per_hectare"
    ].mean().sort_values(ascending=False)

    st.bar_chart(crop_yield, height=350)

    st.divider()

    st.subheader("Year-wise Crop Yield")

    yearly_yield = df.groupby("Year")[
        "Yield_ton_per_hectare"
    ].mean()

    st.line_chart(yearly_yield, height=350)

    st.divider()

    st.subheader("State-wise Average Farmer Profit")

    state_profit = df.groupby("State")[
        "Farmer_Profit_INR"
    ].mean().sort_values(ascending=False)

    st.bar_chart(state_profit, height=350)

    st.divider()

    st.subheader("Season-wise Average Crop Yield")

    season_yield = df.groupby("Season")[
        "Yield_ton_per_hectare"
    ].mean()

    st.bar_chart(season_yield, height=300)

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        height=350
    )


# =====================================================
# 10. CROP ANALYSIS
# =====================================================

elif page == "Crop Analysis":

    st.header("Crop Performance Analysis")

    st.write(
        "Analyse crop performance across states and seasons."
    )

    crop_options = [
        "All Crops"
    ] + sorted(
        df["Crop_Type"].dropna().unique().tolist()
    )

    selected_crop = st.selectbox(
        "Select Crop",
        crop_options,
        key="analysis_crop"
    )

    filtered = df.copy()

    if selected_crop != "All Crops":

        filtered = filtered[
            filtered["Crop_Type"] == selected_crop
        ]

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("State-wise Average Yield")

        state_yield = filtered.groupby("State")[
            "Yield_ton_per_hectare"
        ].mean().sort_values(ascending=False)

        st.bar_chart(state_yield, height=350)

    with col2:

        st.subheader("Season-wise Average Yield")

        season_yield = filtered.groupby("Season")[
            "Yield_ton_per_hectare"
        ].mean()

        st.bar_chart(season_yield, height=350)

    st.divider()

    st.subheader("Filtered Crop Records")

    st.dataframe(
        filtered,
        use_container_width=True,
        height=400
    )


# =====================================================
# 11. CROP YIELD FORECASTING
# =====================================================

elif page == "Crop Yield Forecasting":

    st.header("🌱 Crop Yield Forecasting")

    st.write(
        "Estimate crop yield using weather conditions "
        "and farming information."
    )

    st.info(
        "Complete the details below and click Forecast Crop Yield."
    )

    with st.form("yield_forecast_form"):

        inputs = get_prediction_inputs(
            include_yield=False
        )

        st.divider()

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

        st.divider()

        st.subheader("Crop Yield Forecast Result")

        st.metric(
            "Predicted Crop Yield",
            f"{prediction:.2f} tons/hectare"
        )

        st.success(
            "Crop yield forecast generated successfully."
        )


# =====================================================
# 12. FARMER PROFITABILITY PREDICTION
# =====================================================

elif page == "Farmer Profitability Prediction":

    st.header("💰 Farmer Profitability Prediction")

    st.write(
        "Estimate farmer profit using crop, weather, "
        "farming conditions and expected yield."
    )

    st.info(
        "Enter the expected yield and farming details "
        "to estimate profitability."
    )

    with st.form("profit_forecast_form"):

        inputs = get_prediction_inputs(
            include_yield=True
        )

        st.divider()

        submitted = st.form_submit_button(
            "Predict Farmer Profit",
            use_container_width=True
        )

    if submitted:

        with st.spinner("Calculating farmer profitability..."):

            model, features = train_model(
                "Farmer_Profit_INR"
            )

            input_data = pd.DataFrame(
                [inputs],
                columns=features
            )

            prediction = model.predict(input_data)[0]

        st.divider()

        st.subheader("Farmer Profitability Result")

        st.metric(
            "Estimated Farmer Profit",
            f"₹{prediction:,.2f}"
        )

        st.success(
            "Farmer profitability prediction generated successfully."
        )


# =====================================================
# 13. FOOTER
# =====================================================

st.divider()

st.markdown(
    "🌾 **AgriSmart | Smart Agriculture Dashboard**"
)

st.caption(
    "Crop Yield Forecasting and Farmer Profitability Prediction"
)
