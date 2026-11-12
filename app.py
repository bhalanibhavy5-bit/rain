
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
    page_title="AgriSmart | Smart Agriculture",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. PROFESSIONAL UI DESIGN
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #F5F8F5;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #153D2B, #245B3A);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] hr {
    border-color: #52745D !important;
}

/* Main headings */
h1 {
    color: #17452D !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #245B3A !important;
    font-weight: 700 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border: 1px solid #E1EBE2;
    border-left: 5px solid #328451;
    border-radius: 13px;
    box-shadow: 0 3px 12px rgba(30, 70, 40, 0.06);
}

[data-testid="stMetricLabel"] {
    color: #54705C !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #17452D !important;
    font-weight: 800;
}

/* Forms */
[data-testid="stForm"] {
    background: white;
    padding: 25px;
    border: 1px solid #E1EBE2;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(30, 70, 40, 0.05);
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #1C603A, #328451);
    color: white !important;
    border: none;
    border-radius: 9px;
    padding: 0.65rem 1.2rem;
    font-weight: 700;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #17452D;
    border: none;
}

/* Input fields */
.stNumberInput input,
.stTextInput input {
    border-radius: 8px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #E1EBE2;
    border-radius: 10px;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Divider */
hr {
    border-color: #DCE8DE !important;
}

/* Footer */
footer {
    visibility: hidden;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }

    [data-testid="stForm"] {
        padding: 14px;
    }
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
        "repository as app.py or into a subfolder."
    )
    st.stop()


# =====================================================
# 4. DATA PREPARATION
# =====================================================

df = df.dropna(
    subset=["Yield_ton_per_hectare", "Farmer_Profit_INR"]
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

st.sidebar.markdown("""
<div style="text-align:center; padding:15px 5px 20px;">
    <div style="font-size:45px;">🌾</div>
    <h2 style="color:white !important; margin:0;">
        AgriSmart
    </h2>
    <p style="color:#D8EBDD; font-size:12px;">
        SMART AGRICULTURE SYSTEM
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard Overview",
        "Crop Analysis",
        "Crop Yield Forecasting",
        "Farmer Profitability Prediction"
    ],
    index=0
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    f"""
    <div style="
        background:rgba(255,255,255,0.10);
        padding:12px;
        border-radius:10px;
        font-size:12px;
    ">
        <b>Dataset Status</b><br>
        <span style="color:#B9E7C5;">● Successfully Loaded</span>
        <br><br>
        <b>Dataset:</b><br>{dataset_name}
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("""
<div style="position:relative; padding-top:30px;
color:#D8EBDD; font-size:11px; text-align:center;">
AgriSmart | Smart Farming
</div>
""", unsafe_allow_html=True)


# =====================================================
# 6. MAIN HEADER
# =====================================================

st.markdown("""
<div style="
    background:linear-gradient(120deg,#153D2B,#287347);
    padding:28px 32px;
    border-radius:16px;
    margin-bottom:25px;
">
    <h1 style="
        color:white !important;
        font-size:34px;
        margin:0;
    ">
        🌾 AgriSmart
    </h1>
    <p style="
        color:#D8EBDD;
        font-size:16px;
        margin:8px 0 0 0;
    ">
        Smart Agriculture Intelligence Dashboard
    </p>
    <p style="
        color:#D8EBDD;
        font-size:13px;
        margin:8px 0 0 0;
    ">
        Crop Yield Forecasting | Farmer Profitability Prediction
    </p>
</div>
""", unsafe_allow_html=True)


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
# 8. PREDICTION INPUT FUNCTION
# =====================================================

def get_prediction_inputs(include_yield=False):

    values = {}

    st.markdown("#### 🌱 Crop and Farming Information")

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
# 9. DASHBOARD OVERVIEW
# =====================================================

if page == "Dashboard Overview":

    st.header("📊 Dashboard Overview")
    st.caption("Agricultural data summary and performance indicators")

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

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("🌾 Crop-wise Average Yield")

        crop_yield = df.groupby("Crop_Type")[
            "Yield_ton_per_hectare"
        ].mean().sort_values(ascending=False)

        st.bar_chart(crop_yield)

    with right:
        st.subheader("📈 Year-wise Average Yield")

        yearly_yield = df.groupby("Year")[
            "Yield_ton_per_hectare"
        ].mean()

        st.line_chart(yearly_yield)

    left, right = st.columns(2)

    with left:
        st.subheader("💰 State-wise Average Profit")

        state_profit = df.groupby("State")[
            "Farmer_Profit_INR"
        ].mean().sort_values(ascending=False)

        st.bar_chart(state_profit)

    with right:
        st.subheader("🌦️ Season-wise Average Yield")

        season_yield = df.groupby("Season")[
            "Yield_ton_per_hectare"
        ].mean()

        st.bar_chart(season_yield)

    st.markdown("---")

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# =====================================================
# 10. CROP ANALYSIS
# =====================================================

elif page == "Crop Analysis":

    st.header("🌱 Crop Performance Analysis")
    st.caption("Explore crop performance across states and seasons")

    crop_options = [
        "All Crops"
    ] + sorted(df["Crop_Type"].dropna().unique().tolist())

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

    left, right = st.columns(2)

    with left:
        st.subheader("State-wise Average Yield")

        state_yield = filtered.groupby("State")[
            "Yield_ton_per_hectare"
        ].mean().sort_values(ascending=False)

        st.bar_chart(state_yield)

    with right:
        st.subheader("Season-wise Average Yield")

        season_yield = filtered.groupby("Season")[
            "Yield_ton_per_hectare"
        ].mean()

        st.bar_chart(season_yield)

    st.markdown("---")

    st.subheader("Crop Records")

    st.dataframe(
        filtered,
        use_container_width=True
    )


# =====================================================
# 11. CROP YIELD FORECASTING
# =====================================================

elif page == "Crop Yield Forecasting":

    st.header("🌾 Crop Yield Forecasting")

    st.caption(
        "Estimate agricultural yield using weather and farming conditions."
    )

    with st.container(border=True):

        with st.form("yield_forecast_form"):

            inputs = get_prediction_inputs(
                include_yield=False
            )

            submitted = st.form_submit_button(
                "🌱 Forecast Crop Yield",
                use_container_width=True
            )

    if submitted:

        with st.spinner("Analysing farming conditions..."):

            model, features = train_model(
                "Yield_ton_per_hectare"
            )

            input_data = pd.DataFrame(
                [inputs],
                columns=features
            )

            prediction = model.predict(input_data)[0]

        st.markdown("---")

        st.subheader("📊 Crop Yield Forecast Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Crop Yield",
                f"{prediction:.2f} tons/hectare"
            )

        with col2:
            st.metric(
                "Selected Crop",
                inputs["Crop_Type"]
            )

        st.success(
            "Crop yield forecast generated successfully."
        )


# =====================================================
# 12. FARMER PROFITABILITY PREDICTION
# =====================================================

elif page == "Farmer Profitability Prediction":

    st.header("💰 Farmer Profitability Prediction")

    st.caption(
        "Estimate potential farmer profit using crop and farming information."
    )

    with st.container(border=True):

        with st.form("profit_forecast_form"):

            inputs = get_prediction_inputs(
                include_yield=True
            )

            submitted = st.form_submit_button(
                "💰 Predict Farmer Profit",
                use_container_width=True
            )

    if submitted:

        with st.spinner("Analysing farmer profitability..."):

            model, features = train_model(
                "Farmer_Profit_INR"
            )

            input_data = pd.DataFrame(
                [inputs],
                columns=features
            )

            prediction = model.predict(input_data)[0]

        st.markdown("---")

        st.subheader("📊 Farmer Profitability Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Estimated Farmer Profit",
                f"₹{prediction:,.2f}"
            )

        with col2:
            st.metric(
                "Selected Crop",
                inputs["Crop_Type"]
            )

        st.success(
            "Farmer profitability prediction generated successfully."
        )


# =====================================================
# 13. FOOTER
# =====================================================

st.markdown("---")

st.markdown("""
<div style="
    background:#153D2B;
    padding:18px;
    border-radius:12px;
    text-align:center;
">
    <p style="
        color:white;
        font-weight:700;
        margin:0;
    ">
        🌾 AgriSmart | Smart Agriculture Dashboard
    </p>
    <p style="
        color:#D8EBDD;
        font-size:12px;
        margin:6px 0 0 0;
    ">
        Crop Yield Forecasting and Farmer Profitability Prediction
    </p>
</div>
""", unsafe_allow_html=True)
