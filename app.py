%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AgriSmart - Agricultural Forecasting",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 AgriSmart")
st.subheader("Crop Yield Forecasting & Farmer Profitability Prediction")

st.write(
    "An AI-powered agricultural forecasting system "
    "using Machine Learning."
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Agricultural Dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload your agricultural CSV dataset.")
    st.stop()

df = pd.read_csv(uploaded_file)

# ==========================================
# 2. DATA VALIDATION
# ==========================================

YIELD = "Yield_ton_per_hectare"
PROFIT = "Farmer_Profit_INR"

required_columns = [
    "Record_ID", "State", "District", "Year",
    "Season", "Crop_Type", "Rainfall_mm",
    "Avg_Temperature_C", "Humidity_percent",
    "Soil_Type", "Irrigation_Type",
    "Fertilizer_Used_kg", "Pesticide_Used_kg",
    "Cultivated_Area_hectares",
    YIELD, "Market_Price_per_ton", PROFIT
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    st.error("Missing columns: " + ", ".join(missing))
    st.stop()

df = df.dropna(
    subset=[YIELD, PROFIT, "Year"]
).copy()

if len(df) < 30:
    st.error("Insufficient data for model training.")
    st.stop()

# ==========================================
# 3. DATASET OVERVIEW
# ==========================================

st.header("1. Agricultural Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", f"{len(df):,}")
c2.metric("Total Features", len(df.columns))
c3.metric(
    "Missing Values",
    f"{df.isnull().sum().sum():,}"
)
c4.metric(
    "Year Range",
    f"{int(df.Year.min())}-{int(df.Year.max())}"
)

with st.expander("View Dataset"):
    st.dataframe(df.head(30), use_container_width=True)

with st.expander("Statistical Summary"):
    st.dataframe(df.describe(include="all"))

# ==========================================
# 4. FEATURE SELECTION
# ==========================================

# Remove ID and both target variables
base_features = [
    col for col in df.columns
    if col not in ["Record_ID", YIELD, PROFIT]
]

# Crop yield predictors
yield_features = base_features.copy()

# Profit predictors include expected/observed yield
profit_features = base_features + [YIELD]

# ==========================================
# 5. TRAIN TEST SPLIT
# ==========================================

if df["Year"].nunique() >= 2:

    latest_year = df["Year"].max()

    train = df[
        df["Year"] < latest_year
    ].copy()

    test = df[
        df["Year"] == latest_year
    ].copy()

    st.info(
        f"Training: Years before {latest_year} | "
        f"Testing: {latest_year}"
    )

else:

    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        df,
        test_size=0.20,
        random_state=42
    )

if len(train) < 20 or len(test) < 5:
    st.error("Insufficient training or testing records.")
    st.stop()

# ==========================================
# 6. DATA PREPROCESSING
# ==========================================

def create_preprocessor(features):

    categorical = train[features].select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical = [
        col for col in features
        if col not in categorical
    ]

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    return ColumnTransformer([
        ("cat", categorical_pipeline, categorical),
        ("num", numerical_pipeline, numerical)
    ])

# ==========================================
# 7. MACHINE LEARNING MODELS
# ==========================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

    "Decision Tree":
        DecisionTreeRegressor(
            min_samples_leaf=5,
            random_state=42
        ),

    "Support Vector Regression":
        SVR(C=10),

    "KNN":
        KNeighborsRegressor(
            n_neighbors=9,
            weights="distance"
        ),

    "ANN (MLP)":
        MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=700,
            early_stopping=True,
            random_state=42
        ),

    "Lasso Regression":
        Lasso(
            alpha=0.01,
            max_iter=10000
        ),

    "Ridge Regression":
        Ridge(alpha=10)
}

# ==========================================
# 8. MODEL EVALUATION FUNCTION
# ==========================================

def evaluate_models(target, features):

    results = []
    trained_models = {}

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessing", create_preprocessor(features)),
            ("model", model)
        ])

        pipeline.fit(
            train[features],
            train[target]
        )

        predictions = pipeline.predict(
            test[features]
        )

        trained_models[name] = pipeline

        results.append({

            "Model": name,

            "MAE": mean_absolute_error(
                test[target], predictions
            ),

            "RMSE": np.sqrt(
                mean_squared_error(
                    test[target], predictions
                )
            ),

            "R2 Score": r2_score(
                test[target], predictions
            )
        })

    # Baseline comparison

    baseline = DummyRegressor(strategy="mean")

    baseline.fit(
        train[features],
        train[target]
    )

    baseline_pred = baseline.predict(
        test[features]
    )

    baseline_mae = mean_absolute_error(
        test[target], baseline_pred
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            test[target], baseline_pred
        )
    )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        ["MAE", "RMSE"]
    ).reset_index(drop=True)

    best = results_df.iloc[0]

    if (
        best["MAE"] < baseline_mae
        and best["RMSE"] < baseline_rmse
    ):

        best_name = best["Model"]
        best_model = trained_models[best_name]
        valid = True

    else:

        best_name = "Mean Baseline"
        best_model = baseline
        valid = False

    return results_df, best_model, best_name, valid


# ==========================================
# 9. CROP YIELD FORECASTING
# ==========================================

st.divider()

st.header("2. Crop Yield Forecasting")

with st.spinner("Training Crop Yield Models..."):

    yield_results, yield_model, yield_best, yield_valid = (
        evaluate_models(
            YIELD,
            yield_features
        )
    )

st.subheader("Crop Yield Model Performance")

st.dataframe(
    yield_results,
    use_container_width=True,
    hide_index=True
)

st.success(
    f"Selected Yield Model: {yield_best}"
)

if not yield_valid:
    st.warning(
        "No tested ML model beats the mean baseline "
        "on both MAE and RMSE."
    )

# ==========================================
# 10. FARMER PROFITABILITY PREDICTION
# ==========================================

st.divider()

st.header("3. Farmer Profitability Prediction")

with st.spinner("Training Profitability Models..."):

    profit_results, profit_model, profit_best, profit_valid = (
        evaluate_models(
            PROFIT,
            profit_features
        )
    )

st.subheader("Profitability Model Performance")

st.dataframe(
    profit_results,
    use_container_width=True,
    hide_index=True
)

st.success(
    f"Selected Profitability Model: {profit_best}"
)

if not profit_valid:
    st.warning(
        "No tested ML model beats the mean baseline "
        "on both MAE and RMSE."
    )

# ==========================================
# 11. PREDICTION INPUT FUNCTION
# ==========================================

def prediction_form(features, form_key):

    values = {}

    with st.form(form_key):

        left, right = st.columns(2)

        for i, col in enumerate(features):

            container = left if i % 2 == 0 else right

            with container:

                if (
                    df[col].dtype == "object"
                    or str(df[col].dtype) == "category"
                ):

                    options = sorted(
                        df[col].dropna().astype(str).unique()
                    )

                    values[col] = st.selectbox(
                        col,
                        options,
                        key=form_key + col
                    )

                else:

                    series = pd.to_numeric(
                        df[col], errors="coerce"
                    ).dropna()

                    values[col] = st.number_input(
                        col,
                        value=float(series.median()),
                        min_value=float(series.min()),
                        max_value=float(series.max()),
                        key=form_key + col
                    )

        submit = st.form_submit_button(
            "Generate Prediction",
            type="primary"
        )

    return values, submit

# ==========================================
# 12. CROP YIELD PREDICTION INTERFACE
# ==========================================

st.divider()

st.header("4. Forecast Crop Yield")

st.write(
    "Enter agricultural and climatic parameters "
    "to estimate expected crop production."
)

yield_values, yield_submit = prediction_form(
    yield_features,
    "yield_prediction"
)

if yield_submit:

    input_data = pd.DataFrame(
        [yield_values],
        columns=yield_features
    )

    prediction = float(
        yield_model.predict(input_data)[0]
    )

    st.metric(
        "Forecasted Crop Yield",
        f"{prediction:.2f} Ton/Hectare"
    )

    if not yield_valid:
        st.caption(
            "This estimate uses the mean baseline because "
            "no tested ML model beat it."
        )

# ==========================================
# 13. PROFITABILITY PREDICTION INTERFACE
# ==========================================

st.divider()

st.header("5. Forecast Farmer Profitability")

st.write(
    "Enter agricultural information, market price, "
    "cultivated area and expected crop yield to "
    "estimate farmer profitability."
)

profit_values, profit_submit = prediction_form(
    profit_features,
    "profit_prediction"
)

if profit_submit:

    input_data = pd.DataFrame(
        [profit_values],
        columns=profit_features
    )

    prediction = float(
        profit_model.predict(input_data)[0]
    )

    st.metric(
        "Forecasted Farmer Profit",
        f"₹ {prediction:,.2f}"
    )

    if not profit_valid:
        st.caption(
            "This estimate uses the mean baseline because "
            "no tested ML model beat it."
        )

# ==========================================
# 14. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    ### About AgriSmart

    AgriSmart is an AI-powered agricultural forecasting
    platform designed to predict crop yield and farmer
    profitability using historical agricultural data
    and machine learning algorithms.

    **Prediction Objectives:**
    1. Crop Yield Forecasting
    2. Farmer Profitability Prediction
    """
)
