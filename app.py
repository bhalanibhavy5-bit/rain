
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor

from sklearn.linear_model import (
    LinearRegression, Lasso, Ridge, LogisticRegression
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# 1. STREAMLIT CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AgriSmart - AI Agriculture",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 AgriSmart")
st.subheader("AI-Based Crop Yield & Farmer Profitability Prediction")

st.write(
    "An intelligent agricultural prediction system "
    "using Machine Learning algorithms."
)

st.sidebar.title("AgriSmart Navigation")

uploaded_file = st.sidebar.file_uploader(
    "Upload Agricultural Dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload your agricultural CSV dataset.")
    st.stop()

df = pd.read_csv(uploaded_file)

# --------------------------------------------------
# 2. DATASET VALIDATION
# --------------------------------------------------

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

missing = [c for c in required_columns if c not in df.columns]

if missing:
    st.error("Missing columns: " + ", ".join(missing))
    st.stop()

df = df.dropna(subset=[YIELD, PROFIT, "Year"]).copy()

# --------------------------------------------------
# 3. DATASET OVERVIEW
# --------------------------------------------------

st.header("1. Agricultural Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", f"{len(df):,}")
col2.metric("Total Features", len(df.columns))
col3.metric("Missing Values", int(df.isnull().sum().sum()))
col4.metric(
    "Year Range",
    f"{int(df.Year.min())}-{int(df.Year.max())}"
)

st.dataframe(df.head(20), use_container_width=True)

with st.expander("View Statistical Summary"):
    st.write(df.describe(include="all"))

with st.expander("View Missing Values"):
    st.write(df.isnull().sum())

# --------------------------------------------------
# 4. FEATURE SELECTION
# --------------------------------------------------

# Exclude ID and target variables from yield features.
base_features = [
    c for c in df.columns
    if c not in ["Record_ID", YIELD, PROFIT]
]

yield_features = base_features

# Yield is included because profitability depends
# on actual or expected crop production.
profit_features = base_features + [YIELD]

# --------------------------------------------------
# 5. TRAIN AND TEST DATA
# --------------------------------------------------

if df["Year"].nunique() >= 2:

    latest_year = df["Year"].max()

    train = df[df["Year"] < latest_year].copy()
    test = df[df["Year"] == latest_year].copy()

    split_message = (
        f"Training data: Years before {latest_year} | "
        f"Testing data: {latest_year}"
    )

else:

    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        df,
        test_size=0.20,
        random_state=42
    )

    split_message = "Random 80:20 train-test split"

if len(train) < 20 or len(test) < 5:
    st.error("Insufficient data for model evaluation.")
    st.stop()

st.info(split_message)

# --------------------------------------------------
# 6. DATA PREPROCESSING
# --------------------------------------------------

def create_preprocessor(features):

    categorical = train[features].select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical = [
        c for c in features
        if c not in categorical
    ]

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipeline, categorical),
        ("numerical", numerical_pipeline, numerical)
    ])

    return preprocessor


# --------------------------------------------------
# 7. REGRESSION MODELS
# --------------------------------------------------

regression_models = {

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


# --------------------------------------------------
# 8. REGRESSION MODEL EVALUATION
# --------------------------------------------------

def evaluate_regression(target, features):

    results = []
    trained_models = {}

    for name, model in regression_models.items():

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

    # Baseline model
    baseline = DummyRegressor(strategy="mean")

    baseline.fit(
        train[features],
        train[target]
    )

    baseline_predictions = baseline.predict(
        test[features]
    )

    baseline_mae = mean_absolute_error(
        test[target], baseline_predictions
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            test[target], baseline_predictions
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
        beats_baseline = True

    else:

        best_name = "Mean Baseline"
        best_model = baseline
        beats_baseline = False

    return (
        results_df,
        best_model,
        best_name,
        beats_baseline
    )


# --------------------------------------------------
# 9. CROP YIELD PREDICTION MODELS
# --------------------------------------------------

st.header("2. Crop Yield Prediction")

with st.spinner("Training crop yield models..."):

    yield_results, yield_model, yield_best, yield_valid = (
        evaluate_regression(
            YIELD,
            yield_features
        )
    )

st.subheader("Yield Regression Performance")

st.dataframe(
    yield_results,
    use_container_width=True,
    hide_index=True
)

st.success(f"Selected Yield Model: {yield_best}")

if not yield_valid:
    st.warning(
        "No tested ML model beats the mean baseline "
        "on both MAE and RMSE."
    )

# --------------------------------------------------
# 10. PROFITABILITY PREDICTION MODELS
# --------------------------------------------------

st.header("3. Farmer Profitability Prediction")

with st.spinner("Training profitability models..."):

    profit_results, profit_model, profit_best, profit_valid = (
        evaluate_regression(
            PROFIT,
            profit_features
        )
    )

st.subheader("Profitability Regression Performance")

st.dataframe(
    profit_results,
    use_container_width=True,
    hide_index=True
)

st.success(f"Selected Profit Model: {profit_best}")

if not profit_valid:
    st.warning(
        "No tested ML model beats the mean baseline "
        "on both MAE and RMSE."
    )

# --------------------------------------------------
# 11. CLASSIFICATION MODELS
# --------------------------------------------------

st.header("4. Crop Yield Classification")

# Training-only thresholds
q1, q2 = train[YIELD].quantile(
    [1/3, 2/3]
).tolist()

class_labels = ["Low", "Medium", "High"]

def create_yield_classes(values):

    return pd.cut(
        values,
        bins=[-np.inf, q1, q2, np.inf],
        labels=class_labels,
        include_lowest=True
    ).astype(str)


label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(
    create_yield_classes(train[YIELD])
)

y_test = label_encoder.transform(
    create_yield_classes(test[YIELD])
)

classification_models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        ),

    "Support Vector Machine":
        SVC(
            C=2,
            class_weight="balanced"
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=9,
            weights="distance"
        ),

    "ANN (MLP)":
        MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=700,
            early_stopping=True,
            random_state=42
        )
}


classification_results = []
classification_pipelines = {}

for name, model in classification_models.items():

    pipeline = Pipeline([
        ("preprocessing", create_preprocessor(yield_features)),
        ("model", model)
    ])

    pipeline.fit(
        train[yield_features],
        y_train
    )

    predictions = pipeline.predict(
        test[yield_features]
    )

    classification_pipelines[name] = pipeline

    classification_results.append({

        "Model": name,

        "Accuracy": accuracy_score(
            y_test, predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )
    })


classification_df = pd.DataFrame(
    classification_results
).sort_values(
    ["F1 Score", "Accuracy"],
    ascending=False
).reset_index(drop=True)

best_classifier_name = classification_df.iloc[0]["Model"]

best_classifier = classification_pipelines[
    best_classifier_name
]

st.dataframe(
    classification_df,
    use_container_width=True,
    hide_index=True
)

st.success(
    f"Best Classification Model: {best_classifier_name}"
)

st.write(
    f"Yield categories: Low ≤ {q1:.2f}, "
    f"Medium ≤ {q2:.2f}, High > {q2:.2f}"
)


# --------------------------------------------------
# 12. INTERACTIVE PREDICTION INPUTS
# --------------------------------------------------

def create_input_form(features, form_key):

    values = {}

    with st.form(form_key):

        left, right = st.columns(2)

        for i, column in enumerate(features):

            container = left if i % 2 == 0 else right

            with container:

                if (
                    df[column].dtype == "object"
                    or str(df[column].dtype) == "category"
                ):

                    options = sorted(
                        df[column]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    values[column] = st.selectbox(
                        column,
                        options,
                        key=form_key + column
                    )

                else:

                    series = pd.to_numeric(
                        df[column],
                        errors="coerce"
                    ).dropna()

                    values[column] = st.number_input(
                        column,
                        value=float(series.median()),
                        min_value=float(series.min()),
                        max_value=float(series.max()),
                        key=form_key + column
                    )

        submitted = st.form_submit_button(
            "Generate Prediction",
            type="primary"
        )

    return values, submitted


# --------------------------------------------------
# 13. YIELD PREDICTION
# --------------------------------------------------

st.divider()

st.header("5. Predict Crop Yield")

yield_values, yield_submit = create_input_form(
    yield_features,
    "yield_form"
)

if yield_submit:

    input_data = pd.DataFrame(
        [yield_values],
        columns=yield_features
    )

    prediction = float(
        yield_model.predict(input_data)[0]
    )

    predicted_class = label_encoder.inverse_transform(
        best_classifier.predict(input_data).astype(int)
    )[0]

    col1, col2 = st.columns(2)

    col1.metric(
        "Predicted Crop Yield",
        f"{prediction:.2f} ton/hectare"
    )

    col2.metric(
        "Yield Category",
        predicted_class
    )

    if not yield_valid:
        st.caption(
            "The selected yield estimate is the mean baseline."
        )


# --------------------------------------------------
# 14. PROFITABILITY PREDICTION
# --------------------------------------------------

st.divider()

st.header("6. Predict Farmer Profitability")

st.info(
    "Enter an expected or observed crop yield. "
    "For pre-harvest estimates, use expected yield."
)

profit_values, profit_submit = create_input_form(
    profit_features,
    "profit_form"
)

if profit_submit:

    input_data = pd.DataFrame(
        [profit_values],
        columns=profit_features
    )

    profit_prediction = float(
        profit_model.predict(input_data)[0]
    )

    st.metric(
        "Estimated Farmer Profit",
        f"₹ {profit_prediction:,.2f}"
    )

    if not profit_valid:
        st.caption(
            "The selected profit estimate is the mean baseline."
        )


# --------------------------------------------------
# 15. FINAL INFORMATION
# --------------------------------------------------

st.divider()

st.subheader("About AgriSmart")

st.write(
    "AgriSmart uses machine learning to analyse "
    "agricultural conditions, estimate crop yield, "
    "classify yield performance and predict farmer "
    "profitability. Predictions should be validated "
    "with independent agricultural data before "
    "being used for actual farming or financial decisions."
)
