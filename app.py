import streamlit as st
import pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.svm import SVR,SVC
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,f1_score
from sklearn.compose import ColumnTransformer

st.set_page_config(page_title="AgriSmart Crop Yield",page_icon="🌾",layout="wide")
st.title("🌾 AgriSmart — Rainfall & Crop Yield Prediction")
st.caption("Upload the supplied CSV. The app trains and benchmarks models using the latest year as a holdout.")
file=st.sidebar.file_uploader("Upload crop dataset CSV",type=["csv"])
if file is None:
    st.info("Upload india_rainfall_crop_yield_2014_2024(1).csv to continue."); st.stop()
df=pd.read_csv(file)
target="Yield_ton_per_hectare"
required=["Year",target,"State","District","Season","Crop_Type","Rainfall_mm","Avg_Temperature_C","Humidity_percent","Soil_Type","Irrigation_Type","Fertilizer_Used_kg","Pesticide_Used_kg","Cultivated_Area_hectares","Market_Price_per_ton"]
missing=[c for c in required if c not in df]
if missing: st.error("Missing columns: "+", ".join(missing)); st.stop()
features=[c for c in required if c!=target]
cat=df[features].select_dtypes(include=["object","category"]).columns.tolist()
num=[c for c in features if c not in cat]
def prep():
    return ColumnTransformer([("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat),
      ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num)])
if df.Year.nunique()>1:
    tr=df[df.Year<df.Year.max()].copy(); te=df[df.Year==df.Year.max()].copy()
else:
    from sklearn.model_selection import train_test_split
    tr,te=train_test_split(df,test_size=.2,random_state=42)
if len(tr)<20 or len(te)<5: st.error("Insufficient train/test records."); st.stop()
reg_models={"Linear Regression":LinearRegression(),"Random Forest":RandomForestRegressor(n_estimators=300,min_samples_leaf=2,random_state=42,n_jobs=-1),
"Decision Tree":DecisionTreeRegressor(min_samples_leaf=5,random_state=42),"SVR":SVR(C=10),
"KNN":KNeighborsRegressor(n_neighbors=9,weights="distance"),"ANN (MLP)":MLPRegressor(hidden_layer_sizes=(100,50),max_iter=700,early_stopping=True,random_state=42),
"Lasso":Lasso(alpha=.01,max_iter=10000),"Ridge":Ridge(alpha=10)}
rr=[]; rp={}
with st.spinner("Training regression models..."):
    for name,model in reg_models.items():
        pipe=Pipeline([("prep",prep()),("model",model)]); pipe.fit(tr[features],tr[target]); pred=pipe.predict(te[features]); rp[name]=pipe
        rr.append({"Model":name,"MAE":mean_absolute_error(te[target],pred),"RMSE":mean_squared_error(te[target],pred)**.5,"R2":r2_score(te[target],pred)})
    dummy=DummyRegressor(strategy="mean").fit(tr[features],tr[target]); dp=dummy.predict(te[features])
    dm=mean_absolute_error(te[target],dp); dr=mean_squared_error(te[target],dp)**.5
reg=pd.DataFrame(rr).sort_values(["MAE","RMSE"]).reset_index(drop=True); w=reg.iloc[0]
if w.MAE<dm and w.RMSE<dr: chosen=rp[w.Model]; chosen_name=w.Model; good=True
else: chosen=dummy; chosen_name="Mean baseline (no ML model beats baseline)"; good=False
q1,q2=tr[target].quantile([1/3,2/3]).tolist(); names=["Low","Medium","High"]
def band(s): return pd.cut(s,[-np.inf,q1,q2,np.inf],labels=names,include_lowest=True).astype(str)
label_encoder=LabelEncoder(); yt=label_encoder.fit_transform(band(tr[target])); yv=label_encoder.transform(band(te[target]))
clfs={"Logistic Regression":LogisticRegression(max_iter=2000,class_weight="balanced"),
"Random Forest":RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight="balanced",random_state=42,n_jobs=-1),
"Decision Tree":DecisionTreeClassifier(min_samples_leaf=5,class_weight="balanced",random_state=42),
"SVC":SVC(C=2,class_weight="balanced"),"KNN":KNeighborsClassifier(n_neighbors=9,weights="distance"),
"ANN (MLP)":MLPClassifier(hidden_layer_sizes=(100,50),max_iter=700,early_stopping=True,random_state=42)}
cc=[]; cp={}
with st.spinner("Training classification models..."):
    for name,model in clfs.items():
        pipe=Pipeline([("prep",prep()),("model",model)]); pipe.fit(tr[features],yt); p=pipe.predict(te[features]); cp[name]=pipe
        cc.append({"Model":name,"Accuracy":accuracy_score(yv,p),"Macro F1":f1_score(yv,p,average="macro",zero_division=0)})
ct=pd.DataFrame(cc).sort_values(["Macro F1","Accuracy"],ascending=False).reset_index(drop=True)
st.subheader("Dataset overview")
a,b,c=st.columns(3); a.metric("Rows",f"{len(df):,}"); b.metric("Features",len(features)); c.metric("Year range",f"{df.Year.min()}–{df.Year.max()}")
st.subheader("Regression benchmark — yield (ton/hectare)")
st.dataframe(reg,use_container_width=True,hide_index=True); st.write("Selected:",chosen_name)
if not good: st.warning(f"No ML regressor beats the mean baseline on both MAE and RMSE. Prediction uses the baseline; this dataset does not currently support a claim of accurate yield prediction.")
st.subheader("Classification benchmark — yield band")
st.dataframe(ct,use_container_width=True,hide_index=True)
bestc=ct.iloc[0].Model; st.write("Best classifier by macro-F1:",bestc)
st.caption(f"Training-set yield cutoffs: Low ≤ {q1:.2f}; Medium ≤ {q2:.2f}; High > {q2:.2f}.")
st.divider(); st.subheader("Predict")
with st.form("prediction"):
    vals={}; left,right=st.columns(2)
    for i,col in enumerate(features):
        with (left if i%2==0 else right):
            if col in cat: vals[col]=st.selectbox(col,sorted(df[col].dropna().astype(str).unique()))
            else:
                s=pd.to_numeric(df[col],errors="coerce").dropna()
                vals[col]=st.number_input(col,value=float(s.median()),min_value=float(s.min()),max_value=float(s.max()))
    go=st.form_submit_button("Predict yield and category",type="primary")
if go:
    row=pd.DataFrame([vals],columns=features); pred=float(chosen.predict(row)[0]); category=label_encoder.inverse_transform(cp[bestc].predict(row).astype(int))[0]
    x,y=st.columns(2); x.metric("Predicted yield",f"{pred:.2f} ton/hectare"); y.metric("Yield band",str(category))
    if not good: st.caption("Yield estimate is the training mean because no tested ML model beat the baseline.")
st.caption("For real deployment, validate source data and evaluate on independent future seasons and locations.")
