import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="LiverAI | Liver Condition Analysis", page_icon="🧬", layout="wide")
DATA_PATH = Path(__file__).parent / "data" / "ilpd.csv"
FEATURES = ["Age", "Gender", "TB", "DB", "Alkphos", "Sgpt", "Sgot", "TP", "ALB", "A/G Ratio"]
NUMERIC = [c for c in FEATURES if c != "Gender"]

@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        try:
            from ucimlrepo import fetch_ucirepo
            ds = fetch_ucirepo(id=225)
            frame = pd.concat([ds.data.features, ds.data.targets], axis=1)
            frame.to_csv(DATA_PATH, index=False)
        except Exception as e:
            raise RuntimeError("Dataset not found. Connect to the internet once, or place the UCI ILPD CSV at data/ilpd.csv. Details: " + str(e))
    df = pd.read_csv(DATA_PATH)
    # Normalize common dataset column names from UCI mirrors / ucimlrepo.
    rename = {c: c.strip() for c in df.columns}
    df = df.rename(columns=rename)
    aliases = {"Age":"Age", "Gender":"Gender", "TB":"TB", "DB":"DB", "Alkphos":"Alkphos", "Sgpt":"Sgpt", "Sgot":"Sgot", "TP":"TP", "ALB":"ALB", "A/G Ratio":"A/G Ratio", "Selector":"Selector"}
    df = df.rename(columns=aliases)
    if "Selector" not in df.columns:
        targets = [c for c in df.columns if c.lower() in ("selector", "class", "target")]
        if targets: df = df.rename(columns={targets[0]: "Selector"})
    if not set(FEATURES).issubset(df.columns) or "Selector" not in df.columns:
        raise ValueError(f"Unexpected dataset columns: {list(df.columns)}")
    df = df[FEATURES + ["Selector"]].copy()
    df["Selector"] = pd.to_numeric(df["Selector"], errors="coerce")
    df = df.dropna(subset=["Selector"])
    # UCI ILPD coding: 1 = liver patient, 2 = non-liver patient.
    df["Target"] = (df["Selector"] == 1).astype(int)
    return df.drop(columns="Selector")

@st.cache_resource(show_spinner="Training machine-learning models...")
def train_models():
    df = load_data()
    X, y = df[FEATURES], df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric_pipe, NUMERIC), ("cat", categorical_pipe, ["Gender"])])
    base = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=350, min_samples_leaf=2, class_weight="balanced", random_state=42),
        "SVM": SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=160, learning_rate=0.06, l2_regularization=1.0, random_state=42),
    }
    models = {name: Pipeline([("prep", prep), ("model", model)]) for name, model in base.items()}
    # The ensemble sits after one shared preprocessing stage; its base estimators
    # therefore operate on the already-transformed feature matrix.
    estimators = [(k.lower().replace(" ", "_"), model) for k, model in base.items()]
    stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=2000), stack_method="predict_proba", cv=5, passthrough=False)
    models["AI Stacked Ensemble"] = Pipeline([("prep", prep), ("model", stack)])
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = {"model": model, "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0), "recall": recall_score(y_test, pred, zero_division=0), "f1": f1_score(y_test, pred, zero_division=0), "auc": roc_auc_score(y_test, proba), "pred": pred, "proba": proba}
    return df, X_test, y_test, results

st.title("🧬 LiverAI — Liver Condition Analysis")
st.caption("Machine-learning decision-support prototype based on the methodology in your research paper")
st.warning("Educational research prototype only. It is NOT a medical diagnostic tool. Do not use it to make treatment decisions; consult a qualified clinician.")
try:
    df, X_test, y_test, results = train_models()
except Exception as exc:
    st.error(str(exc)); st.info("Run: pip install -r requirements.txt  then  streamlit run app.py"); st.stop()

page = st.sidebar.radio("Navigation", ["Dashboard", "Predict", "Model Evaluation", "Explainable AI", "About & Dataset"])
if page == "Dashboard":
    st.subheader("Project overview")
    a,b,c = st.columns(3)
    a.metric("Dataset records", f"{len(df)}")
    b.metric("Input features", "10")
    c.metric("Models compared", f"{len(results)}")
    left,right = st.columns(2)
    with left:
        st.markdown("**Class distribution**")
        counts = df.Target.map({1:"Liver patient",0:"Non-liver patient"}).value_counts()
        st.bar_chart(counts)
    with right:
        st.markdown("**Model accuracy on held-out test split**")
        st.bar_chart(pd.Series({k:v['accuracy'] for k,v in results.items()}, name="Accuracy"))
    st.markdown("**Workflow**: UCI ILPD data → preprocessing → Logistic Regression / Random Forest / SVM / Gradient Boosting → stacked ensemble → evaluation → prediction explanation.")
elif page == "Predict":
    st.subheader("Enter patient laboratory values")
    st.write("Enter values from a laboratory report. The app does not interpret units or replace medical advice.")
    with st.form("patient_form"):
        c1,c2,c3 = st.columns(3)
        age = c1.number_input("Age (years)", 1, 100, 40)
        gender = c1.selectbox("Gender", ["Female", "Male"])
        tb = c2.number_input("Total Bilirubin (TB)", min_value=0.0, value=0.8, step=0.1)
        db = c2.number_input("Direct Bilirubin (DB)", min_value=0.0, value=0.2, step=0.1)
        alk = c3.number_input("Alkaline Phosphotase (Alkphos)", min_value=0.0, value=180.0, step=1.0)
        sgpt = c1.number_input("Alamine Aminotransferase (Sgpt)", min_value=0.0, value=25.0, step=1.0)
        sgot = c2.number_input("Aspartate Aminotransferase (Sgot)", min_value=0.0, value=25.0, step=1.0)
        tp = c3.number_input("Total Proteins (TP)", min_value=0.0, value=6.8, step=0.1)
        alb = c1.number_input("Albumin (ALB)", min_value=0.0, value=3.5, step=0.1)
        ag = c2.number_input("Albumin / Globulin Ratio", min_value=0.0, value=1.0, step=0.05)
        model_name = c3.selectbox("Prediction model", list(results.keys()), index=list(results.keys()).index("AI Stacked Ensemble"))
        submitted = st.form_submit_button("Analyze patient data", type="primary")
    if submitted:
        row = pd.DataFrame([[age,gender,tb,db,alk,sgpt,sgot,tp,alb,ag]], columns=FEATURES)
        model = results[model_name]["model"]
        prediction = int(model.predict(row)[0]); probability = float(model.predict_proba(row)[0,1])
        if prediction:
            st.error(f"Model output: Liver-patient class (estimated model score: {probability:.1%})")
        else:
            st.success(f"Model output: Non-liver-patient class (estimated liver-patient score: {probability:.1%})")
        st.caption("This percentage is a model score, not a clinically validated personal risk probability.")
        st.info("This output is only a research demonstration. A clinician must interpret laboratory results and decide whether further testing is needed.")
elif page == "Model Evaluation":
    st.subheader("Held-out test-set performance")
    table = pd.DataFrame({k:{m:v[m] for m in ["accuracy","precision","recall","f1","auc"]} for k,v in results.items()}).T
    st.dataframe(table.style.format("{:.3f}"), use_container_width=True)
    chosen = st.selectbox("Confusion matrix model", list(results.keys()), index=len(results)-1)
    cm = confusion_matrix(y_test, results[chosen]["pred"])
    fig,ax = plt.subplots(); ax.imshow(cm); ax.set_title(chosen); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_xticks([0,1], ["Non-liver","Liver"]); ax.set_yticks([0,1], ["Non-liver","Liver"])
    for (i,j),val in np.ndenumerate(cm): ax.text(j,i,str(val),ha="center",va="center")
    st.pyplot(fig)
    st.caption("Metrics are recalculated when this project trains on its bundled UCI ILPD data. They may differ from figures in the paper due to split, preprocessing, and implementation differences.")
elif page == "Explainable AI":
    st.subheader("Explainable AI — feature importance")
    st.write("This page uses permutation importance on the held-out data to show which input columns most affect the selected model's score. This is a model-level explanation, not a causal or clinical explanation.")
    chosen = st.selectbox("Choose model", list(results.keys()))
    model = results[chosen]["model"]
    with st.spinner("Calculating feature importance..."):
        imp = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, scoring="f1")
    imp_df = pd.DataFrame({"Feature":FEATURES,"Importance":imp.importances_mean}).sort_values("Importance", ascending=True)
    fig,ax=plt.subplots(figsize=(8,5)); ax.barh(imp_df.Feature, imp_df.Importance); ax.set_xlabel("Mean decrease in F1 when shuffled"); st.pyplot(fig)
    st.caption("Optional SHAP integration can be added separately; permutation importance is included so the project works without SHAP dependencies.")
elif page == "About & Dataset":
    st.subheader("Research alignment")
    st.markdown("""**Paper:** Artificial Intelligence Frameworks for Liver Condition Analysis\n\n**Included:** data preprocessing, Logistic Regression, Random Forest, SVM, Gradient Boosting, stacked ensemble, evaluation metrics, and explainable feature importance.\n\n**Dataset:** UCI Indian Liver Patient Dataset (ILPD), 583 records and 10 predictor attributes. UCI's label coding is used here: `Selector = 1` liver patient; `Selector = 2` non-liver patient.\n\n**Not implemented from the paper:** image/CNN fusion, longitudinal LSTM progression, federated learning, and SHAP/LIME. The supplied ILPD dataset is tabular and does not include images or patient time series.""")
    st.markdown("Dataset source: [UCI Machine Learning Repository — ILPD](https://archive.ics.uci.edu/dataset/225/ilpd)")
    st.dataframe(df.head(10), use_container_width=True)
