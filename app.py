import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Equity Pledge Default Prediction",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f4f7fb;
}
h1, h2, h3 {
    color: #1f4e79;
}
</style>
""", unsafe_allow_html=True)

st.title("Equity Pledge Default Risk Prediction")

st.markdown("""
This application predicts whether an equity pledge financing case will  
**Default (1)** or remain **Normal (0)**  
based on financial data from Chinese listed companies (2017–2022).
""")

st.markdown("---")


@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("model/logistic.pkl"),
        "Decision Tree": joblib.load("model/dt.pkl"),
        "KNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/nb.pkl"),
        "Random Forest": joblib.load("model/rf.pkl"),
        "XGBoost": joblib.load("model/xgb.pkl"),
    }
    scaler = joblib.load("model/scaler.pkl")
    return models, scaler


models, scaler = load_models()

st.sidebar.header("⚙ Model Settings")

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader(" Uploaded Dataset Preview")
    st.dataframe(data.head())

    if "target" not in data.columns:
        st.error("Target column 'target' not found in dataset.")
    else:
        X = data.drop("target", axis=1)
        y = data["target"]

        model = models[selected_model_name]


        if selected_model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
            X = scaler.transform(X)

        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_proba)
        else:
            auc = None

        # Metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)


        st.markdown("---")
        st.subheader(f" Evaluation Metrics — {selected_model_name}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col1.metric("Precision", f"{precision:.4f}")

        col2.metric("Recall", f"{recall:.4f}")
        col2.metric("F1 Score", f"{f1:.4f}")

        col3.metric("MCC Score", f"{mcc:.4f}")
        if auc is not None:
            col3.metric("AUC Score", f"{auc:.4f}")

    
        st.markdown("---")
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    
        st.markdown("---")
        st.subheader("Classification Report")

        report = classification_report(y, y_pred)
        st.text(report)

else:
    st.info("⬅ Upload a test CSV file ")
