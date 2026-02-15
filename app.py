import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Breast Cancer Classification App")

# Load scaler
scaler = joblib.load("model/scaler.pkl")

# Load models
models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "kNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl")
}

model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]

uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.write(data.head())

    if "target" in data.columns:

        X = data.drop("target", axis=1)
        y = data["target"]

        if model_name in ["Logistic Regression", "kNN", "Naive Bayes"]:
            X = scaler.transform(X)

        y_pred = model.predict(X)

        st.subheader("Evaluation Metrics")
        st.write("Accuracy:", accuracy_score(y, y_pred))

        st.subheader("Classification Report")
        st.text(classification_report(y, y_pred))

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    else:
        st.error("CSV must contain 'target' column.")