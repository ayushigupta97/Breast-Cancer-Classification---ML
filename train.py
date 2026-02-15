import os
import numpy as np
import pandas as pd
import joblib

# Sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ===============================
# 1️⃣ DATA LOADING
# ===============================

print("Loading Dataset...")
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

print("Dataset Shape:", df.shape)

# ===============================
# 2️⃣ DATA VALIDATION
# ===============================

print("\nChecking Missing Values...")
print(df.isnull().sum().sum())

print("\nData Types:")
print(df.dtypes)

print("\nClass Distribution:")
print(df["target"].value_counts())

# ===============================
# 3️⃣ FEATURE ENGINEERING
# ===============================

print("\nApplying Feature Engineering...")

df["radius_texture_ratio"] = df["mean radius"] / df["mean texture"]

X = df.drop("target", axis=1)
y = df["target"]

# ===============================
# 4️⃣ TRAIN TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 5️⃣ FEATURE SCALING
# ===============================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")

# ===============================
# 6️⃣ MODEL TRAINING
# ===============================

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

results = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Scaling only for required models
    if name in ["Logistic Regression", "kNN", "Naive Bayes"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    # Save model
    joblib.dump(model, f"model/{name.replace(' ', '_').lower()}.pkl")

    # Evaluation
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    })

# ===============================
# 7️⃣ COMPARISON TABLE
# ===============================

results_df = pd.DataFrame(results)
print("\nFinal Model Comparison:")
print(results_df)

# ===============================
# 8️⃣ SAVE TEST CSV FOR STREAMLIT
# ===============================

test_df = pd.DataFrame(X_test, columns=X.columns)
test_df["target"] = y_test.values

test_df.to_csv("breast_cancer_test.csv", index=False)

print("\nTest CSV created successfully!")
print("Training pipeline completed.")