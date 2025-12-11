# ============================================================
# train.py
# LOCAL TRAINING SCRIPT FOR CHURNGUARD AI (scikit-learn 1.2.2)
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "dataset.csv"   # <-- set your local dataset path here

df = pd.read_csv(DATA_PATH)

print("Loaded dataset:", df.shape)
print(df.head())


# ============================================================
# 2. FEATURE ENGINEERING (same as your Colab pipeline)
# ============================================================

df_processed = df.copy()

# Drop ID column if present
if "customerID" in df_processed.columns:
    df_processed = df_processed.drop(columns=["customerID"])

# Convert TotalCharges
df_processed["TotalCharges"] = pd.to_numeric(df_processed["TotalCharges"], errors="coerce")
df_processed["TotalCharges"].fillna(df_processed["MonthlyCharges"] * df_processed["tenure"], inplace=True)

# Additional engineered features
df_processed["avg_monthly_spend"] = df_processed["TotalCharges"] / (df_processed["tenure"] + 1)

df_processed["tenure_group"] = pd.cut(
    df_processed["tenure"],
    bins=[-1, 12, 36, 100],
    labels=["New", "Mid", "Loyal"]
)

service_cols = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]

df_processed["service_count"] = (df_processed[service_cols] == "Yes").sum(axis=1)

df_processed["is_fiber_optic"] = (df_processed["InternetService"] == "Fiber optic").astype(int)

# Encode target
df_processed["Churn"] = df_processed["Churn"].map({"Yes": 1, "No": 0}).astype(int)


# ============================================================
# 3. SEPARATE FEATURES / TARGET
# ============================================================

target = "Churn"
X = df_processed.drop(columns=[target])
y = df_processed[target]

print("X shape:", X.shape)
print("y distribution:")
print(y.value_counts())


# ============================================================
# 4. DEFINE NUMERIC & CATEGORICAL COLUMNS
# ============================================================

numeric_cols = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "avg_monthly_spend", "service_count", "is_fiber_optic"
]

categorical_cols = [
    col for col in X.columns
    if X[col].dtype == object
]

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)


# ============================================================
# 5. BUILD PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
    ]
)

# Fit transformer locally (NOT using the old one)
preprocessor.fit(X)

print("\nTransformer fitted successfully!\n")


# ============================================================
# 6. TRANSFORM THE DATA
# ============================================================

X_trans = preprocessor.transform(X)

print("Transformed shape:", X_trans.shape)


# ============================================================
# 7. TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_trans, y, test_size=0.2, random_state=42, stratify=y
)

# Handle class imbalance
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos

print("scale_pos_weight =", scale_pos_weight)


# ============================================================
# 8. TRAIN XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=30,
    verbose=20
)

print("\nModel training completed!\n")


# ============================================================
# 9. EVALUATE
# ============================================================

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

print("=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 10. SAVE ARTIFACTS
# ============================================================

os.makedirs("models", exist_ok=True)

with open("models/transformer.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save feature names
try:
    feature_names = preprocessor.get_feature_names_out()
except:
    feature_names = [f"f_{i}" for i in range(X_trans.shape[1])]

with open("models/features.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("\nSaved model artifacts:")
print(os.listdir("models"))
print("\nTraining complete!")
