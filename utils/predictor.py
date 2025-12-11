# utils/predictor.py
import pickle
import pandas as pd
import numpy as np
from typing import Tuple, List

_ARTIFACTS_PATH = "models"

def load_artifacts(model_path: str = f"{_ARTIFACTS_PATH}/model.pkl",
                   transformer_path: str = f"{_ARTIFACTS_PATH}/transformer.pkl",
                   features_path: str = f"{_ARTIFACTS_PATH}/features.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(transformer_path, "rb") as f:
        transformer = pickle.load(f)
    with open(features_path, "rb") as f:
        feature_names = pickle.load(f)
    return model, transformer, feature_names

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same deterministic preprocessing as training."""
    df = df.copy()
    # drop ID if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    # TotalCharges -> numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    # engineered features
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["tenure_group"] = pd.cut(df["tenure"], bins=[-1,12,36,100], labels=["New","Mid","Loyal"])
    service_cols = [
        "OnlineSecurity","OnlineBackup","DeviceProtection",
        "TechSupport","StreamingTV","StreamingMovies"
    ]
    df["service_count"] = (df[service_cols] == "Yes").sum(axis=1)
    df["is_fiber_optic"] = (df["InternetService"] == "Fiber optic").astype(int)
    return df

def predict(df: pd.DataFrame, model, transformer) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      probs: 1D array of churn probabilities (float 0-1)
      preds: binary predictions (0/1) using default 0.5 threshold
    """
    X = preprocess_input(df)
    X_trans = transformer.transform(X)
    probs = model.predict_proba(X_trans)[:,1]
    preds = (probs >= 0.5).astype(int)
    return probs, preds

def risk_segment(probs: List[float], thresholds: Tuple[float,float]=(0.7,0.4)) -> List[str]:
    """
    thresholds: (high_thresh, med_thresh)
    - prob >= high_thresh -> 'High'
    - prob >= med_thresh -> 'Medium'
    - else -> 'Low'
    """
    high, med = thresholds
    labels = []
    for p in probs:
        if p >= high:
            labels.append("High")
        elif p >= med:
            labels.append("Medium")
        else:
            labels.append("Low")
    return labels
