import joblib
import pandas as pd

rf = joblib.load("models/random_forest.pkl")

def predict(sample_df):

    # REMOVE LABEL IF PRESENT
    if "attack_detected" in sample_df.columns:
        sample_df = sample_df.drop(columns=["attack_detected"])

    # REMOVE ID COLUMN (CRITICAL FIX)
    if "session_id" in sample_df.columns:
        sample_df = sample_df.drop(columns=["session_id"])

    # PREDICT
    pred = rf.predict(sample_df)[0]
    proba = rf.predict_proba(sample_df)[0][1]

    risk = int(proba * 100)

    status = "⚠️ Attack Detected" if pred == 1 else "✅ Normal"

    return pred, status, risk
