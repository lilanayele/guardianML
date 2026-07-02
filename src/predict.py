import joblib
import pandas as pd

rf = joblib.load("models/random_forest.pkl")

target_col = "attack_detected"

def predict(sample_df):

    # REMOVE TARGET IF IT EXISTS
    if target_col in sample_df.columns:
        sample_df = sample_df.drop(columns=[target_col])

    # PREDICT
    pred = rf.predict(sample_df)[0]
    proba = rf.predict_proba(sample_df)[0][1]

    risk = int(proba * 100)

    if pred == 1:
        status = "⚠️ Attack Detected"
    else:
        status = "✅ Normal"

    return pred, status, risk
