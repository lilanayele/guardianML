import joblib
import pandas as pd

rf = joblib.load("models/random_forest.pkl")
encoders = joblib.load("models/encoders.pkl")


def predict(sample_df):

    # REMOVE LABEL + ID
    if "attack_detected" in sample_df.columns:
        sample_df = sample_df.drop(columns=["attack_detected"])

    if "session_id" in sample_df.columns:
        sample_df = sample_df.drop(columns=["session_id"])

    #ENCODE CATEGORICAL FEATURES
for col, encoder in encoders.items():
    if col in sample_df.columns:

        # Convert everything to string first
        sample_df[col] = sample_df[col].astype(str)

        # SAFE mapping
        sample_df[col] = sample_df[col].apply(
            lambda x: encoder.transform([x])[0]
            if x in encoder.classes_
            else -1   # unknown category
        )
    # PREDICT
    pred = rf.predict(sample_df)[0]
    proba = rf.predict_proba(sample_df)[0][1]

    risk = int(proba * 100)

    status = "⚠️ Attack Detected" if pred == 1 else "✅ Normal"

    return pred, status, risk
