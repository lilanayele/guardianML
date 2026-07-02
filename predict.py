import joblib
import pandas as pd

rf = joblib.load("models/random_forest.pkl")
iso = joblib.load("models/isolation_forest.pkl")


def predict(sample_df):

    pred = rf.predict(sample_df)[0]

    anomaly = iso.predict(sample_df)[0]

    prob = rf.predict_proba(sample_df)[0]
    risk = int(max(prob) * 100)

    status = "Anomaly Detected" if anomaly == -1 else "Normal"

    return pred, status, risk
