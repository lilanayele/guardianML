import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    df = df.copy()  # IMPORTANT FIX

    encoders = {}

    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df.loc[:, col] = le.fit_transform(df[col])  # FIX WARNING
            encoders[col] = le

    return df, encoders
