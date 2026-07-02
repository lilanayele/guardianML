import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    df = df.dropna()

    encoders = {}

    for col in df.select_dtypes(include="object").columns:
        if col != "label":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    return df, encoders
