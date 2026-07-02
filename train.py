import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from preprocessing import load_data, preprocess_data
from feature_engineering import create_features


# LOAD DATA
df = load_data("data/raw/network_data.csv")

# PREPROCESS
df, encoders = preprocess_data(df)

# FEATURES
df = create_features(df)

# SPLIT
X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODEL 1: Classification
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# MODEL 2: Anomaly Detection
iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X)

# SAVE MODELS
joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(iso, "models/isolation_forest.pkl")

print("✅ Training complete. Models saved.")
