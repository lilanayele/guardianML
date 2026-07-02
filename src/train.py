import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from preprocessing import load_data, preprocess_data
from feature_engineering import create_features


# LOAD DATA
df = load_data("data/raw/network_data.csv")

# REMOVE ID COLUMN (IMPORTANT FIX)
if "session_id" in df.columns:
    df = df.drop(columns=["session_id"])

# PREPROCESS
df, encoders = preprocess_data(df)

# FEATURE ENGINEERING
df = create_features(df)

# TARGET COLUMN
target_col = "attack_detected"

# SPLIT
X = df.drop(columns=[target_col])
y = df[target_col]

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODEL 1: CLASSIFIER
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# MODEL 2: ANOMALY DETECTOR
iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X)

# SAVE MODELS
joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(iso, "models/isolation_forest.pkl")

print("✅ Training complete. Models saved successfully.")
