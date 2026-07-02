import streamlit as st
import pandas as pd

from src.predict import predict

st.set_page_config(page_title="🛡️ GuardianML Cyber Threat Detector", layout="wide")

st.title("🛡️ GuardianML Cyber Threat Detector")
st.write("Upload a CSV file to detect suspicious network activity.")

# Upload file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    # REMOVE TARGET COLUMN IF PRESENT (IMPORTANT FIX)
    if "attack_detected" in df.columns:
        df = df.drop(columns=["attack_detected"])

    st.subheader("🔍 Running Predictions...")

    results = []

    # Predict row by row
    for i in range(len(df)):
        sample = df.iloc[[i]]  # keep as DataFrame

        pred, status, risk = predict(sample)

        results.append({
            "Prediction": pred,
            "Status": status,
            "Risk Score": risk
        })

    result_df = pd.DataFrame(results)

    st.subheader("📈 Results")
    st.dataframe(result_df)

    # Risk summary
    st.subheader("⚠️ Risk Summary")

    high_risk = (result_df["Risk Score"] > 70).sum()
    normal = (result_df["Risk Score"] <= 70).sum()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("High Risk Events", high_risk)

    with col2:
        st.metric("Normal Events", normal)

else:
    st.info("Please upload a CSV file to begin analysis.")
