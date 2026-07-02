import streamlit as st
import pandas as pd

from src.predict import predict

st.title("🛡️ GuardianML Cyber Threat Detector")

file = st.file_uploader("Upload CSV file", type=["csv"])

if file:

    df = pd.read_csv(file)
    st.write(df.head())

    if st.button("Analyze"):

        results = []

        for _, row in df.iterrows():
            sample = pd.DataFrame([row])

            pred, status, risk = predict(sample)

            results.append([pred, status, risk])

        result_df = pd.DataFrame(results, columns=["Prediction", "Status", "Risk Score"])

        st.success("Analysis Complete")
        st.dataframe(result_df)

        st.bar_chart(result_df["Risk Score"])
