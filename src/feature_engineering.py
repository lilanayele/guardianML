def create_features(df):
    # simple safe features (works for most datasets)
    if "bytes" in df.columns and "duration" in df.columns:
        df["packet_rate"] = df["bytes"] / (df["duration"] + 1)

    return df
