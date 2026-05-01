import numpy as np
import pandas as pd

def create_features(df):

    df = df.copy()

    # Convert Time column
    df["Hour"] = (df["Time"] // 3600) % 24

    # Log transform Amount
    df["Log_Amount"] = np.log1p(df["Amount"])

    # Simple velocity feature
    df["Velocity"] = (
        df["Amount"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    # Night transaction flag
    df["Is_Night"] = df["Hour"].apply(
        lambda x: 1 if x < 6 else 0
    )

    return df