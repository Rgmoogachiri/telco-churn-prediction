"""Cleaning steps for the Telco customer churn dataset."""
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_PATH = Path("data/processed/telco_clean.csv")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw Telco data.

    Works on training data (has customerID and Churn) and on new
    customers (e.g. from the app), which may not have those columns.
    """
    df = df.copy()

    # TotalCharges is stored as text; blanks become NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Blank TotalCharges should only happen for brand-new customers (tenure 0)
    blank = df["TotalCharges"].isna()
    if not df.loc[blank, "tenure"].eq(0).all():
        raise ValueError("Missing TotalCharges found for customers with tenure > 0")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Identifier carries no pattern for the model
    df = df.drop(columns="customerID", errors="ignore")

    # Target: Yes -> 1, No -> 0 (only present in training data)
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    return df


if __name__ == "__main__":
    raw = pd.read_csv(RAW_PATH)
    clean = clean_data(raw)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved {PROCESSED_PATH} with shape {clean.shape}")