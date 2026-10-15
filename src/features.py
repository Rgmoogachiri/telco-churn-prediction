"""Loading, splitting and preprocessing for the Churn Model"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLEAN_PATH = Path("data/processed/telco_clean.csv")
TARGET = "Churn"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_clean_data(path: Path = CLEAN_PATH) -> pd.DataFrame:
    """Load the cleaned dataset created by src/data_cleaning.py."""
    return pd.read_csv(path)

def split_data(df: pd.DataFrame):
    """Stratified train/test split. Returns X_train, X_test, y_train, y_test."""
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Return a NEW, unfitted preprocessor for the columns in X.

    Numeric columns are standardised; text columns are one-hot encoded.
    Call this once per model so each pipeline fits its own copy.
    """
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
    numeric_cols = X.select_dtypes(include="number").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(
                    drop="if_binary", handle_unknown="ignore", sparse_output=False
                ),
                categorical_cols,
            ),
        ]
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor