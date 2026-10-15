"""Train the final churn model and save it to models/churn_model.joblib."""
from pathlib import Path

import joblib
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import build_preprocessor, load_clean_data, split_data

MODEL_PATH = Path("models/churn_model.joblib")
THRESHOLD = 0.55  # chosen on cross-validated training predictions (notebook 05)


def build_model(X_train, y_train) -> Pipeline:
    """Return an unfitted pipeline: preprocessing + XGBoost."""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X_train)),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=42,
                ),
            ),
        ]
    )


def train_and_save(model_path: Path = MODEL_PATH):
    df = load_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = build_model(X_train, y_train)
    pipeline.fit(X_train, y_train)

    # One final check on the held-out test set
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    print(classification_report(y_test, pred))
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 3))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": THRESHOLD}, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    train_and_save()