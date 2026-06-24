from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_PATH = Path("data/processed/ai4i_features.csv")
MODEL_DIR = Path("models")


def train():
    df = pd.read_csv(DATA_PATH)

    if "target" not in df.columns:
        raise ValueError("Expected 'target' column in AI4I features file.")

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train_scaled, y_train)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "ai4i_xgb.joblib")
    joblib.dump(scaler, MODEL_DIR / "ai4i_scaler.joblib")
    joblib.dump(list(X.columns), MODEL_DIR / "ai4i_columns.joblib")


if __name__ == "__main__":
    train()
