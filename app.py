from pathlib import Path
import io
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

AI4I_MODEL_PATH = MODEL_DIR / "ai4i_xgb.joblib"
AI4I_SCALER_PATH = MODEL_DIR / "ai4i_scaler.joblib"
AI4I_COLUMNS_PATH = MODEL_DIR / "ai4i_columns.joblib"

IMS_MODEL_PATH = MODEL_DIR / "ims_autoencoder.keras"
IMS_SCALER_PATH = MODEL_DIR / "ims_scaler.joblib"
IMS_METRICS_PATH = MODEL_DIR / "ims_autoencoder_metrics.json"

app = FastAPI(title="PlantPulse MVP API", version="1.0.0")


class AI4IInput(BaseModel):
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float
    type_h: int = 0
    type_l: int = 0
    type_m: int = 0


def load_ai4i_artifacts():
    if not AI4I_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing AI4I model: {AI4I_MODEL_PATH}")
    if not AI4I_SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing AI4I scaler: {AI4I_SCALER_PATH}")
    if not AI4I_COLUMNS_PATH.exists():
        raise FileNotFoundError(f"Missing AI4I columns: {AI4I_COLUMNS_PATH}")
    model = joblib.load(AI4I_MODEL_PATH)
    scaler = joblib.load(AI4I_SCALER_PATH)
    columns = joblib.load(AI4I_COLUMNS_PATH)
    return model, scaler, columns


def load_ims_artifacts():
    if not IMS_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing IMS model: {IMS_MODEL_PATH}")
    if not IMS_SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing IMS scaler: {IMS_SCALER_PATH}")
    if not IMS_METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing IMS metrics: {IMS_METRICS_PATH}")
    model = tf.keras.models.load_model(IMS_MODEL_PATH)
    scaler = joblib.load(IMS_SCALER_PATH)
    with open(IMS_METRICS_PATH, "r") as f:
        metrics = json.load(f)
    return model, scaler, metrics


def preprocess_ims_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["filename", "file_name", "timestamp", "time", "date", "datetime"]:
            drop_cols.append(col)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        df = df.drop(columns=non_numeric_cols)
    if df.empty:
        raise ValueError("No numeric feature columns left after preprocessing.")
    return df


def reconstruction_mse(model, x: np.ndarray) -> np.ndarray:
    pred = model.predict(x, verbose=0)
    return np.mean(np.square(x - pred), axis=1)


def classify_severity(mse: float, thr_3sigma: float, thr_95: float, thr_99: float) -> str:
    if mse > thr_99:
        return "critical"
    if mse > thr_3sigma:
        return "high"
    if mse > thr_95:
        return "medium"
    return "normal"


@app.on_event("startup")
def startup_event():
    global ai4i_model, ai4i_scaler, ai4i_columns
    global ims_model, ims_scaler, ims_metrics
    ai4i_model, ai4i_scaler, ai4i_columns = load_ai4i_artifacts()
    ims_model, ims_scaler, ims_metrics = load_ims_artifacts()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": {
            "ai4i": AI4I_MODEL_PATH.exists(),
            "ims": IMS_MODEL_PATH.exists()
        }
    }


@app.post("/predict/ai4i")
def predict_ai4i(payload: AI4IInput):
    try:
        input_dict = payload.model_dump()
        df = pd.DataFrame([input_dict])
        for col in ai4i_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[ai4i_columns]
        x_scaled = ai4i_scaler.transform(df)
        prob = float(ai4i_model.predict_proba(x_scaled)[0, 1])
        pred = int(prob >= 0.5)
        return {
            "prediction": pred,
            "prediction_label": "machine_failure" if pred == 1 else "normal",
            "failure_probability": prob
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/ims")
async def predict_ims(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Please upload a CSV file.")
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        df_clean = preprocess_ims_features(df)
        expected_n_features = ims_metrics["n_features"]
        if df_clean.shape[1] != expected_n_features:
            raise HTTPException(
                status_code=400,
                detail=f"Feature mismatch: model expects {expected_n_features} columns, got {df_clean.shape[1]}."
            )
        x = df_clean.to_numpy(dtype=np.float32)
        x_scaled = ims_scaler.transform(x)
        mse = reconstruction_mse(ims_model, x_scaled)
        thr_3sigma = ims_metrics["threshold_3sigma"]
        thr_95 = ims_metrics["threshold_95"]
        thr_99 = ims_metrics["threshold_99"]
        results = []
        for i, score in enumerate(mse):
            results.append({
                "row_index": int(i),
                "reconstruction_mse": float(score),
                "anomaly_3sigma": int(score > thr_3sigma),
                "anomaly_95": int(score > thr_95),
                "anomaly_99": int(score > thr_99),
                "severity": classify_severity(float(score), thr_3sigma, thr_95, thr_99)
            })
        return {
            "rows_scored": len(results),
            "thresholds": {
                "threshold_3sigma": thr_3sigma,
                "threshold_95": thr_95,
                "threshold_99": thr_99
            },
            "summary": {
                "anomalies_3sigma": int(sum(r["anomaly_3sigma"] for r in results)),
                "critical_anomalies": int(sum(1 for r in results if r["severity"] == "critical"))
            },
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
