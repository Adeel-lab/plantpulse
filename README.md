<div align="center">

# 🌿 PlantPulse — Predictive Maintenance AI Platform

**Real-time machine failure prediction and bearing degradation detection, served via a production-ready REST API and interactive Streamlit dashboard.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-189AB4?style=flat-square)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

</div>

---

## What is PlantPulse?

PlantPulse is an end-to-end **predictive maintenance system** that detects equipment failures and bearing degradation before they happen. It targets industrial plants, manufacturing lines, and process automation — environments where unexpected downtime costs thousands of euros per hour.

Two production-quality ML models are deployed behind a FastAPI backend:

| Model | Dataset | Task | Technique |
|---|---|---|---|
| **AI4I Failure Predictor** | AI4I 2020 (UCI) — 10,000 machine records | Binary classification: predict machine failure | XGBoost + StandardScaler |
| **IMS Bearing Anomaly Detector** | IMS Bearing Dataset (NASA) — 2,156 time-windows | Unsupervised anomaly detection, degradation onset | Deep Autoencoder + 3σ thresholding |

---

## Key Results

- **IMS Autoencoder** — test MSE mean of **4.76**, train MSE mean of **0.029**, confirming the model correctly learned normal behaviour and flags degradation with high sensitivity
- **652 out of 863 test windows** flagged as anomalous (3σ threshold), with **sustained degradation onset detected at global row 1,505**
- **AI4I XGBoost classifier** trained on 8,000 samples with 200 estimators and evaluated with ROC-AUC and full classification report
- Three anomaly severity thresholds: `3σ`, `95th percentile`, `99th percentile`

---

## System Architecture

```
Raw Data
   │
   ├── src/ai4i_features.py       # Feature engineering: AI4I 2020
   └── src/ims_features.py        # Feature engineering: IMS Bearing
         │
         ▼
   ┌─────────────────────────────────┐
   │         Training Pipeline       │
   │  src/train_ai4i.py (XGBoost)    │
   │  src/train_ims_autoencoder.py   │
   └─────────────────────────────────┘
         │
         ▼  saves to models/
   ┌─────────────────────────────────┐
   │       Model Artifacts           │
   │  ai4i_xgb.joblib                │
   │  ai4i_scaler.joblib             │
   │  ai4i_columns.joblib            │
   │  ims_autoencoder.keras          │
   │  ims_scaler.joblib              │
   │  ims_autoencoder_metrics.json   │
   └─────────────────────────────────┘
         │
         ▼
   ┌─────────────────────────────────┐
   │      FastAPI Backend (app.py)   │
   │  GET  /health                   │
   │  POST /predict/ai4i             │
   │  POST /predict/ims              │
   └─────────────────────────────────┘
         │
         ▼
   ┌─────────────────────────────────┐
   │   Streamlit Frontend            │
   │   streamlit_app.py              │
   └─────────────────────────────────┘
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/Adeel-lab/plantpulse.git
cd plantpulse
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the models

```bash
python src/ai4i_features.py
python src/train_ai4i.py
python src/train_ims_autoencoder.py
```

Model artifacts are saved to `models/`.

### 4. Start the API

```bash
uvicorn app:app --reload
```

API runs at `http://127.0.0.1:8000`  
Interactive docs at `http://127.0.0.1:8000/docs`

### 5. Launch the Streamlit dashboard

```bash
streamlit run streamlit_app.py
```

---

## API Reference

### `GET /health`

```json
{ "status": "ok", "models_loaded": { "ai4i": true, "ims": true } }
```

### `POST /predict/ai4i`

```json
{
  "air_temperature_k": 298.1,
  "process_temperature_k": 308.6,
  "rotational_speed_rpm": 1551,
  "torque_nm": 42.8,
  "tool_wear_min": 10,
  "type_h": 0, "type_l": 0, "type_m": 1
}
```

### `POST /predict/ims`

Upload a CSV with 48 numeric feature columns. Returns row-wise MSE, anomaly flags, severity, and degradation onset.

---

## Project Structure

```
plantpulse/
├── app.py
├── streamlit_app.py
├── requirements.txt
├── README.md
├── src/
│   ├── ai4i_features.py
│   ├── train_ai4i.py
│   ├── ims_features.py
│   └── train_ims_autoencoder.py
├── models/
│   ├── ims_autoencoder_metrics.json
│   └── ims_reconstruction_results.csv
└── data/
    ├── raw/          # Not tracked in Git
    └── processed/    # Not tracked in Git
```

---

## Datasets

| Dataset | Source | Records |
|---|---|---|
| AI4I 2020 Predictive Maintenance | [UCI ML Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) | 10,000 |
| IMS Bearing Dataset | [NASA Prognostics Center](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) | 2,156 windows |

Raw data is **not committed**. Download from the links above and place in `data/raw/`.

---

## Use Cases

- **Manufacturing plants** — early detection of motor, gearbox, or spindle failures
- **Process automation** — continuous health monitoring of rotating equipment
- **Condition-based maintenance (CBM)** — replace reactive schedules with data-driven service intervals
- **OEM integration** — REST API fits directly into SCADA or MES systems

---

## Roadmap

- [ ] Live sensor streaming (MQTT / OPC-UA)
- [ ] SHAP explainability for AI4I predictions
- [ ] Docker containerization
- [ ] MSE time-series chart in Streamlit
- [ ] Multi-asset support

---

## License

MIT License

> *"Every minute of unplanned downtime in a manufacturing plant costs between €1,000 and €10,000. PlantPulse turns sensor data into early warnings."*
