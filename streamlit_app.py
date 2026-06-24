import json
from io import StringIO

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="PlantPulse MVP", page_icon="🌿", layout="wide")

DEFAULT_API_URL = "http://127.0.0.1:8000"

st.title("🌿 PlantPulse MVP")
st.caption("Streamlit frontend for the AI4I machine-failure model and IMS anomaly-detection API.")

with st.sidebar:
    st.header("API Settings")
    api_url = st.text_input("FastAPI base URL", value=DEFAULT_API_URL)
    st.markdown("Example: `http://127.0.0.1:8000`")

    if st.button("Check API health"):
        try:
            r = requests.get(f"{api_url}/health", timeout=10)
            r.raise_for_status()
            st.success("API is reachable")
            st.json(r.json())
        except Exception as e:
            st.error(f"Health check failed: {e}")


tab1, tab2 = st.tabs(["AI4I Prediction", "IMS Anomaly Detection"])

with tab1:
    st.subheader("AI4I machine-failure prediction")
    st.write("Enter one machine record and send it to the FastAPI endpoint.")

    col1, col2, col3 = st.columns(3)
    with col1:
        air_temperature_k = st.number_input("Air temperature (K)", value=298.1)
        process_temperature_k = st.number_input("Process temperature (K)", value=308.6)
        rotational_speed_rpm = st.number_input("Rotational speed (RPM)", value=1551.0)
    with col2:
        torque_nm = st.number_input("Torque (Nm)", value=42.8)
        tool_wear_min = st.number_input("Tool wear (min)", value=10.0)
    with col3:
        machine_type = st.selectbox("Machine type", options=["L", "M", "H"], index=1)

    payload = {
        "air_temperature_k": float(air_temperature_k),
        "process_temperature_k": float(process_temperature_k),
        "rotational_speed_rpm": float(rotational_speed_rpm),
        "torque_nm": float(torque_nm),
        "tool_wear_min": float(tool_wear_min),
        "type_h": 1 if machine_type == "H" else 0,
        "type_l": 1 if machine_type == "L" else 0,
        "type_m": 1 if machine_type == "M" else 0,
    }

    st.code(json.dumps(payload, indent=2), language="json")

    if st.button("Predict AI4I", type="primary"):
        try:
            r = requests.post(f"{api_url}/predict/ai4i", json=payload, timeout=30)
            r.raise_for_status()
            result = r.json()
            st.success("Prediction completed")
            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", result.get("prediction_label", "N/A"))
            c2.metric("Class", result.get("prediction", "N/A"))
            c3.metric("Failure probability", f"{result.get('failure_probability', 0):.4f}")
            st.json(result)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tab2:
    st.subheader("IMS anomaly detection")
    st.write("Upload a feature CSV and send it to the IMS FastAPI endpoint.")

    uploaded_file = st.file_uploader("Upload IMS feature CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            preview_df = pd.read_csv(uploaded_file)
            st.write("CSV preview")
            st.dataframe(preview_df.head(10), use_container_width=True)
            csv_bytes = uploaded_file.getvalue()
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            csv_bytes = None
    else:
        csv_bytes = None

    if st.button("Run IMS scoring", type="primary"):
        if csv_bytes is None:
            st.warning("Please upload a CSV file first.")
        else:
            try:
                files = {"file": (uploaded_file.name, csv_bytes, "text/csv")}
                r = requests.post(f"{api_url}/predict/ims", files=files, timeout=120)
                r.raise_for_status()
                result = r.json()
                st.success("IMS scoring completed")
                summary = result.get("summary", {})
                thresholds = result.get("thresholds", {})
                results = result.get("results", [])
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows scored", result.get("rows_scored", 0))
                c2.metric("3-sigma anomalies", summary.get("anomalies_3sigma", 0))
                c3.metric("Critical anomalies", summary.get("critical_anomalies", 0))
                st.write("Thresholds")
                st.json(thresholds)
                if results:
                    results_df = pd.DataFrame(results)
                    st.write("Prediction results")
                    st.dataframe(results_df.head(50), use_container_width=True)
                    csv_out = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download IMS results CSV",
                        data=csv_out,
                        file_name="ims_api_results.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No row-level results returned.")
                with st.expander("Raw API response"):
                    st.json(result)
            except Exception as e:
                st.error(f"IMS scoring failed: {e}")

st.markdown("---")
st.caption("Run FastAPI first, then launch this Streamlit app.")
