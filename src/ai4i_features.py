from pathlib import Path
import pandas as pd

RAW_PATH = Path(r"data/raw/ai4i2020.csv")
OUT_PATH = Path("data/processed/ai4i_features.csv")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace("[", "").replace("]", "").replace(" ", "_") for c in df.columns]
    return df


def build_ai4i_features(input_path: Path = RAW_PATH, output_path: Path = OUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = clean_columns(df)

    drop_cols = [c for c in ["udi", "product_id", "twf", "hdf", "pwf", "osf", "rnf"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    if "type" in df.columns:
        df["type"] = df["type"].astype(str)
        df = pd.get_dummies(df, columns=["type"], prefix="type")

    rename_map = {
        "air_temperature_k": "air_temperature_k",
        "process_temperature_k": "process_temperature_k",
        "rotational_speed_rpm": "rotational_speed_rpm",
        "torque_nm": "torque_nm",
        "tool_wear_min": "tool_wear_min",
        "machine_failure": "target"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    out = build_ai4i_features()
    print(out.head())
    print(out.shape)
