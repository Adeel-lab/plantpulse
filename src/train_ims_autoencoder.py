from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

DATA_PATH = Path("data/raw/ims/feature_df.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.60
RANDOM_SEED = 42
EPOCHS = 50
BATCH_SIZE = 32

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_features(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    drop_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["filename", "file_name", "timestamp", "time", "date", "datetime"]:
            drop_cols.append(col)
    if drop_cols:
        print("Dropping known metadata columns:", drop_cols)
        df = df.drop(columns=drop_cols)
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print("Dropping non-numeric columns:", non_numeric_cols)
        df = df.drop(columns=non_numeric_cols)
    if df.empty:
        raise ValueError("No numeric feature columns left after dropping metadata/non-numeric columns.")
    print("Final feature shape:", df.shape)
    return df


def build_autoencoder(input_dim: int) -> Model:
    inp = Input(shape=(input_dim,))
    x = Dense(32, activation="relu")(inp)
    x = Dense(16, activation="relu")(x)
    latent = Dense(8, activation="relu")(x)
    x = Dense(16, activation="relu")(latent)
    x = Dense(32, activation="relu")(x)
    out = Dense(input_dim, activation="linear")(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model


def reconstruction_mse(model: Model, x: np.ndarray) -> np.ndarray:
    pred = model.predict(x, verbose=0)
    return np.mean(np.square(x - pred), axis=1)


def find_sustained_onset(flags: np.ndarray, window: int = 10, min_anoms: int = 8):
    for i in range(len(flags) - window + 1):
        if flags[i:i + window].sum() >= min_anoms:
            return i
    return None


def main():
    print(f"Loading features from: {DATA_PATH}")
    df = load_features(DATA_PATH)
    X = df.to_numpy(dtype=np.float32)
    n_train = int(len(X) * TRAIN_RATIO)
    X_train = X[:n_train]
    X_test = X[n_train:]
    print("Full shape:", X.shape)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    autoencoder = build_autoencoder(X_train_scaled.shape[1])
    early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    history = autoencoder.fit(
        X_train_scaled, X_train_scaled,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.1, shuffle=True, verbose=1,
        callbacks=[early_stop]
    )

    train_mse = reconstruction_mse(autoencoder, X_train_scaled)
    test_mse = reconstruction_mse(autoencoder, X_test_scaled)

    threshold_3sigma = float(train_mse.mean() + 3 * train_mse.std())
    threshold_95 = float(np.percentile(train_mse, 95))
    threshold_99 = float(np.percentile(train_mse, 99))

    train_anomalies_3sigma = train_mse > threshold_3sigma
    test_anomalies_3sigma = test_mse > threshold_3sigma
    train_idx = np.where(train_anomalies_3sigma)[0]
    test_idx = np.where(test_anomalies_3sigma)[0]

    onset_idx = find_sustained_onset(test_anomalies_3sigma, window=10, min_anoms=8)
    global_onset_idx = int(len(train_mse) + onset_idx) if onset_idx is not None else None

    metrics = {
        "data_path": str(DATA_PATH),
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_ratio": TRAIN_RATIO,
        "epochs_requested": EPOCHS,
        "epochs_trained": int(len(history.history["loss"])),
        "batch_size": BATCH_SIZE,
        "train_mse_mean": float(train_mse.mean()),
        "train_mse_std": float(train_mse.std()),
        "test_mse_mean": float(test_mse.mean()),
        "test_mse_std": float(test_mse.std()),
        "threshold_3sigma": threshold_3sigma,
        "threshold_95": threshold_95,
        "threshold_99": threshold_99,
        "train_anomaly_count_3sigma": int(train_anomalies_3sigma.sum()),
        "test_anomaly_count_3sigma": int(test_anomalies_3sigma.sum()),
        "test_anomaly_count_95": int((test_mse > threshold_95).sum()),
        "test_anomaly_count_99": int((test_mse > threshold_99).sum()),
        "sustained_onset_test_idx": None if onset_idx is None else int(onset_idx),
        "sustained_onset_global_idx": global_onset_idx
    }

    autoencoder.save(MODEL_DIR / "ims_autoencoder.keras")
    joblib.dump(scaler, MODEL_DIR / "ims_scaler.joblib")
    np.save(MODEL_DIR / "train_mse.npy", train_mse)
    np.save(MODEL_DIR / "test_mse.npy", test_mse)
    np.save(MODEL_DIR / "train_flags_3sigma.npy", train_anomalies_3sigma)
    np.save(MODEL_DIR / "test_flags_3sigma.npy", test_anomalies_3sigma)

    results_df = pd.DataFrame({
        "row_index": np.arange(len(X)),
        "split": ["train"] * len(train_mse) + ["test"] * len(test_mse),
        "reconstruction_mse": np.concatenate([train_mse, test_mse]),
        "anomaly_3sigma": np.concatenate([train_anomalies_3sigma, test_anomalies_3sigma]).astype(int)
    })
    results_df.to_csv(MODEL_DIR / "ims_reconstruction_results.csv", index=False)

    with open(MODEL_DIR / "ims_autoencoder_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- SUMMARY ---")
    print(json.dumps(metrics, indent=2))
    if onset_idx is not None:
        print(f"Sustained anomaly onset in test region: {onset_idx}")
        print(f"Global row index: {global_onset_idx}")
    else:
        print("No sustained anomaly onset found.")
    print(f"\nSaved model artifacts to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
