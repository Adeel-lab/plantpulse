from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense

feature1_path = Path("data/raw/ims/feature_df.csv")
feature2_path = Path("data/raw/ims/feature2_df.csv")
feature3_path = Path("data/raw/ims/feature3_df.csv")

features1 = pd.read_csv(feature1_path)
features2 = pd.read_csv(feature2_path)
features3 = pd.read_csv(feature3_path)


def keep_only_numeric(df, name):
    print(f"\n{name} original columns:")
    print(df.dtypes)

    cols_to_drop = []
    for col in df.columns:
        if df[col].dtype == "object":
            cols_to_drop.append(col)

    if cols_to_drop:
        print(f"\n{name} dropping non-numeric columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.fillna(df.median(numeric_only=True))

    print(f"\n{name} final shape: {df.shape}")
    print(df.dtypes)
    return df


features1 = keep_only_numeric(features1, "features1")
features2 = keep_only_numeric(features2, "features2")
features3 = keep_only_numeric(features3, "features3")

common_cols = sorted(set(features1.columns) & set(features2.columns) & set(features3.columns))
print("\nCommon numeric columns:", common_cols)

features1 = features1[common_cols]
features2 = features2[common_cols]
features3 = features3[common_cols]

X1 = features1.values
X2 = features2.values
X3 = features3.values

n_train = int(len(X2) * 0.5)
X2_train = X2[:n_train]
X2_test = X2[n_train:]

scaler = StandardScaler()
X2_train_scaled = scaler.fit_transform(X2_train)
X2_test_scaled = scaler.transform(X2_test)
X3_scaled = scaler.transform(X3)
X1_scaled = scaler.transform(X1)

input_dim = X2_train_scaled.shape[1]

inp = Input(shape=(input_dim,))
x = Dense(32, activation="relu")(inp)
x = Dense(16, activation="relu")(x)
latent = Dense(8, activation="relu")(x)
x = Dense(16, activation="relu")(latent)
x = Dense(32, activation="relu")(x)
out = Dense(input_dim, activation="linear")(x)

autoencoder = Model(inp, out)
autoencoder.compile(optimizer="adam", loss="mse")

history = autoencoder.fit(
    X2_train_scaled, X2_train_scaled,
    epochs=50, batch_size=32,
    validation_split=0.1, shuffle=True, verbose=1
)

X2_train_pred = autoencoder.predict(X2_train_scaled)
X2_test_pred = autoencoder.predict(X2_test_scaled)
X3_pred = autoencoder.predict(X3_scaled)

train_mse = np.mean(np.square(X2_train_scaled - X2_train_pred), axis=1)
test_mse = np.mean(np.square(X2_test_scaled - X2_test_pred), axis=1)
test3_mse = np.mean(np.square(X3_scaled - X3_pred), axis=1)

threshold = train_mse.mean() + 3 * train_mse.std()

print("\nThreshold:", threshold)
print("Train anomalies:", (train_mse > threshold).sum())
print("Test anomalies:", (test_mse > threshold).sum())
print("Test3 anomalies:", (test3_mse > threshold).sum())
