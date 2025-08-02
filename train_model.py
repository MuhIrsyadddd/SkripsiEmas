import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import os

# === Konfigurasi ===
DATA_PATH = "Gold Price (2013-2023).csv"
SAVED_MODEL_PATH = "emas_saved_model" # Nama direktori untuk format SavedModel
LOOK_BACK = 60

# === 1. Load dan bersihkan data ===
df = pd.read_csv(DATA_PATH)
df['Price'] = df['Price'].str.replace(',', '').astype(float)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# === 2. Normalisasi harga ===
prices = df[['Price']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# === 3. Buat data time series ===
def create_dataset(dataset, look_back=LOOK_BACK):
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i - look_back:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_prices, LOOK_BACK)
X = X.reshape((X.shape[0], X.shape[1], 1))

# === 4. Split data menjadi training dan validation ===
split_index = int(len(X) * 0.9)
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

# === 5. Bangun model LSTM ===
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(units=50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# === 6. Latih model ===
print("Memulai pelatihan model...")
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[EarlyStopping(patience=3)],
    verbose=1
)

# === 7. Simpan model dalam format SavedModel (lebih efisien) ===
# Ganti baris ini
model.export(SAVED_MODEL_PATH)
print(f"Model berhasil disimpan ke direktori: {SAVED_MODEL_PATH}")