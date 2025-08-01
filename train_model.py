import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import math

# === 1. Load dan bersihkan data ===
df = pd.read_csv("Gold Price (2013-2023).csv")
df['Price'] = df['Price'].str.replace(',', '').astype(float)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# === 2. Normalisasi harga ===
prices = df[['Price']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# === 3. Buat data time series ===
def create_dataset(dataset, look_back=60):
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i - look_back:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

look_back = 60
X, y = create_dataset(scaled_prices, look_back)
X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, time steps, features)

# === 4. Split data menjadi training dan validation ===
split_index = int(len(X) * 0.9)
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

# === 5. Bangun model LSTM ===
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# === 6. Latih model ===
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[EarlyStopping(patience=3)],
    verbose=1
)

# === 7. Evaluasi model ===
y_pred = model.predict(X_val)
y_pred_inv = scaler.inverse_transform(y_pred)
y_val_inv = scaler.inverse_transform(y_val.reshape(-1, 1))

mse = mean_squared_error(y_val_inv, y_pred_inv)
rmse = math.sqrt(mse)
mae = mean_absolute_error(y_val_inv, y_pred_inv)
mape = np.mean(np.abs((y_val_inv - y_pred_inv) / y_val_inv)) * 100
accuracy = 100 - mape

print(f"Validation MSE: {mse:.2f}")
print(f"Validation RMSE: {rmse:.2f}")
print(f"Validation MAE: {mae:.2f}")
print(f"Validation MAPE: {mape:.2f}%")
print(f"Model memprediksi harga emas dengan akurasi sekitar {accuracy:.2f}%")

# === 8. Simpan model ===
model.save("EMAS_model.h5")
print("Model berhasil disimpan ke EMAS_model.h5")
