# main.py (Versi Flask)
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime

# === Inisialisasi Aplikasi Flask ===
app = Flask(__name__)

# === Konfigurasi File ===
MODEL_PATH = "EMAS_model.h5"
DATA_PATH = "Gold Price (2013-2023).csv"

# === Global variables untuk model dan data (dimuat sekali saat startup) ===
model = None
scaled_prices = None
scaler = None
last_60_days = None

def load_model_and_data():
    """Memuat model dan data ke dalam variabel global."""
    global model, scaled_prices, scaler, last_60_days
    try:
        print("Memuat model dan data Emas...")
        model = load_model(MODEL_PATH)
        
        df = pd.read_csv(DATA_PATH)
        df['Price'] = df['Price'].str.replace(',', '').astype(float)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        prices = df[['Price']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)
        
        last_60_days = scaled_prices[-60:]
        print("Model dan data Emas berhasil dimuat.")
    except Exception as e:
        print(f"ERROR saat memuat model atau data Emas: {e}")

# === Prediksi Harga Ke Depan (Tidak ada perubahan) ===
def predict_future_prices(model, last_60_days_input, scaler_input, days_ahead=360):
    today = datetime.now().date()
    np.random.seed(int(today.strftime("%Y%m%d")))
    predicted_prices = []
    current_input = last_60_days_input.copy()
    for _ in range(days_ahead):
        input_reshaped = np.reshape(current_input, (1, current_input.shape[0], 1))
        predicted_price = model.predict(input_reshaped, verbose=0)[0][0]
        noise = np.random.normal(loc=0.0, scale=0.01)
        predicted_price = np.clip(predicted_price + noise, 0, 1)
        predicted_prices.append(predicted_price)
        current_input = np.append(current_input[1:], predicted_price)
    predicted_prices_inv = scaler_input.inverse_transform(np.array(predicted_prices).reshape(-1, 1)).flatten()
    return predicted_prices_inv.tolist()

# === Fungsi Helper untuk Rekomendasi (Tidak ada perubahan) ===
def get_recommendation(price_list, available_funds):
    min_fund_threshold = 100.0
    if available_funds <= min_fund_threshold:
        return "TUNGGU" # Diubah agar lebih simpel
    if not price_list or len(price_list) < 2:
        return "TUNGGU"
    return "BELI" if price_list[-1] > price_list[0] else "TUNGGU"

# === ENDPOINT API ===
@app.route("/predict/gold/v3", methods=['POST'])
def predict_gold_v3():
    """Endpoint utama untuk prediksi emas."""
    if model is None:
        return jsonify({"error": "Model tidak berhasil dimuat."}), 500

    try:
        # Mengambil data JSON dari request
        data = request.get_json()
        if not data or 'income' not in data or 'expenses' not in data:
            return jsonify({"error": "Harap kirim 'income' dan 'expenses' dalam format JSON."}), 400

        income = float(data['income'])
        expenses = float(data['expenses'])
        available_funds = income - expenses

        # Prediksi utama
        future_prices_360_days = predict_future_prices(model, last_60_days, scaler, days_ahead=360)

        # Siapkan data untuk setiap periode
        prices_hourly = np.linspace(future_prices_360_days[0], future_prices_360_days[1], 24).tolist() if len(future_prices_360_days) >= 2 else [future_prices_360_days[0]] * 24
        prices_monthly = future_prices_360_days[:30]
        prices_yearly = future_prices_360_days[:360]

        # Buat dictionary untuk respons
        response_data = {
            "harian": {"prices": prices_hourly, "recommendation": get_recommendation(prices_hourly, available_funds)},
            "bulanan": {"prices": prices_monthly, "recommendation": get_recommendation(prices_monthly, available_funds)},
            "tahunan": {"prices": prices_yearly, "recommendation": get_recommendation(prices_yearly, available_funds)}
        }
        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": f"Kesalahan prediksi emas: {str(e)}"}), 500

# === Jalankan loader saat aplikasi pertama kali dimulai ===
with app.app_context():
    load_model_and_data()

# Untuk menjalankan lokal: python main.py
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)