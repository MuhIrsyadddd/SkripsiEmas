# main.py (Versi Flask - OPTIMIZED & Waktu Tetap)
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from datetime import datetime, timedelta, timezone

# === Inisialisasi Aplikasi Flask ===
app = Flask(__name__)

# === Konfigurasi File ===
SAVED_MODEL_PATH = "emas_saved_model"
DATA_PATH = "Gold Price (2013-2023).csv"

# === Global variables (dimuat sekali saat startup) ===
model = None
scaler = None
last_60_days = None

def load_model_and_data():
    """Memuat model dan data ke dalam variabel global."""
    global model, scaler, last_60_days
    try:
        print("Memuat model dan data Emas dari SavedModel...")
        model = tf.saved_model.load(SAVED_MODEL_PATH)
        
        df = pd.read_csv(DATA_PATH)
        df['Price'] = df['Price'].str.replace(',', '').astype(float)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        prices = df[['Price']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(prices)
        
        last_prices = prices[-60:]
        last_60_days = scaler.transform(last_prices)
        print("Model dan data Emas berhasil dimuat.")
    except Exception as e:
        print(f"ERROR saat memuat model atau data Emas: {e}")

# === PERUBAHAN DI SINI: Prediksi Harga Ke Depan ===
def predict_future_prices_optimized(last_60_days_input, days_ahead=360):
    """Prediksi 360 hari ke depan dengan random seed berbasis WIB."""
    
    # Membuat zona waktu WIB (UTC+7)
    wib_timezone = timezone(timedelta(hours=7))
    
    # Mengambil waktu saat ini di UTC, lalu konversi ke WIB
    now_wib = datetime.now(timezone.utc).astimezone(wib_timezone)
    
    # Gunakan tanggal dari zona waktu WIB sebagai seed
    # Ini memastikan seed hanya berubah setelah pukul 00:00 WIB
    date_seed = int(now_wib.strftime("%Y%m%d"))
    np.random.seed(date_seed)
    
    # Proses prediksi tetap sama
    temp_input = list(last_60_days_input.flatten())
    predictions_scaled = []
    
    x_input = np.array(temp_input).reshape(1, -1, 1)
    x_input_tensor = tf.convert_to_tensor(x_input, dtype=tf.float32)

    for _ in range(days_ahead):
        pred_tensor = model.signatures['serving_default'](x_input_tensor)['dense']
        pred_scaled = pred_tensor.numpy()[0][0]
        
        predictions_scaled.append(pred_scaled)
        
        new_input_list = x_input.flatten().tolist()[1:]
        new_input_list.append(pred_scaled)
        x_input = np.array(new_input_list).reshape(1, -1, 1)
        x_input_tensor = tf.convert_to_tensor(x_input, dtype=tf.float32)

    predicted_prices_inv = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
    return predicted_prices_inv.tolist()

# === Fungsi Helper untuk Rekomendasi (Tidak ada perubahan) ===
def get_recommendation(price_list, available_funds):
    if available_funds <= 100.0 or not price_list or len(price_list) < 2:
        return "TUNGGU"
    return "BELI" if price_list[-1] > price_list[0] else "TUNGGU"

# === ENDPOINT API (Tidak ada perubahan) ===
@app.route("/predict/gold/v3", methods=['POST'])
def predict_gold_v3():
    if model is None:
        return jsonify({"error": "Model tidak berhasil dimuat."}), 500
    try:
        data = request.get_json()
        if not data or 'income' not in data or 'expenses' not in data:
            return jsonify({"error": "Harap kirim 'income' dan 'expenses' dalam format JSON."}), 400

        available_funds = float(data['income']) - float(data['expenses'])
        
        future_prices_360_days = predict_future_prices_optimized(last_60_days, days_ahead=360)

        prices_hourly = np.linspace(future_prices_360_days[0], future_prices_360_days[1], 24).tolist()
        prices_monthly = future_prices_360_days[:30]
        prices_yearly = future_prices_360_days

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