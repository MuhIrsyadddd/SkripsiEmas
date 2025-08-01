# main.py
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import uvicorn
from datetime import datetime

# === Konfigurasi File ===
MODEL_PATH = "EMAS_model.h5"
DATA_PATH = "Gold Price (2013-2023).csv"

# === Kelas Permintaan dan Respons API ===
class GoldRequest(BaseModel):
    income: float
    expenses: float

# --- PERUBAHAN STRUKTUR RESPONS ---
class PredictionDetail(BaseModel):
    """Model untuk menampung harga dan rekomendasi per periode."""
    prices: List[float]
    recommendation: str

class GoldResponseV3(BaseModel):
    """Struktur respons baru dengan rekomendasi terpisah."""
    harian: PredictionDetail
    bulanan: PredictionDetail
    tahunan: PredictionDetail

# Model respons lama (untuk endpoint yang usang)
class GoldResponseV2(BaseModel):
    harian: List[float]
    bulanan: List[float]
    tahunan: List[float]
    recommendation: str

# === Load dan Bersihkan Data ===
def load_and_prepare_data(filepath: str):
    df = pd.read_csv(filepath)
    df['Price'] = df['Price'].str.replace(',', '').astype(float)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    prices = df[['Price']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    return scaled_prices, scaler

# === Prediksi Harga Ke Depan ===
def predict_future_prices(model, last_60_days, scaler, days_ahead=360):
    today = datetime.now().date()
    np.random.seed(int(today.strftime("%Y%m%d")))
    predicted_prices = []
    current_input = last_60_days.copy()
    for _ in range(days_ahead):
        input_reshaped = np.reshape(current_input, (1, current_input.shape[0], 1))
        predicted_price = model.predict(input_reshaped, verbose=0)[0][0]
        noise = np.random.normal(loc=0.0, scale=0.01)
        predicted_price = np.clip(predicted_price + noise, 0, 1)
        predicted_prices.append(predicted_price)
        current_input = np.append(current_input[1:], predicted_price)
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1)).flatten()
    return predicted_prices.tolist()

# === Fungsi Helper untuk Rekomendasi ===
def get_recommendation(price_list: List[float], available_funds: float) -> str:
    """Menentukan rekomendasi berdasarkan tren harga dan dana."""
    min_fund_threshold = 100.0
    
    if available_funds <= min_fund_threshold:
        return "TUNGGU - Dana tidak mencukupi"
    
    if not price_list or len(price_list) < 2:
        return "TUNGGU - Data tidak cukup"

    # Cek tren berdasarkan harga pertama dan terakhir dalam list
    is_trend_up = price_list[-1] > price_list[0]
    
    if is_trend_up:
        return "BELI"
    else:
        return "TUNGGU"

# === Inisialisasi API dan Model ===
app = FastAPI(title="Prediksi Harga Emas")
model = None
try:
    model = load_model(MODEL_PATH)
    scaled_prices, scaler = load_and_prepare_data(DATA_PATH)
    last_60_days = scaled_prices[-60:]
except Exception as e:
    print(f"ERROR saat memuat model atau data: {e}")

# === ENDPOINTS API ===

@app.post("/predict/gold/v3", response_model=GoldResponseV3)
def predict_gold_v3(data: GoldRequest):
    """
    Memprediksi harga emas dengan rekomendasi terpisah untuk
    harian (interpolasi 24 jam), bulanan (30 hari), dan tahunan (360 hari).
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model tidak berhasil dimuat.")
    try:
        # 1. Lakukan prediksi utama untuk 360 hari
        future_prices_360_days = predict_future_prices(model, last_60_days, scaler, days_ahead=360)
        available_funds = float(data.income) - float(data.expenses)

        # 2. Siapkan data harga untuk setiap periode
        # Harian (Interpolasi 24 jam)
        prices_hourly_interpolated = []
        if len(future_prices_360_days) >= 2:
            prices_hourly_interpolated = np.linspace(future_prices_360_days[0], future_prices_360_days[1], 24).tolist()
        elif future_prices_360_days:
            prices_hourly_interpolated = [future_prices_360_days[0]] * 24
        
        # Bulanan dan Tahunan
        prices_monthly = future_prices_360_days[:30]
        prices_yearly = future_prices_360_days[:360]

        # 3. Dapatkan rekomendasi untuk setiap periode
        reco_harian = get_recommendation(prices_hourly_interpolated, available_funds)
        reco_bulanan = get_recommendation(prices_monthly, available_funds)
        reco_tahunan = get_recommendation(prices_yearly, available_funds)

        # 4. Susun respons akhir
        response = GoldResponseV3(
            harian=PredictionDetail(prices=prices_hourly_interpolated, recommendation=reco_harian),
            bulanan=PredictionDetail(prices=prices_monthly, recommendation=reco_bulanan),
            tahunan=PredictionDetail(prices=prices_yearly, recommendation=reco_tahunan)
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kesalahan prediksi emas: {str(e)}")


@app.post("/predict/gold/v2", response_model=GoldResponseV2, deprecated=True)
def predict_gold_v2(data: GoldRequest):
    """
    Endpoint ini sudah usang, silakan gunakan /predict/gold/v3
    """
    # Implementasi lama tetap dipertahankan untuk kompatibilitas
    if model is None:
        raise HTTPException(status_code=500, detail="Model tidak berhasil dimuat.")
    future_prices_360_days = predict_future_prices(model, last_60_days, scaler, days_ahead=360)
    prices_hourly_interpolated = []
    if len(future_prices_360_days) >= 2:
        prices_hourly_interpolated = np.linspace(future_prices_360_days[0], future_prices_360_days[1], 24).tolist()
    elif future_prices_360_days:
        prices_hourly_interpolated = [future_prices_360_days[0]] * 24
    prices_monthly = future_prices_360_days[:30]
    prices_yearly = future_prices_360_days[:360]
    is_trend_up = prices_yearly[-1] > prices_yearly[0] if prices_yearly else False
    available_funds = float(data.income) - float(data.expenses)
    min_fund_threshold = 100.0
    if available_funds <= min_fund_threshold:
        recommendation = "TUNGGU - Dana tidak mencukupi"
    elif is_trend_up:
        recommendation = "BELI"
    else:
        recommendation = "TUNGGU"
    return GoldResponseV2(harian=prices_hourly_interpolated, bulanan=prices_monthly, tahunan=prices_yearly, recommendation=recommendation)

# Untuk menjalankan: uvicorn main:app --reload
