# backend/app/predictor.py
import joblib, os, numpy as np
import pandas as pd
from app.config import MODEL_PATH
from app.fetcher import fetch_daily_yfinance
from app.feature import create_technical_features
from app.sentiment import add_batch_sentiment

FEATURES = [
    "Close","Volume",
    "rsi","macd","macd_signal",
    "bb_h","bb_l",
    "ema_20","ema_50",
    "returns_1","returns_2",
    "day_of_week",
    "sent_1d_avg","sent_1d_pos","sent_1d_neg","sent_1d_cnt",
    "sent_2d_avg","sent_2d_pos","sent_2d_neg","sent_2d_cnt",
]

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    raise FileNotFoundError("Model not found. Train first.")

def prepare_full_df(symbol, company_keyword, lookbacks=[1,2], period="3y"):
    df = fetch_daily_yfinance(symbol, period=period)
    df_tech = create_technical_features(df)
    df_with_sent = add_batch_sentiment(df_tech, company_keyword, lookbacks=lookbacks)
    return df_with_sent

def predict_historical(df_with_sent, model=None):
    """Generate predictions for historical data to compare with actual values"""
    if model is None:
        model = load_model()
    
    predictions = []
    for idx in range(len(df_with_sent)):
        row = df_with_sent.iloc[idx].copy()
        for f in FEATURES:
            if f not in row.index:
                row[f] = 0.0
        X_row = pd.DataFrame([row[FEATURES].values], columns=FEATURES)
        pred = float(model.predict(X_row)[0])
        predictions.append(pred)
    
    return predictions

def predict_next(symbol, company_keyword, model=None):
    df_with_sent = prepare_full_df(symbol, company_keyword)
    last_row = df_with_sent.iloc[-1].copy()
    for f in FEATURES:
        if f not in last_row.index:
            last_row[f] = 0.0
    X_last = pd.DataFrame([last_row[FEATURES].values], columns=FEATURES)
    if model is None:
        model = load_model()
    pred = float(model.predict(X_last)[0])
    return {
        "symbol": symbol,
        "last_close": float(df_with_sent['Close'].iloc[-1]),
        "predicted_next_close": pred,
        "predicted_date": (df_with_sent.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "timestamp_utc": pd.Timestamp.now(tz=None).isoformat(),
    }, df_with_sent
