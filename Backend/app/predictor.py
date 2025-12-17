# backend/app/predictor.py
import os
import joblib
import pandas as pd

from app.utils import symbol_to_model_path
from app.trainer import train_and_save
from app.fetcher import fetch_daily_yfinance
from app.feature import create_technical_features
from app.sentiment import add_batch_sentiment


# -----------------------------
# FEATURES (MUST MATCH TRAINING)
# -----------------------------
FEATURES = [
    "Close", "Volume",
    "rsi", "macd", "macd_signal",
    "bb_h", "bb_l",
    "ema_20", "ema_50",
    "returns_1", "returns_2",
    "day_of_week",
    "sent_1d_avg", "sent_1d_pos", "sent_1d_neg", "sent_1d_cnt",
    "sent_2d_avg", "sent_2d_pos", "sent_2d_neg", "sent_2d_cnt",
]


# -----------------------------
# PREPARE FULL DATAFRAME
# -----------------------------
def prepare_full_df(symbol, company_keyword, lookbacks=[1, 2], period="3y"):
    df = fetch_daily_yfinance(symbol, period=period)
    df_tech = create_technical_features(df)
    df_with_sent = add_batch_sentiment(
        df_tech,
        company_keyword,
        lookbacks=lookbacks
    )
    return df_with_sent


# -----------------------------
# LOAD MODEL (PER STOCK)
# -----------------------------
def load_model(symbol):
    model_path = symbol_to_model_path(symbol)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found for {symbol}")
    return joblib.load(model_path)


# -----------------------------
# HISTORICAL PREDICTIONS
# -----------------------------
def predict_historical(df_with_sent, model):
    """
    Predict historical values safely (drop NaNs first)
    """
    df_feat = df_with_sent.copy()

    # 🔥 Drop rows with NaNs in required features
    df_feat = df_feat.dropna(subset=FEATURES)

    if df_feat.empty:
        return []

    X = df_feat[FEATURES].copy()

    preds = model.predict(X)

    # Return as list aligned with df_feat index
    return preds.tolist()


# -----------------------------
# NEXT-DAY PREDICTION (MAIN API)
# -----------------------------
def predict_next(symbol, company_keyword):
    model_path = symbol_to_model_path(symbol)

    # Prepare data
    df_with_sent = prepare_full_df(symbol, company_keyword)

    # 🔥 LOAD OR TRAIN MODEL
    if os.path.exists(model_path):
        print(f"📦 Loading existing model: {model_path}")
        model = joblib.load(model_path)
    else:
        print(f"🧠 Training new model for: {symbol}")
        model, rmse, model_path = train_and_save(
            df_with_sent,
            FEATURES,
            symbol
        )

    # Prepare last row
    last_row = df_with_sent.iloc[-1].copy()
    for f in FEATURES:
        if f not in last_row.index:
            last_row[f] = 0.0

    X_last = pd.DataFrame([last_row[FEATURES]], columns=FEATURES)

    pred = float(model.predict(X_last)[0])

    return {
        "symbol": symbol,
        "last_close": float(last_row["Close"]),
        "predicted_next_close": pred,
        "predicted_date": (
            df_with_sent.index[-1] + pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d"),
        "model_used": os.path.basename(model_path)
    }, df_with_sent