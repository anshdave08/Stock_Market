# backend/app/trainer.py
import os
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from app.utils import symbol_to_model_path
from app.db import save_model_metadata

def prepare_dataset(df_with_sent, FEATURES):
    df = df_with_sent.copy()

    # next-day prediction target
    df["target"] = df["Close"].shift(-1)
    df = df.dropna()

    X = df[FEATURES].copy()
    y = df["target"].astype(float).values

    return X, y, df


def train_and_save(df_with_sent, FEATURES, symbol):
    """
    Trains LightGBM model for a specific stock symbol
    and saves it as <symbol>.pkl
    """

    # -----------------------------
    # Prepare data
    # -----------------------------
    X, y, _ = prepare_dataset(df_with_sent, FEATURES)

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # -----------------------------
    # Model
    # -----------------------------
    model = LGBMRegressor(
        objective="regression",
        learning_rate=0.05,
        n_estimators=10000,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse"
    )

    # -----------------------------
    # Save model PER STOCK
    # -----------------------------
    model_path = symbol_to_model_path(symbol)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print(f"💾 Saving model for {symbol} → {model_path}")
    joblib.dump(model, model_path)
    
    # -----------------------------
    # Evaluation
    # -----------------------------
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = mse ** 0.5
    save_model_metadata(symbol, model_path, rmse)
    
    return model, float(rmse), model_path
