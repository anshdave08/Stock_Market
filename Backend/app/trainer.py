# backend/app/trainer.py
import joblib, os, numpy as np
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from app.config import MODEL_PATH

def prepare_dataset(df_with_sent, FEATURES):
    df = df_with_sent.copy()
    df['target'] = df['Close'].shift(-1)
    df = df.dropna()
    X = df[FEATURES].copy()
    y = df['target'].astype(float).values
    return X, y, df

def train_and_save(df_with_sent, FEATURES, symbol="symbol"):
    X, y, df = prepare_dataset(df_with_sent, FEATURES)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    model = LGBMRegressor(objective='regression', learning_rate=0.05, n_estimators=10000, num_leaves=31, random_state=42, verbosity=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse')
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = mse ** 0.5

    return model, float(rmse), MODEL_PATH
