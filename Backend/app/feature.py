# backend/app/features.py
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ['Open','High','Low','Close','Volume']:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}")
    df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    close = df['Close']
    df['rsi'] = RSIIndicator(close, window=14).rsi()
    macd = MACD(close)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    boll = BollingerBands(close)
    df['bb_h'] = boll.bollinger_hband()
    df['bb_l'] = boll.bollinger_lband()
    df['ema_20'] = close.ewm(span=20, adjust=False).mean()
    df['ema_50'] = close.ewm(span=50, adjust=False).mean()
    df['returns_1'] = close.pct_change(1)
    df['returns_2'] = close.pct_change(2)
    df['day_of_week'] = df.index.dayofweek
    df = df.dropna()
    return df
