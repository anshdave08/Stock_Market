# backend/app/fetcher.py
import pandas as pd
import yfinance as yf
from typing import Tuple

def fetch_daily_yfinance(symbol: str, period: str = "3y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}.")
    # flatten multiindex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open','High','Low','Close','Volume']].copy()
    df = df.apply(pd.to_numeric, errors='coerce')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().dropna()
    return df
