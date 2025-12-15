# backend/app/sentiment.py
import time, json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon', quiet=True)

_sent_cache = Path("./sent_cache")
_sent_cache.mkdir(exist_ok=True)
_vader = SentimentIntensityAnalyzer()

# Optional: NewsAPI client usage (if key present)
def score_texts_vader(texts):
    if not texts:
        return 0.0, 0.0, 0.0, 0
    ssum = 0.0; pos = neg = total = 0
    for t in texts:
        vs = _vader.polarity_scores(t)
        c = vs['compound']
        ssum += c
        if c >= 0.05: pos += 1
        elif c <= -0.05: neg += 1
        total += 1
    avg = ssum / total if total else 0.0
    return avg, (pos/total if total else 0.0), (neg/total if total else 0.0), total

def cached_daily_sentiment(keyword, date, window_days=1):
    date = pd.Timestamp(date)
    date_key = (date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fname = _sent_cache / f"{keyword.replace(' ','_')}_{date_key}_w{window_days}.json"
    if fname.exists():
        try:
            return json.loads(fname.read_text(encoding='utf-8'))
        except Exception:
            pass
    # fallback: no news API integration here -> return neutral 0
    out = {"avg": 0.0, "pos_ratio": 0.0, "neg_ratio": 0.0, "count": 0}
    try:
        fname.write_text(json.dumps(out), encoding='utf-8')
    except Exception:
        pass
    time.sleep(0.05)
    return out

def add_batch_sentiment(df_tech, company_keyword, lookbacks=[1,2]):
    df = df_tech.copy()
    df.index = pd.to_datetime(df.index)
    for w in lookbacks:
        avg_col = f"sent_{w}d_avg"
        pos_col = f"sent_{w}d_pos"
        neg_col = f"sent_{w}d_neg"
        cnt_col = f"sent_{w}d_cnt"
        avg_vals = []; pos_vals = []; neg_vals = []; cnt_vals = []
        for dt in df.index:
            res = cached_daily_sentiment(company_keyword, dt, window_days=w)
            avg_vals.append(res.get("avg",0.0))
            pos_vals.append(res.get("pos_ratio",0.0))
            neg_vals.append(res.get("neg_ratio",0.0))
            cnt_vals.append(res.get("count",0))
        df[avg_col] = avg_vals
        df[pos_col] = pos_vals
        df[neg_col] = neg_vals
        df[cnt_col] = cnt_vals
    if "sent_1d_avg" in df.columns:
        df["sent_1d_avg_3d"] = df["sent_1d_avg"].rolling(window=3, min_periods=1).mean()
    return df
