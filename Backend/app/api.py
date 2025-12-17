# backend/app/api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
import os
import traceback
import pandas as pd 

from app.predictor import (
    predict_next,
    prepare_full_df,
    predict_historical,
    FEATURES
)
from app.trainer import train_and_save
from app.utils import symbol_to_model_path

router = APIRouter()


# -----------------------------
# REQUEST SCHEMAS
# -----------------------------
class PredictRequest(BaseModel):
    symbol: str
    company_keyword: str | None = None


class TrainRequest(BaseModel):
    symbol: str
    company_keyword: str | None = None


# -----------------------------
# PREDICT ENDPOINT
# -----------------------------
@router.post("/predict")
def predict_endpoint(req: PredictRequest):
    try:
        company_keyword = req.company_keyword or req.symbol.split(".")[0]
        result, _ = predict_next(req.symbol, company_keyword)
        return {"success": True, "result": result}
    except Exception as e:
        print("🔥 PREDICT ERROR 🔥")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# TRAIN ENDPOINT
# -----------------------------
@router.post("/train")
def train_endpoint(req: TrainRequest):
    try:
        company_keyword = req.company_keyword or req.symbol.split(".")[0]
        df_with_sent = prepare_full_df(req.symbol, company_keyword)
        model, rmse, model_path = train_and_save(
            df_with_sent,
            FEATURES,
            symbol=req.symbol
        )
        return {
            "success": True,
            "rmse": rmse,
            "model": os.path.basename(model_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# HISTORY ENDPOINT (WITH PREDICTIONS)
# -----------------------------
@router.get("/history/{symbol}")
def history(symbol: str, company_keyword: str | None = None, period: str = "2y"):
    try:
        company_keyword = company_keyword or symbol.split(".")[0]

        # 1️⃣ Prepare full dataframe
        df = prepare_full_df(symbol, company_keyword, period=period)

        # Ensure datetime index
        df.index = pd.to_datetime(df.index)

        # 2️⃣ Default predicted_close column (ALWAYS set)
        df["predicted_close"] = None

        # 3️⃣ Load model per stock if exists
        model_path = symbol_to_model_path(symbol)
        if os.path.exists(model_path):
            model = joblib.load(model_path)

            preds = predict_historical(df, model)

            valid_idx = df.dropna(subset=FEATURES).index
            df.loc[valid_idx, "predicted_close"] = preds

        # 4️⃣ SAFE reset index → date
        df_out = df.reset_index()
        index_col = df_out.columns[0]
        df_out = df_out.rename(columns={index_col: "date"})
        df_out["date"] = df_out["date"].astype(str)

        return {"success": True, "data": df_out.to_dict(orient="records")}

    except Exception as e:
        # 🔥 REAL ERROR WILL BE SHOWN
        raise HTTPException(status_code=500, detail=str(e))

