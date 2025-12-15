# backend/app/api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.predictor import predict_next, prepare_full_df
from app.trainer import train_and_save, prepare_dataset
from app.predictor import FEATURES
import joblib
from app.config import MODEL_PATH
import traceback

router = APIRouter()

class PredictRequest(BaseModel):
    symbol: str
    company_keyword: str = None

class TrainRequest(BaseModel):
    symbol: str
    company_keyword: str = None

@router.post("/predict")
def predict_endpoint(req: PredictRequest):
    try:
        company_keyword = req.company_keyword or req.symbol.split('.')[0]
        res, df_with_sent = predict_next(req.symbol, company_keyword)
        return {"success": True, "result": res}
    except Exception as e:
        print("🔥 PREDICT ERROR 🔥")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
def train_endpoint(req: TrainRequest):
    try:
        company_keyword = req.company_keyword or req.symbol.split('.')[0]
        df_with_sent = prepare_full_df(req.symbol, company_keyword)
        model, rmse, path = train_and_save(df_with_sent, FEATURES, symbol=req.symbol)
        return {"success": True, "rmse": rmse, "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{symbol}")
def history(symbol: str, company_keyword: str = None, period: str = "2y"):
    try:
        company_keyword = company_keyword or symbol.split('.')[0]
        df = prepare_full_df(symbol, company_keyword, period=period)
        # reduce to JSON-friendly types
        df2 = df.reset_index().rename(columns={'index':'date'})
        return {"success": True, "data": df2.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
