from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb://localhost:27017")
db = client.stock_predictor

def save_model_metadata(symbol, model_path, rmse=None):
    db.models.update_one(
        {"symbol": symbol},
        {"$set": {
            "symbol": symbol,
            "model_path": model_path,
            "rmse": rmse,
            "updated_at": datetime.utcnow()
        }},
        upsert=True
    )
