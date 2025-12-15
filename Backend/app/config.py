# backend/app/config.py
import os
from dotenv import load_dotenv
load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
MODEL_DIR = os.getenv("MODEL_DIR", "./app/models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, os.getenv("MODEL_FILENAME", "lgbm_model.pkl"))