import os
import re
from app.config import MODEL_DIR

def symbol_to_model_path(symbol: str) -> str:
    """
    Convert RELIANCE.NS → reliance_ns.pkl
    """
    safe_name = symbol.lower().replace(".", "_")
    return os.path.join(MODEL_DIR, f"{safe_name}.pkl")
