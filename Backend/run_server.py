#!/usr/bin/env python3
"""
Run script for the FastAPI backend server.
This ensures the correct Python path is set up.
"""
import sys
import os
from pathlib import Path

# Add the Backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

