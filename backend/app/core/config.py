import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel

# Load backend/.env or project root .env if present
_backend_dir = Path(__file__).resolve().parent.parent.parent
load_dotenv(_backend_dir / ".env")
load_dotenv(_backend_dir.parent / ".env")

class Settings(BaseModel):
    PROJECT_NAME: str = "SAHAAY AI"
    VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api"
    CORS_ORIGINS: List[str] = ["*"]
    SARVAM_API_KEY: Optional[str] = os.getenv("SARVAM_API_KEY")

settings = Settings()
