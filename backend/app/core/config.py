import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel

# Load backend/.env or project root .env if present
_backend_dir = Path(__file__).resolve().parent.parent.parent
load_dotenv(_backend_dir / ".env")
load_dotenv(_backend_dir.parent / ".env")

DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://localhost:5173",
    "https://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


def _parse_cors_origins() -> List[str]:
    raw = (os.getenv("CORS_ORIGINS") or "").strip()
    if not raw:
        return DEFAULT_CORS_ORIGINS
    origins = [o.strip() for o in raw.split(",") if o.strip() and o.strip() != "*"]
    return origins or DEFAULT_CORS_ORIGINS


class Settings(BaseModel):
    PROJECT_NAME: str = "SAHAAY AI"
    VERSION: str = "0.2.0"
    API_V1_PREFIX: str = "/api"
    CORS_ORIGINS: List[str] = _parse_cors_origins()
    SARVAM_API_KEY: Optional[str] = os.getenv("SARVAM_API_KEY")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")


settings = Settings()
