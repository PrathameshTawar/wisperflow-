from fastapi import APIRouter
from datetime import datetime, timezone
from app.core.config import settings

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("")
def get_health():
    return {
        "status": "healthy",
        "app": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "api": "online",
            "drishti_vision": "ready",
            "samvaad_sign": "ready",
            "whisper_speech": "ready"
        }
    }
