from fastapi import APIRouter
from datetime import datetime, timezone
from app.core.config import settings
from app.services.vision_service import vision_service
from app.services.speech_service import speech_service

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("")
def get_health():
    v_status = vision_service.get_status()
    s_status = speech_service.get_status()
    return {
        "status": "healthy",
        "app": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "api": "online",
            "drishti_vision": v_status["active_engine"],
            "samvaad_sign": "8-gesture-demo",
            "whisper_speech": "sarvam+whisper" if s_status["sarvam_configured"] else s_status["engine"]
        }
    }
