from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services.speech_service import speech_service

router = APIRouter(prefix="/speech", tags=["SAHAAY Speech (Sarvam & Whisper)"])

class TTSRequest(BaseModel):
    text: str
    lang: Optional[str] = "hi-IN"
    speaker: Optional[str] = "anushka"

@router.get("/status")
def get_speech_status():
    """Return status and readiness of Sarvam AI & faster-whisper speech engines."""
    return speech_service.get_status()

@router.post("/transcribe")
async def transcribe_speech(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    engine: str = Form("auto")
):
    """
    Transcribe recorded microphone audio (webm/ogg/wav/mp4) using Sarvam AI (saarika:v2.5)
    when SARVAM_API_KEY is set, with automatic fallback to local faster-whisper.
    """
    try:
        audio_bytes = await file.read()
        result = speech_service.transcribe_audio(audio_bytes, language=language, engine=engine)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech transcription failed: {str(e)}")

@router.post("/tts")
def synthesize_speech(payload: TTSRequest):
    """
    Generate natural Indian-language speech via Sarvam AI Bulbul v2 (speaker: anushka).
    Returns audio_b64=None and provider='browser-fallback' if SARVAM_API_KEY is unset.
    """
    return speech_service.synthesize_speech(
        text=payload.text,
        lang=payload.lang or "hi-IN",
        speaker=payload.speaker or "anushka"
    )
