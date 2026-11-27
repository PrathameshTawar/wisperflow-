from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from typing import Optional
import base64
from app.services.vision_service import vision_service

router = APIRouter(prefix="/vision", tags=["Drishti Vision"])

class Base64ImageRequest(BaseModel):
    image_base64: str
    mode: Optional[str] = "scene"
    question: Optional[str] = None
    language: Optional[str] = "en"

class OCRTextRequest(BaseModel):
    raw_text: str

@router.get("/status")
def get_vision_status():
    """Return active vision engine (Gemini/Claude VLM or local YOLOv8n fallback)."""
    return vision_service.get_status()

@router.post("/detect")
async def detect_from_upload(
    file: UploadFile = File(...),
    mode: str = Form("scene"),
    question: Optional[str] = Form(None),
    language: str = Form("en")
):
    """Analyze an uploaded camera image for scene description, objects, or experimental currency."""
    try:
        contents = await file.read()
        return vision_service.analyze_frame(contents, mode=mode, question=question, language=language)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@router.post("/detect-base64")
async def detect_from_base64(payload: Base64ImageRequest):
    """Analyze a base64 encoded video frame from the web client."""
    try:
        img_data = payload.image_base64
        if "," in img_data:
            img_data = img_data.split(",")[1]
        raw_bytes = base64.b64decode(img_data)
        return vision_service.analyze_frame(
            raw_bytes,
            mode=payload.mode or "scene",
            question=payload.question,
            language=payload.language or "en"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

@router.post("/analyze-ocr-text")
def analyze_ocr(payload: OCRTextRequest):
    """Analyze OCR text from documents or medicine labels and extract critical entities."""
    return vision_service.analyze_document_text(payload.raw_text)
