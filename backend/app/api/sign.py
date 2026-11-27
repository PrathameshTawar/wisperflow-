from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import base64
from app.services.sign_service import sign_service

router = APIRouter(prefix="/sign", tags=["Samvaad Sign"])

class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float

class SignPredictRequest(BaseModel):
    landmarks: List[LandmarkPoint]

class Base64SignRequest(BaseModel):
    image_base64: str

class GestureSequenceRequest(BaseModel):
    gestures: List[str]

@router.post("/predict")
def predict_sign(request: SignPredictRequest):
    """Predict Indian Sign Language gesture from 21 MediaPipe hand landmarks."""
    lm_list = [{"x": p.x, "y": p.y, "z": p.z} for p in request.landmarks]
    return sign_service.predict_from_landmarks(lm_list)

@router.post("/detect-frame-base64")
async def detect_sign_base64(payload: Base64SignRequest):
    """Detect hand sign from a base64 camera frame streamed from the web browser."""
    try:
        img_data = payload.image_base64
        if "," in img_data:
            img_data = img_data.split(",")[1]
        raw_bytes = base64.b64decode(img_data)
        return sign_service.predict_from_image(raw_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image sign detection error: {str(e)}")

@router.post("/detect-frame")
async def detect_sign_file(file: UploadFile = File(...)):
    """Detect hand sign from an uploaded image file."""
    try:
        contents = await file.read()
        return sign_service.predict_from_image(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File sign processing failed: {str(e)}")

@router.post("/sequence-sentence")
def compose_sentence(payload: GestureSequenceRequest):
    """Assemble a sequence of continuous ISL signs into a fluent conversational sentence."""
    return sign_service.assemble_sentence(payload.gestures)

@router.get("/vocabulary")
def get_vocabulary():
    """Retrieve supported Indian Sign Language vocabulary."""
    return sign_service.vocabulary
