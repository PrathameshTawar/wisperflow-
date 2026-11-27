from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import base64
from app.services.sign_service import sign_service

router = APIRouter(prefix="/sign", tags=["Samvaad Gesture Demo"])

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
    """Gesture-to-speech demo: map 21 hand landmark points to 8 sample gestures."""
    lm_list = [{"x": p.x, "y": p.y, "z": p.z} for p in request.landmarks]
    return sign_service.predict_from_landmarks(lm_list)

@router.post("/detect-frame-base64")
async def detect_sign_base64(payload: Base64SignRequest):
    """Gesture-to-speech demo: detect 1 of 8 sample hand poses from a base64 camera frame."""
    try:
        img_data = payload.image_base64
        if "," in img_data:
            img_data = img_data.split(",")[1]
        raw_bytes = base64.b64decode(img_data, validate=True)
        return sign_service.predict_from_image(raw_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image gesture detection error: {str(e)}")

@router.post("/detect-frame")
async def detect_sign_file(file: UploadFile = File(...)):
    """Gesture-to-speech demo: detect sample hand gesture from an uploaded image file."""
    try:
        contents = await file.read()
        return sign_service.predict_from_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File gesture processing failed: {str(e)}")

@router.post("/sequence-sentence")
def compose_sentence(payload: GestureSequenceRequest):
    """Assemble sample gesture sequence into a bilingual phrase."""
    return sign_service.assemble_sentence(payload.gestures)

@router.get("/vocabulary")
def get_vocabulary():
    """Retrieve the 8 sample gestures supported by the gesture-to-speech demo."""
    return sign_service.vocabulary
