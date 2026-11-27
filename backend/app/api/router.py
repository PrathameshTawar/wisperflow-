from fastapi import APIRouter
from app.api.health import router as health_router
from app.api.vision import router as vision_router
from app.api.sign import router as sign_router
from app.api.speech import router as speech_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(vision_router)
api_router.include_router(sign_router)
api_router.include_router(speech_router)
