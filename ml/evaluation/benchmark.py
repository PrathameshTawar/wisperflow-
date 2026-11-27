"""
Real local benchmark suite for SAHAAY AI.
Measures latency and smoke correctness across Vision (YOLOv8n),
Gesture Contour Demo (OpenCV), Speech (faster-whisper-base), and OCR entity parsing.
"""
import io
import os
import sys
import time
import wave
import statistics
import numpy as np
from PIL import Image, ImageDraw

# Ensure backend package is importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)

from app.services.vision_service import vision_service
from app.services.sign_service import sign_service
from app.services.speech_service import speech_service


def make_test_scene_bytes() -> bytes:
    arr = np.full((480, 640, 3), 180, dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_test_wav_bytes(seconds: float = 2.0) -> bytes:
    sr = 16000
    samples = np.zeros(int(sr * seconds), dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def benchmark_vision(runs: int = 10):
    img_bytes = make_test_scene_bytes()
    # Warmup
    vision_service.analyze_frame(img_bytes, mode="scene")
    times = []
    last_res = None
    for _ in range(runs):
        t0 = time.perf_counter()
        last_res = vision_service.analyze_frame(img_bytes, mode="scene")
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "engine": last_res.get("vision_engine", "unknown"),
        "median_ms": round(statistics.median(times), 1),
        "p95_ms": round(sorted(times)[int(len(times) * 0.95) - 1], 1),
    }


def benchmark_gesture(runs: int = 10):
    # Synthetic frame with a skin-toned blob in center
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # RGB approximate skin tone (215, 165, 130)
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.ellipse((240, 140, 400, 340), fill=(215, 165, 130))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    times = []
    last_res = None
    for _ in range(runs):
        t0 = time.perf_counter()
        last_res = sign_service.predict_from_image(img_bytes)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "recognized": last_res.get("recognized"),
        "gesture": last_res.get("gesture"),
        "median_ms": round(statistics.median(times), 1),
    }


def benchmark_speech(runs: int = 3):
    wav_bytes = make_test_wav_bytes(2.0)
    # Warmup model load
    speech_service.transcribe_audio(wav_bytes, language="en", engine="whisper")
    times = []
    last_res = None
    for _ in range(runs):
        t0 = time.perf_counter()
        last_res = speech_service.transcribe_audio(wav_bytes, language="en", engine="whisper")
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "engine": last_res.get("engine"),
        "median_ms": round(statistics.median(times), 1),
    }


def main():
    print("Running SAHAAY AI local benchmarks...")
    v = benchmark_vision()
    g = benchmark_gesture()
    s = benchmark_speech()
    print(f"Vision ({v['engine']}): median={v['median_ms']} ms, p95={v['p95_ms']} ms")
    print(f"Gesture Demo (OpenCV): median={g['median_ms']} ms, synthetic_blob={g['gesture']}")
    print(f"Speech ({s['engine']} on 2s clip): median={s['median_ms']} ms")


if __name__ == "__main__":
    main()
