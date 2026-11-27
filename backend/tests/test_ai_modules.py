import io
import wave
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def create_dummy_image_bytes():
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    arr[50:150, 50:150] = [200, 100, 50]
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_vision_status_and_no_meter_guesses():
    status_res = client.get("/api/vision/status")
    assert status_res.status_code == 200
    s_data = status_res.json()
    assert "active_engine" in s_data

    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/api/vision/detect",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"mode": "scene"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "narration" in data
    assert "proximity_score" in data
    assert "vision_engine" in data
    # Distance claims must remain qualitative, never guessed meters/feet
    narration_lower = data["narration"].lower()
    assert "meter" not in narration_lower
    assert "metre" not in narration_lower

def test_vision_ocr_entity_extraction():
    sample_text = "Paracetamol 500mg\nBatch: B12\nExp: 08/2027\nRs. 45.00\nCaution: Take after food"
    response = client.post(
        "/api/vision/analyze-ocr-text",
        json={"raw_text": sample_text}
    )
    assert response.status_code == 200
    data = response.json()
    assert "entities" in data
    assert "500mg" in data["entities"]["dosage"]
    assert "08/2027" in data["entities"]["expiry"]
    assert "speech_summary" in data

def test_sign_vocabulary():
    response = client.get("/api/sign/vocabulary")
    assert response.status_code == 200
    data = response.json()
    assert "NAMASTE" in data
    assert "THANK_YOU" in data

def test_sign_sentence_composition():
    response = client.post(
        "/api/sign/sequence-sentence",
        json={"gestures": ["NAMASTE", "HELP"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_compound"] is True
    assert "Hello" in data["sentence_en"]
    assert "नमस्ते" in data["sentence_hi"]

def test_speech_status():
    response = client.get("/api/speech/status")
    assert response.status_code == 200
    data = response.json()
    assert "faster-whisper" in data["engine"]
    assert "sarvam_configured" in data
    assert data["sarvam_speaker"] == "anushka"
    assert "hi" in data["supported_languages"]
    assert "mr" in data["supported_languages"]

def test_speech_transcribe_wav():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(16000, dtype=np.int16).tobytes())

    response = client.post(
        "/api/speech/transcribe",
        files={"file": ("silence.wav", buf.getvalue(), "audio/wav")},
        data={"language": "en", "engine": "whisper"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "faster-whisper" in data["engine"]
    assert data["language"] == "en"

def test_speech_tts_endpoint():
    response = client.post(
        "/api/speech/tts",
        json={"text": "Namaste", "lang": "hi-IN"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "provider" in data
    assert "audio_b64" in data

def test_corrupt_uploads_return_400():
    r_vis = client.post("/api/vision/detect", files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")})
    assert r_vis.status_code == 400

    r_sign = client.post("/api/sign/detect-frame", files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")})
    assert r_sign.status_code == 400

    r_aud = client.post("/api/speech/transcribe", files={"file": ("bad.webm", b"not-audio", "audio/webm")})
    assert r_aud.status_code == 400

def test_no_false_positive_currency_or_smooth_face():
    # Solid blue shirt should not trigger ₹50 note
    blue_img = Image.new("RGB", (400, 300), (60, 110, 190))
    buf = io.BytesIO()
    blue_img.save(buf, format="JPEG")
    r_cur = client.post(
        "/api/vision/detect",
        files={"file": ("shirt.jpg", buf.getvalue(), "image/jpeg")},
        data={"mode": "currency"}
    )
    assert r_cur.status_code == 200
    assert r_cur.json()["detections"] == []

    # Smooth skin oval without finger/knuckle texture should not trigger YES
    from PIL import ImageDraw
    face_img = Image.new("RGB", (640, 480), (230, 230, 230))
    ImageDraw.Draw(face_img).ellipse((240, 60, 400, 260), fill=(215, 165, 130))
    buf2 = io.BytesIO()
    face_img.save(buf2, format="JPEG")
    r_face = client.post(
        "/api/sign/detect-frame",
        files={"file": ("face.jpg", buf2.getvalue(), "image/jpeg")}
    )
    assert r_face.status_code == 200
    assert r_face.json()["recognized"] is False
