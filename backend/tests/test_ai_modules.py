import io
import math
import wave
import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SKIN_COLOR = (215, 165, 130)
CREASE_COLOR = (130, 85, 60)


def create_dummy_image_bytes():
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    arr[50:150, 50:150] = [200, 100, 50]
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_landmarks(thumb=False, index=False, middle=False, ring=False, pinky=False):
    pts = [{"x": 0.5, "y": 0.65, "z": 0.0} for _ in range(21)]
    pts[0] = {"x": 0.50, "y": 0.85, "z": 0.0}
    pts[1] = {"x": 0.44, "y": 0.78, "z": 0.0}
    pts[2] = {"x": 0.40, "y": 0.72, "z": 0.0}
    pts[3] = {"x": 0.36, "y": 0.64, "z": 0.0}
    pts[4] = {"x": 0.22, "y": 0.38, "z": 0.0} if thumb else {"x": 0.44, "y": 0.66, "z": 0.0}

    pts[5] = {"x": 0.42, "y": 0.55, "z": 0.0}
    pts[6] = {"x": 0.42, "y": 0.45, "z": 0.0}
    pts[7] = {"x": 0.42, "y": 0.35 if index else 0.53, "z": 0.0}
    pts[8] = {"x": 0.42, "y": 0.25 if index else 0.62, "z": 0.0}

    pts[9] = {"x": 0.50, "y": 0.55, "z": 0.0}
    pts[10] = {"x": 0.50, "y": 0.44, "z": 0.0}
    pts[11] = {"x": 0.50, "y": 0.33 if middle else 0.53, "z": 0.0}
    pts[12] = {"x": 0.50, "y": 0.22 if middle else 0.62, "z": 0.0}

    pts[13] = {"x": 0.58, "y": 0.56, "z": 0.0}
    pts[14] = {"x": 0.58, "y": 0.46, "z": 0.0}
    pts[15] = {"x": 0.58, "y": 0.35 if ring else 0.54, "z": 0.0}
    pts[16] = {"x": 0.58, "y": 0.25 if ring else 0.62, "z": 0.0}

    pts[17] = {"x": 0.65, "y": 0.58, "z": 0.0}
    pts[18] = {"x": 0.65, "y": 0.50, "z": 0.0}
    pts[19] = {"x": 0.65, "y": 0.40 if pinky else 0.56, "z": 0.0}
    pts[20] = {"x": 0.65, "y": 0.30 if pinky else 0.64, "z": 0.0}
    return pts


def _draw_finger(draw: ImageDraw.ImageDraw, base_xy, tip_xy, width_px=24):
    bx, by = base_xy
    tx, ty = tip_xy
    dx, dy = tx - bx, ty - by
    length = max(math.hypot(dx, dy), 1.0)
    nx, ny = -dy / length * (width_px / 2.0), dx / length * (width_px / 2.0)
    poly = [(bx - nx, by - ny), (bx + nx, by + ny), (tx + nx * 0.85, ty + ny * 0.85), (tx - nx * 0.85, ty - ny * 0.85)]
    draw.polygon(poly, fill=SKIN_COLOR)
    r = int(width_px * 0.42)
    draw.ellipse((tx - r, ty - r, tx + r, ty + r), fill=SKIN_COLOR)


def make_synthetic_hand_image(pose: str) -> bytes:
    """Generates clean skin-toned silhouettes with natural fanned fingers and palm creases."""
    img = Image.new("RGB", (640, 480), (20, 25, 35))
    draw = ImageDraw.Draw(img)

    if pose == "YES":
        # Compact square fist
        draw.rounded_rectangle((250, 170, 390, 320), radius=24, fill=SKIN_COLOR)
        for y_line in (210, 240, 270):
            draw.line((270, y_line, 370, y_line), fill=CREASE_COLOR, width=4)
    elif pose == "NO":
        # Tall slender vertical index finger + narrow palm base
        draw.rounded_rectangle((285, 220, 355, 360), radius=15, fill=SKIN_COLOR)
        draw.rounded_rectangle((302, 75, 338, 250), radius=16, fill=SKIN_COLOR)
        for y_line in (260, 290, 320):
            draw.line((295, y_line, 345, y_line), fill=CREASE_COLOR, width=3)
    elif pose == "GOOD":
        # Wide horizontal thumb-out silhouette (aspect > 1.2)
        draw.rounded_rectangle((170, 195, 450, 330), radius=25, fill=SKIN_COLOR)
        for x_line in (230, 290, 350):
            draw.line((x_line, 220, x_line, 305), fill=CREASE_COLOR, width=4)
    elif pose in ("THANK_YOU", "HELP", "WATER", "NAMASTE", "STOP"):
        # Palm base with natural rounded top arc
        draw.rounded_rectangle((245, 245, 395, 385), radius=30, fill=SKIN_COLOR)
        for y_line in (285, 320, 350):
            draw.line((270, y_line, 370, y_line), fill=CREASE_COLOR, width=4)

        if pose == "THANK_YOU":
            # Narrow 2-finger V-pose (< 55 deg valley)
            _draw_finger(draw, (295, 255), (275, 95), width_px=24)
            _draw_finger(draw, (340, 255), (365, 95), width_px=24)
        elif pose == "HELP":
            # Wide L-pose (thumb sideways + upright index -> angle >= 55 deg)
            _draw_finger(draw, (335, 255), (335, 75), width_px=28)
            _draw_finger(draw, (255, 295), (125, 275), width_px=30)
        elif pose == "WATER":
            # 3 fanned fingers (W-pose, middle taller than neighbors -> 2 distinct defects)
            _draw_finger(draw, (265, 260), (220, 115), width_px=24)
            _draw_finger(draw, (315, 250), (315, 70), width_px=24)
            _draw_finger(draw, (365, 260), (415, 115), width_px=24)
        elif pose == "NAMASTE":
            # 4 fanned fingers with natural length arc -> 3 distinct defects
            _draw_finger(draw, (260, 265), (210, 125), width_px=22)
            _draw_finger(draw, (298, 252), (282, 75), width_px=22)
            _draw_finger(draw, (340, 255), (355, 88), width_px=22)
            _draw_finger(draw, (378, 270), (430, 145), width_px=20)
        elif pose == "STOP":
            # 5 fanned fingers including angled thumb -> 4 distinct defects
            _draw_finger(draw, (250, 305), (135, 215), width_px=26)
            _draw_finger(draw, (268, 260), (225, 110), width_px=22)
            _draw_finger(draw, (306, 250), (300, 68), width_px=22)
            _draw_finger(draw, (344, 255), (370, 88), width_px=22)
            _draw_finger(draw, (380, 272), (438, 148), width_px=20)
    elif pose == "UNSUPPORTED_TOO_MANY_DEFECTS":
        # 6 fanned rays along a convex arc -> 5 defects -> must be rejected as UNKNOWN
        draw.rounded_rectangle((210, 255, 430, 385), radius=24, fill=SKIN_COLOR)
        for y_line in (295, 335):
            draw.line((240, y_line, 400, y_line), fill=CREASE_COLOR, width=4)
        cx, cy = 320, 290
        for ang_deg in (-75, -45, -15, 15, 45, 75):
            rad = math.radians(ang_deg - 90)
            bx = cx + math.cos(rad) * 45
            by = cy + math.sin(rad) * 45
            tx = cx + math.cos(rad) * 210
            ty = cy + math.sin(rad) * 210
            _draw_finger(draw, (bx, by), (tx, ty), width_px=18)

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
    assert data.get("proximity_experimental") is True
    assert "proximity_note" in data
    assert "vision_engine" in data
    # Distance claims must remain qualitative, never guessed meters/feet or false safety claims
    narration_lower = data["narration"].lower()
    assert "meter" not in narration_lower
    assert "metre" not in narration_lower
    assert "path is clear" not in narration_lower
    assert "safe to walk" not in narration_lower


def test_vision_ocr_entity_extraction():
    sample_text = "Paracetamol 500mg\nBatch: B12\nExp: 08/2027\nRs. 45.00\nCaution: Take after food"
    response = client.post(
        "/api/vision/analyze-ocr-text",
        json={"raw_text": sample_text, "confidence": 88.5}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["unclear"] is False
    assert data["is_medical_verification"] is False
    assert "entities" in data
    assert "500mg" in data["entities"]["dosage"]
    assert "08/2027" in data["entities"]["expiry"]
    summary = data["speech_summary"]
    # Raw extracted text must appear before structured dosage/expiry interpretation
    assert summary.index("Paracetamol 500mg") < summary.index("Dosage text:")


def test_vision_ocr_unclear_and_mfg_not_confused_with_expiry():
    # Mfg date must not be mislabeled as Expiry date
    mfg_only = "Ibuprofen 400mg\nMfg: 01/2025"
    r_mfg = client.post("/api/vision/analyze-ocr-text", json={"raw_text": mfg_only, "confidence": 82.0})
    assert r_mfg.status_code == 200
    d_mfg = r_mfg.json()
    assert "expiry" not in d_mfg["entities"]
    assert "mfg" in d_mfg["entities"]
    assert "expiry date was not clearly detected" in d_mfg["speech_summary"].lower()

    # Low-confidence blurry OCR must be flagged unclear instead of guessing medicine details
    r_blur = client.post(
        "/api/vision/analyze-ocr-text",
        json={"raw_text": "Prcm 500", "confidence": 28.0}
    )
    assert r_blur.status_code == 200
    d_blur = r_blur.json()
    assert d_blur["unclear"] is True
    assert d_blur["entities"] == {}
    assert "unclear" in d_blur["speech_summary"].lower()


def test_sign_vocabulary():
    response = client.get("/api/sign/vocabulary")
    assert response.status_code == 200
    data = response.json()
    expected_keys = {"NAMASTE", "THANK_YOU", "HELP", "WATER", "YES", "NO", "GOOD", "STOP"}
    assert expected_keys.issubset(set(data.keys()))


def test_sign_all_eight_landmarks_and_unknown_rejection():
    expected_poses = {
        "YES": dict(),
        "NO": dict(index=True),
        "GOOD": dict(thumb=True),
        "THANK_YOU": dict(index=True, middle=True),
        "HELP": dict(thumb=True, index=True),
        "WATER": dict(index=True, middle=True, ring=True),
        "NAMASTE": dict(index=True, middle=True, ring=True, pinky=True),
        "STOP": dict(thumb=True, index=True, middle=True, ring=True, pinky=True),
    }
    scores = set()
    for expected_label, kwargs in expected_poses.items():
        res = client.post("/api/sign/predict", json={"landmarks": make_landmarks(**kwargs)})
        assert res.status_code == 200
        data = res.json()
        assert data["recognized"] is True, f"Failed for {expected_label}: {data}"
        assert data["gesture"] == expected_label
        assert data["score_type"] == "heuristic_landmarks"
        assert data["confidence"] >= data["threshold"]
        scores.add(data["confidence"])

    # Unsupported poses must NEVER default to HELP
    unsupported_cases = [
        make_landmarks(pinky=True),                         # pinky only
        make_landmarks(index=True, pinky=True),             # rock-on
        make_landmarks(thumb=True, pinky=True),             # shaka
        [{"x": 0.5, "y": 0.4, "z": 0.0}] * 21,              # collapsed flat points
        [{"x": 0.5, "y": 0.4, "z": 0.0}] * 5,               # incomplete landmarks
    ]
    for lm in unsupported_cases:
        res = client.post("/api/sign/predict", json={"landmarks": lm})
        assert res.status_code == 200
        data = res.json()
        assert data["recognized"] is False
        assert data["gesture"] == "UNKNOWN"


def test_sign_all_eight_image_silhouettes_and_unknown_rejection():
    for pose in ("YES", "NO", "GOOD", "THANK_YOU", "HELP", "WATER", "NAMASTE", "STOP"):
        img_bytes = make_synthetic_hand_image(pose)
        res = client.post("/api/sign/detect-frame", files={"file": (f"{pose}.jpg", img_bytes, "image/jpeg")})
        assert res.status_code == 200
        data = res.json()
        assert data["recognized"] is True, f"Image silhouette {pose} failed: {data}"
        assert data["gesture"] == pose
        assert data["score_type"] == "heuristic_contour"
        assert data["confidence"] >= data["threshold"]

    bad_img = make_synthetic_hand_image("UNSUPPORTED_TOO_MANY_DEFECTS")
    r_bad = client.post("/api/sign/detect-frame", files={"file": ("bad_pose.jpg", bad_img, "image/jpeg")})
    assert r_bad.status_code == 200
    d_bad = r_bad.json()
    assert d_bad["recognized"] is False
    assert d_bad["gesture"] == "UNKNOWN"


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

    # Unknown gesture labels must be filtered out instead of falling back to NAMASTE
    r_unk = client.post("/api/sign/sequence-sentence", json={"gestures": ["UNKNOWN", "NO_HAND"]})
    assert r_unk.status_code == 200
    assert r_unk.json()["gestures"] == []
    assert r_unk.json()["sentence_en"] == ""


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
    face_img = Image.new("RGB", (640, 480), (230, 230, 230))
    ImageDraw.Draw(face_img).ellipse((240, 60, 400, 260), fill=SKIN_COLOR)
    buf2 = io.BytesIO()
    face_img.save(buf2, format="JPEG")
    r_face = client.post(
        "/api/sign/detect-frame",
        files={"file": ("face.jpg", buf2.getvalue(), "image/jpeg")}
    )
    assert r_face.status_code == 200
    assert r_face.json()["recognized"] is False
