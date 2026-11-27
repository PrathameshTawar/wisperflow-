import base64
import io
import os
import re
import cv2
import httpx
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional
from app.core.config import settings

LANG_NAMES = {
    "en": "English",
    "hi": "simple Hindi (Devanagari or easy spoken Hindi)",
    "mr": "simple Marathi",
    "ta": "simple Tamil",
    "te": "simple Telugu",
    "bn": "simple Bengali",
}

VLM_SCENE_PROMPT = """You are Drishti, a voice-first visual assistant for a blind or low-vision user in India.
Look at this camera frame and answer the user's request concisely so it can be read aloud in 1 to 2 short sentences in {lang_name}.

Strict safety & honesty rules:
1. Describe what you clearly see: people, furniture, doors, stairs, signboards, printed labels, vehicles, or general surroundings.
2. Describe positions using ONLY 'on your left', 'directly ahead', or 'on your right'.
3. Describe closeness using ONLY 'very close', 'nearby', or 'far'. NEVER guess distances in meters or feet.
4. NEVER claim a path is 'clear' or 'safe' — you cannot reliably detect potholes, glass doors, or fast-moving hazards.
5. If the image is blurry, too dark, or unreadable, say so honestly instead of guessing.
{question_line}"""

class VisionService:
    def __init__(self):
        self.model = None
        self._init_model()

    def _init_model(self):
        """Loads lightweight YOLOv8n model (80 COCO classes) for local detection."""
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            print("Drishti: YOLOv8n model initialized successfully.")
        except Exception as e:
            print(f"Drishti: YOLO model notice: {e}")
            self.model = None

    def _get_vlm_provider(self) -> Optional[str]:
        gemini_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or settings.GEMINI_API_KEY or "").strip()
        if gemini_key and gemini_key != "your_gemini_api_key_here":
            return "gemini"
        anthropic_key = (os.getenv("ANTHROPIC_API_KEY") or settings.ANTHROPIC_API_KEY or "").strip()
        if anthropic_key and anthropic_key != "your_anthropic_api_key_here":
            return "anthropic"
        return None

    def get_status(self) -> Dict[str, Any]:
        vlm = self._get_vlm_provider()
        return {
            "yolo_loaded": self.model is not None,
            "vlm_provider": vlm,
            "active_engine": "gemini-2.5-flash" if vlm == "gemini" else ("anthropic-claude" if vlm == "anthropic" else "yolov8n-local")
        }

    def decode_image(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame = np.array(image)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def _describe_with_vlm(
        self,
        frame: np.ndarray,
        question: Optional[str] = None,
        language: str = "en",
        mode: str = "scene"
    ) -> Optional[Dict[str, str]]:
        """Calls Gemini or Anthropic Vision API if configured; returns {text, engine} or None."""
        provider = self._get_vlm_provider()
        if not provider:
            return None

        # Encode compact JPEG (max width 800px) for low-latency VLM call
        h, w, _ = frame.shape
        if w > 800:
            scale = 800.0 / w
            resized = cv2.resize(frame, (800, int(h * scale)))
        else:
            resized = frame
        ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return None
        b64_img = base64.b64encode(buf.tobytes()).decode("ascii")

        lang_name = LANG_NAMES.get(language, "English")
        if mode == "ocr" and not question:
            q_line = "User wants you to read visible text, medicine labels, dosage, expiry date, or signboards clearly."
        elif question:
            q_line = f"User's spoken question: \"{question.strip()}\""
        else:
            q_line = "User wants to know what is currently in front of the camera."

        prompt = VLM_SCENE_PROMPT.format(lang_name=lang_name, question_line=q_line)

        if provider == "gemini":
            api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or settings.GEMINI_API_KEY or "").strip()
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}}
                    ]
                }],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 180}
            }
            try:
                with httpx.Client(timeout=15.0) as client:
                    resp = client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                candidates = data.get("candidates") or []
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = " ".join(p.get("text", "") for p in parts).strip()
                    if text:
                        return {"text": text, "engine": "gemini-2.5-flash"}
            except Exception as e:
                print(f"Drishti Gemini VLM error, falling back to YOLOv8n: {e}")

        elif provider == "anthropic":
            api_key = (os.getenv("ANTHROPIC_API_KEY") or settings.ANTHROPIC_API_KEY or "").strip()
            try:
                with httpx.Client(timeout=15.0) as client:
                    resp = client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json"
                        },
                        json={
                            "model": "claude-sonnet-4-5",
                            "max_tokens": 180,
                            "messages": [{
                                "role": "user",
                                "content": [
                                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_img}},
                                    {"type": "text", "text": prompt}
                                ]
                            }]
                        }
                    )
                    resp.raise_for_status()
                    data = resp.json()
                blocks = data.get("content") or []
                text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
                if text:
                    return {"text": text, "engine": "anthropic-claude"}
            except Exception as e:
                print(f"Drishti Claude VLM error, falling back to YOLOv8n: {e}")

        return None

    def analyze_frame(
        self,
        image_bytes: bytes,
        mode: str = "scene",
        question: Optional[str] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Analyze a camera frame using VLM (Gemini/Claude when API key configured) + YOLOv8n boxes,
        or local YOLOv8n (80 COCO classes) fallback.
        """
        frame = self.decode_image(image_bytes)
        height, width, _ = frame.shape
        detections: List[Dict[str, Any]] = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        is_dark = mean_brightness < 40.0

        # 1. Experimental Currency mode
        if mode == "currency":
            currency_result = self.detect_currency(frame)
            if currency_result:
                detections.append(currency_result)
            narration = self._generate_narration(detections, mode="currency", is_dark=is_dark)
            return {
                "detections": detections,
                "obstacle_alert": False,
                "proximity_score": 0.0,
                "narration": narration,
                "vision_engine": "experimental-hsv-heuristic",
                "is_dark": is_dark,
                "dimensions": {"width": width, "height": height}
            }

        # 2. Fast local YOLOv8n detection (for bounding box overlay & sonar proximity)
        if self.model is not None:
            try:
                results = self.model(frame, conf=0.28, verbose=False)[0]
                for box in results.boxes:
                    cls_id = int(box.cls[0].item())
                    label = self.model.names[cls_id]
                    confidence = float(box.conf[0].item())
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                    obj_width = x2 - x1
                    obj_height = y2 - y1
                    area_ratio = (obj_width * obj_height) / (width * height)
                    center_x = (x1 + x2) / 2

                    if center_x < width * 0.33:
                        position = "on your left"
                        sector = "left"
                    elif center_x > width * 0.67:
                        position = "on your right"
                        sector = "right"
                    else:
                        position = "directly ahead"
                        sector = "center"

                    if area_ratio > 0.25:
                        proximity = "very close"
                        is_obstacle = True
                    elif area_ratio > 0.10:
                        proximity = "nearby"
                        is_obstacle = True
                    else:
                        proximity = "far"
                        is_obstacle = False

                    detections.append({
                        "label": label,
                        "confidence": round(confidence, 2),
                        "bbox": [x1, y1, x2, y2],
                        "position": position,
                        "sector": sector,
                        "proximity": proximity,
                        "is_obstacle": is_obstacle,
                        "area_ratio": round(area_ratio, 3)
                    })

                detections.sort(key=lambda d: (d["area_ratio"], d["confidence"]), reverse=True)
            except Exception as e:
                print(f"YOLO inference error: {e}")

        # 3. Try Vision-Language Model (full scene, signboard & question answering) if API key present
        vlm_res = self._describe_with_vlm(frame, question=question, language=language, mode=mode)
        if vlm_res:
            narration = vlm_res["text"]
            vision_engine = vlm_res["engine"]
        else:
            narration = self._generate_narration(detections, mode="scene", is_dark=is_dark)
            vision_engine = "yolov8n-local"

        # 4. Proximity score (0.0 = far/none, 1.0 = very close)
        max_proximity_score = 0.0
        if detections:
            obstacle_areas = [d["area_ratio"] for d in detections if d.get("is_obstacle")]
            if obstacle_areas:
                max_proximity_score = min(1.0, max(obstacle_areas) * 3.0)

        return {
            "detections": detections,
            "obstacle_alert": any(d.get("is_obstacle") and d.get("proximity") == "very close" for d in detections),
            "proximity_score": round(max_proximity_score, 2),
            "narration": narration,
            "vision_engine": vision_engine,
            "is_dark": is_dark,
            "dimensions": {"width": width, "height": height}
        }

    def detect_currency(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        EXPERIMENTAL color-heuristic Indian Rupee (INR) estimator using the center ROI.
        Not safe for unverified financial decisions; user must verify via tactile marks.
        """
        h, w, _ = frame.shape
        roi = frame[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_mean = float(np.mean(hsv[:, :, 0]))
        s_mean = float(np.mean(hsv[:, :, 1]))
        v_mean = float(np.mean(hsv[:, :, 2]))

        detected_note = None
        if 40 <= h_mean <= 85 and 25 <= s_mean <= 115:
            detected_note = "₹500 Indian Rupee Note (Experimental Guess)"
        elif 12 <= h_mean <= 28 and s_mean > 100:
            detected_note = "₹200 Indian Rupee Note (Experimental Guess)"
        elif 120 <= h_mean <= 155 and 30 <= s_mean <= 130:
            detected_note = "₹100 Indian Rupee Note (Experimental Guess)"
        elif 90 <= h_mean <= 118 and s_mean > 75:
            detected_note = "₹50 Indian Rupee Note (Experimental Guess)"
        elif 29 <= h_mean <= 45 and s_mean > 85:
            detected_note = "₹20 Indian Rupee Note (Experimental Guess)"
        elif 5 <= h_mean <= 15 and 50 <= s_mean <= 140 and v_mean < 145:
            detected_note = "₹10 Indian Rupee Note (Experimental Guess)"

        if detected_note:
            return {
                "label": detected_note,
                "confidence": 0.65,
                "bbox": [int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)],
                "position": "directly ahead",
                "sector": "center",
                "proximity": "very close",
                "is_obstacle": False,
                "is_currency": True,
                "experimental": True,
                "area_ratio": 0.36
            }
        return None

    def analyze_document_text(self, raw_text: str) -> Dict[str, Any]:
        """Extracts medical and document entities from OCR text."""
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

        expiry_match = re.search(r'(exp(?:iry)?\.?|mfg\.?|use by)[:.\s]*([0-9]{1,2}[\/\-\.][0-9]{2,4})', raw_text, re.IGNORECASE)
        dosage_match = re.search(r'([0-9]+(?:\.[0-9]+)?\s*(?:mg|ml|g|tablets?|capsules?))', raw_text, re.IGNORECASE)
        price_match = re.search(r'(?:rs\.?|inr|₹)\s*([0-9]+(?:\.[0-9]{2})?)', raw_text, re.IGNORECASE)
        warning_match = re.search(r'(caution|warning|schedule [a-z]|keep out of reach).*', raw_text, re.IGNORECASE)

        entities = {}
        if expiry_match:
            entities["expiry"] = expiry_match.group(0)
        if dosage_match:
            entities["dosage"] = dosage_match.group(0)
        if price_match:
            entities["price"] = f"₹{price_match.group(1)}"
        if warning_match:
            entities["warning"] = warning_match.group(0)

        narration_parts = []
        if "dosage" in entities:
            narration_parts.append(f"Dosage identified: {entities['dosage']}.")
        if "expiry" in entities:
            narration_parts.append(f"Expiry date: {entities['expiry']}.")
        if "warning" in entities:
            narration_parts.append(f"Notice: {entities['warning']}.")
        if "price" in entities:
            narration_parts.append(f"Price: {entities['price']}.")

        if not narration_parts:
            preview = " ".join(lines[:3]) if lines else "No clear text found."
            speech_summary = f"Read from document: {preview}"
        else:
            speech_summary = " ".join(narration_parts)

        return {
            "full_text": raw_text,
            "lines": lines,
            "entities": entities,
            "speech_summary": speech_summary
        }

    def _generate_narration(self, detections: List[Dict[str, Any]], mode: str = "scene", is_dark: bool = False) -> str:
        if mode == "currency":
            if detections:
                return f"Experimental color guess: this may be a {detections[0]['label'].replace(' (Experimental Guess)', '')}. Please verify by touch."
            return "Could not match a note color. This currency check is experimental—please verify by touch."

        if not detections:
            if is_dark:
                return "It is quite dark, and the local detector cannot recognize any objects right now."
            return "I don't recognize any of the 80 standard objects in front of the camera right now."

        descriptions = []
        for d in detections[:4]:
            label = d["label"]
            pos = d["position"]
            if d["proximity"] == "very close":
                descriptions.append(f"a {label} {pos}, very close")
            elif d["proximity"] == "nearby":
                descriptions.append(f"a {label} {pos}, nearby")
            else:
                descriptions.append(f"a {label} {pos}, far")

        if len(descriptions) == 1:
            text = f"I see {descriptions[0]}."
        elif len(descriptions) == 2:
            text = f"I see {descriptions[0]} and {descriptions[1]}."
        else:
            text = "I see " + ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}."

        if is_dark:
            text += " Note: lighting is dim."
        return text

vision_service = VisionService()
