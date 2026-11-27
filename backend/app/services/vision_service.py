import io
import re
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional

class VisionService:
    def __init__(self):
        self.model = None
        self._init_model()

    def _init_model(self):
        """Attempts to load lightweight YOLOv8n model with graceful fallback."""
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            print("Drishti: YOLOv8n model initialized successfully.")
        except Exception as e:
            print(f"Drishti: YOLO model notice: {e}")
            self.model = None

    def decode_image(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame = np.array(image)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def analyze_frame(self, image_bytes: bytes, mode: str = "scene") -> Dict[str, Any]:
        """
        Analyze a camera frame for objects, spatial directions, lighting, or currency,
        and generate clear natural language narration of what the camera sees.
        mode: 'scene' (default object/scene narration) or 'currency' (INR banknote check).
        """
        frame = self.decode_image(image_bytes)
        height, width, _ = frame.shape
        detections: List[Dict[str, Any]] = []

        # Check ambient brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        is_dark = mean_brightness < 40.0

        # 1. Currency mode
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
                "is_dark": is_dark,
                "dimensions": {"width": width, "height": height}
            }

        # 2. Scene / Object detection via YOLOv8n
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

                    # Spatial orientation
                    if center_x < width * 0.33:
                        position = "on your left"
                        sector = "left"
                    elif center_x > width * 0.67:
                        position = "on your right"
                        sector = "right"
                    else:
                        position = "directly ahead"
                        sector = "center"

                    # Proximity estimation
                    if area_ratio > 0.25:
                        proximity = "very close"
                        is_obstacle = True
                    elif area_ratio > 0.10:
                        proximity = "nearby"
                        is_obstacle = True
                    else:
                        proximity = "in view"
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

                # Sort by largest/closest and highest confidence first
                detections.sort(key=lambda d: (d["area_ratio"], d["confidence"]), reverse=True)
            except Exception as e:
                print(f"YOLO inference error: {e}")

        # 3. Generate natural voice narration describing what it sees
        narration = self._generate_narration(detections, mode="scene", is_dark=is_dark)

        # 4. Proximity distance score (0.0 = far/none, 1.0 = very close)
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
            "is_dark": is_dark,
            "dimensions": {"width": width, "height": height}
        }

    def detect_currency(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Lightweight Indian Rupee (INR) banknote detector using the center ROI
        where a user typically holds a note in front of the camera.
        """
        h, w, _ = frame.shape
        # Analyze center 50% ROI to reduce background interference
        roi = frame[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_mean = float(np.mean(hsv[:, :, 0]))
        s_mean = float(np.mean(hsv[:, :, 1]))
        v_mean = float(np.mean(hsv[:, :, 2]))

        detected_note = None
        if 40 <= h_mean <= 85 and 25 <= s_mean <= 115:
            detected_note = "₹500 Indian Rupee Note"
        elif 12 <= h_mean <= 28 and s_mean > 100:
            detected_note = "₹200 Indian Rupee Note"
        elif 120 <= h_mean <= 155 and 30 <= s_mean <= 130:
            detected_note = "₹100 Indian Rupee Note"
        elif 90 <= h_mean <= 118 and s_mean > 75:
            detected_note = "₹50 Indian Rupee Note"
        elif 29 <= h_mean <= 45 and s_mean > 85:
            detected_note = "₹20 Indian Rupee Note"
        elif 5 <= h_mean <= 15 and 50 <= s_mean <= 140 and v_mean < 145:
            detected_note = "₹10 Indian Rupee Note"

        if detected_note:
            return {
                "label": detected_note,
                "confidence": 0.85,
                "bbox": [int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)],
                "position": "directly in front of camera",
                "sector": "center",
                "proximity": "very close",
                "is_obstacle": False,
                "is_currency": True,
                "area_ratio": 0.36
            }
        return None

    def analyze_document_text(self, raw_text: str) -> Dict[str, Any]:
        """
        Extracts medical and document entities from OCR text:
        Dosages, Expiration dates, Warnings, Prices, and Brand labels.
        """
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

        expiry_match = re.search(r'(exp(?:iry)?\.?|mfg\.?|use by)[:\s]*([0-9]{1,2}[\/\-\.][0-9]{2,4})', raw_text, re.IGNORECASE)
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
            preview = " ".join(lines[:3]) if lines else "Document text analyzed."
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
                return f"This looks like a {detections[0]['label']}."
            return "Could not clearly identify an Indian currency note. Hold the note centered in good light."

        if not detections:
            if is_dark:
                return "It is quite dark, and I cannot recognize any objects right now."
            return "I don't recognize any specific objects in front of the camera right now."

        # Describe top 4 objects clearly with spatial direction & closeness
        descriptions = []
        for d in detections[:4]:
            label = d["label"]
            pos = d["position"]
            if d["proximity"] == "very close":
                descriptions.append(f"a {label} {pos}, very close")
            elif d["proximity"] == "nearby":
                descriptions.append(f"a {label} {pos}, nearby")
            else:
                descriptions.append(f"a {label} {pos}")

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
