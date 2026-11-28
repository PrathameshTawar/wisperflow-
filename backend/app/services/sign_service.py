import io
import math
import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional

CONFIDENCE_THRESHOLD = 0.60


class SignService:
    def __init__(self):
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(cascade_path) if os.path.exists(cascade_path) else None
        # 8 sample static hand poses for the gesture-to-speech prototype (not full ISL)
        self.vocabulary = {
            "NAMASTE": {
                "phrase_en": "Namaste / Hello",
                "phrase_hi": "नमस्ते (Namaste)",
                "category": "greeting",
                "description": "Four fingers upright together with thumb tucked (prototype pose for greeting)."
            },
            "THANK_YOU": {
                "phrase_en": "Thank you very much",
                "phrase_hi": "धन्यवाद (Dhanyawad)",
                "category": "politeness",
                "description": "Index and middle fingers raised in a narrow V-pose (prototype pose for thank you)."
            },
            "HELP": {
                "phrase_en": "I need help",
                "phrase_hi": "मुझे मदद चाहिए (Mujhe madad chahiye)",
                "category": "emergency",
                "description": "Thumb and index finger extended wide in an L-pose (prototype pose for help)."
            },
            "WATER": {
                "phrase_en": "Water please",
                "phrase_hi": "पानी चाहिए (Paani chahiye)",
                "category": "essential",
                "description": "Three fingers raised forming 'W' with pinky and thumb tucked."
            },
            "YES": {
                "phrase_en": "Yes / Agreed",
                "phrase_hi": "हाँ (Haan)",
                "category": "response",
                "description": "Closed compact fist with all fingers curled."
            },
            "NO": {
                "phrase_en": "No / Disagree",
                "phrase_hi": "नहीं (Nahi)",
                "category": "response",
                "description": "Single index finger pointing upright."
            },
            "GOOD": {
                "phrase_en": "Good / Everything is fine",
                "phrase_hi": "बहुत अच्छा (Bahut accha)",
                "category": "expression",
                "description": "Thumbs up with other fingers curled."
            },
            "STOP": {
                "phrase_en": "Stop / Wait here",
                "phrase_hi": "रुकिए (Rukiye)",
                "category": "safety",
                "description": "Open palm with all five fingers spread wide."
            }
        }

        # Multi-sign conversational grammar rules
        self.compound_rules = {
            ("NAMASTE", "HELP"): {
                "sentence_en": "Hello, I need your assistance.",
                "sentence_hi": "नमस्ते, मुझे आपकी सहायता चाहिए।"
            },
            ("WATER", "HELP"): {
                "sentence_en": "Please give me drinking water urgently.",
                "sentence_hi": "कृपया मुझे तुरंत पीने का पानी दें।"
            },
            ("YES", "THANK_YOU"): {
                "sentence_en": "Yes, thank you very much.",
                "sentence_hi": "हाँ, आपका बहुत-बहुत धन्यवाद।"
            },
            ("NO", "STOP"): {
                "sentence_en": "No, please wait right there.",
                "sentence_hi": "नहीं, कृपया वहीं रुकिए।"
            },
            ("GOOD", "THANK_YOU"): {
                "sentence_en": "Everything is great, thank you!",
                "sentence_hi": "सब कुछ बहुत अच्छा है, धन्यवाद!"
            },
            ("HELP", "STOP"): {
                "sentence_en": "Emergency! Please halt and assist.",
                "sentence_hi": "आपातकाल! कृपया रुकें और मदद करें।"
            }
        }

    def decode_image(self, image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise ValueError("Empty image payload.")
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            frame = np.array(image)
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Invalid or unsupported image file: {e}")

    def _unknown_response(self, confidence: float, message: str = "Gesture not recognised. Please try again.", score_type: str = "heuristic_contour") -> Dict[str, Any]:
        return {
            "recognized": False,
            "gesture": "UNKNOWN",
            "confidence": round(max(0.0, min(confidence, 0.99)), 2),
            "score_type": score_type,
            "threshold": self.confidence_threshold,
            "phrase_en": "Gesture not recognised. Please try again.",
            "phrase_hi": "इशारा समझ नहीं आया। कृपया फिर से कोशिश करें।",
            "category": "unknown",
            "description": "Hand pose did not match any of the 8 supported sample gestures above the confidence threshold.",
            "message": message,
        }

    def _recognized_response(
        self,
        gesture_key: str,
        confidence: float,
        score_type: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        details = self.vocabulary[gesture_key]
        payload: Dict[str, Any] = {
            "recognized": True,
            "gesture": gesture_key,
            "confidence": round(max(self.confidence_threshold, min(confidence, 0.96)), 2),
            "score_type": score_type,
            "threshold": self.confidence_threshold,
            "phrase_en": details["phrase_en"],
            "phrase_hi": details["phrase_hi"],
            "category": details["category"],
            "description": details["description"],
        }
        if extra:
            payload.update(extra)
        return payload

    def predict_from_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Gesture-to-speech prototype: masks detected faces first, detects hand skin contour,
        verifies internal texture, and uses convexity defects, defect angles, solidity, and aspect
        ratio to distinguish 8 sample poses (or return UNKNOWN below threshold). Not full ISL.
        """
        frame = self.decode_image(image_bytes)
        height, width, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # Mask out detected faces (and chin/neck slightly below) so user's face is not misread as a fist
        if self.face_cascade is not None and not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(40, 40))
            for (fx, fy, fw, fh) in faces:
                x1 = max(0, int(fx - fw * 0.15))
                y1 = max(0, int(fy - fh * 0.10))
                x2 = min(width, int(fx + fw * 1.15))
                y2 = min(height, int(fy + fh * 1.35))
                mask[y1:y2, x1:x2] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {
                "recognized": False,
                "gesture": "NO_HAND",
                "phrase_en": "No hand detected",
                "phrase_hi": "हाथ नहीं दिखा",
                "confidence": 0.0,
                "score_type": "heuristic_contour",
                "threshold": self.confidence_threshold,
                "message": "Please hold your hand in front of the camera."
            }

        valid_contours = [c for c in contours if cv2.contourArea(c) > (width * height * 0.015)]
        if not valid_contours:
            return {
                "recognized": False,
                "gesture": "NO_HAND",
                "phrase_en": "Move hand closer to camera",
                "phrase_hi": "हाथ को कैमरे के पास लाएँ",
                "confidence": 0.0,
                "score_type": "heuristic_contour",
                "threshold": self.confidence_threshold,
                "message": "Hand area too small. Bring hand closer."
            }

        hand_contour = max(valid_contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(hand_contour))
        x, y, w, h = cv2.boundingRect(hand_contour)
        aspect_ratio = float(w) / max(h, 1)

        hull_pts = cv2.convexHull(hand_contour)
        hull_area = float(cv2.contourArea(hull_pts))
        solidity = contour_area / hull_area if hull_area > 0 else 0.0

        # Reject ultra-smooth uniform skin ovals/walls lacking finger/knuckle edge detail inside the contour
        inner_mask = np.zeros_like(mask)
        cv2.drawContours(inner_mask, [hand_contour], -1, 255, -1)
        inner_mask = cv2.erode(inner_mask, kernel, iterations=6)
        interior_lap = cv2.Laplacian(gray, cv2.CV_64F)[inner_mask > 0]
        lap_var = float(interior_lap.var()) if interior_lap.size > 0 else 0.0
        if interior_lap.size == 0 or lap_var < 10.0:
            return {
                "recognized": False,
                "gesture": "NO_HAND",
                "phrase_en": "No clear hand gesture detected",
                "phrase_hi": "स्पष्ट हाथ का इशारा नहीं दिखा",
                "confidence": 0.0,
                "score_type": "heuristic_contour",
                "threshold": self.confidence_threshold,
                "message": "Hold your hand clearly in front of the camera."
            }

        hull = cv2.convexHull(hand_contour, returnPoints=False)
        defect_angles: List[float] = []
        defect_depths_norm: List[float] = []
        min_defect_depth = max(2500.0, float(h) * 0.12 * 256.0)

        if len(hull) > 3:
            defects = cv2.convexityDefects(hand_contour, hull)
            if defects is not None:
                for s, e, f, d in defects.reshape(-1, 4):
                    start = hand_contour[int(s)][0]
                    end = hand_contour[int(e)][0]
                    far = hand_contour[int(f)][0]

                    # Ignore shallow wrist-edge notches near very bottom of bounding box
                    if far[1] >= y + h * 0.88:
                        continue

                    a = np.linalg.norm(end - start)
                    b = np.linalg.norm(far - start)
                    c = np.linalg.norm(end - far)

                    if b * c > 0:
                        cosine_angle = (b**2 + c**2 - a**2) / (2 * b * c)
                        angle = float(np.arccos(np.clip(cosine_angle, -1.0, 1.0)) * 180.0 / np.pi)
                        if angle <= 105.0 and d > min_defect_depth:
                            defect_angles.append(angle)
                            defect_depths_norm.append(float(d) / max(float(h) * 256.0, 1.0))

        defects_count = len(defect_angles)

        # Reject extreme non-hand aspect ratios, fragmented contours, or >4 defects
        if aspect_ratio < 0.22 or aspect_ratio > 2.6 or solidity < 0.28 or defects_count > 4:
            return self._unknown_response(
                confidence=max(0.15, min(0.45, solidity)),
                score_type="heuristic_contour",
            )

        recognized_key: Optional[str] = None
        score = 0.0
        texture_bonus = min(0.08, (lap_var - 10.0) / 800.0)

        if defects_count == 4:
            if 0.40 <= aspect_ratio <= 1.50:
                recognized_key = "STOP"
                depth_factor = min(0.18, sum(defect_depths_norm) / 4.0 * 0.45)
                score = 0.66 + depth_factor + texture_bonus
        elif defects_count == 3:
            if 0.35 <= aspect_ratio <= 1.30:
                recognized_key = "NAMASTE"
                depth_factor = min(0.16, sum(defect_depths_norm) / 3.0 * 0.40)
                score = 0.66 + depth_factor + texture_bonus
        elif defects_count == 2:
            if 0.35 <= aspect_ratio <= 1.40:
                recognized_key = "WATER"
                depth_factor = min(0.16, sum(defect_depths_norm) / 2.0 * 0.40)
                score = 0.66 + depth_factor + texture_bonus
        elif defects_count == 1:
            valley_angle = defect_angles[0]
            if valley_angle < 55.0 and aspect_ratio < 0.92:
                recognized_key = "THANK_YOU"
                margin = min(0.18, (55.0 - valley_angle) / 100.0)
                score = 0.66 + margin + texture_bonus
            elif valley_angle >= 55.0 or aspect_ratio >= 0.92:
                recognized_key = "HELP"
                angle_margin = max(0.0, (valley_angle - 55.0) / 100.0)
                aspect_margin = max(0.0, (aspect_ratio - 0.92) * 0.25)
                score = 0.64 + min(0.18, angle_margin + aspect_margin) + texture_bonus
        elif defects_count == 0:
            if aspect_ratio < 0.68 and solidity >= 0.45:
                recognized_key = "NO"
                margin = min(0.20, (0.68 - aspect_ratio) * 0.65)
                score = 0.64 + margin + texture_bonus
            elif aspect_ratio > 1.18 and solidity >= 0.45:
                recognized_key = "GOOD"
                margin = min(0.20, (aspect_ratio - 1.18) * 0.40)
                score = 0.64 + margin + texture_bonus
            elif 0.72 <= aspect_ratio <= 1.15 and solidity >= 0.74:
                recognized_key = "YES"
                center_fit = max(0.0, 0.12 - abs(aspect_ratio - 0.93) * 0.4)
                solidity_fit = min(0.10, (solidity - 0.74) * 0.4)
                score = 0.64 + center_fit + solidity_fit + texture_bonus
            else:
                # Ambiguous aspect ratio or low-solidity blob: reject instead of guessing
                score = 0.42

        if recognized_key is None or score < self.confidence_threshold:
            return self._unknown_response(confidence=score, score_type="heuristic_contour")

        return self._recognized_response(
            gesture_key=recognized_key,
            confidence=score,
            score_type="heuristic_contour",
            extra={
                "bbox": [x, y, x + w, y + h],
                "fingers_detected": defects_count + 1 if defects_count > 0 else (0 if recognized_key == "YES" else 1),
            },
        )

    def assemble_sentence(self, gesture_sequence: List[str]) -> Dict[str, Any]:
        """
        Takes a sequence of sample gestures recorded over time
        and composes short phrases in Hindi and English.
        """
        if not gesture_sequence:
            return {
                "sentence_en": "",
                "sentence_hi": "",
                "gestures": []
            }

        # Filter out duplicates, NO_HAND, UNKNOWN, and unrecognized labels
        clean_seq = []
        for g in gesture_sequence:
            if g and g in self.vocabulary and (not clean_seq or clean_seq[-1] != g):
                clean_seq.append(g)

        if not clean_seq:
            return {"sentence_en": "", "sentence_hi": "", "gestures": []}

        # Check for 2-gesture compound rules
        if len(clean_seq) >= 2:
            pair = (clean_seq[-2], clean_seq[-1])
            if pair in self.compound_rules:
                rule = self.compound_rules[pair]
                return {
                    "sentence_en": rule["sentence_en"],
                    "sentence_hi": rule["sentence_hi"],
                    "gestures": clean_seq,
                    "is_compound": True
                }

        # Fallback to single latest gesture
        latest = clean_seq[-1]
        item = self.vocabulary[latest]
        return {
            "sentence_en": item["phrase_en"],
            "sentence_hi": item["phrase_hi"],
            "gestures": clean_seq,
            "is_compound": False
        }

    def predict_from_landmarks(self, landmarks: List[Dict[str, float]]) -> Dict[str, Any]:
        if not landmarks or len(landmarks) < 21:
            return self._unknown_response(
                confidence=0.0,
                message="Hand landmarks incomplete or no hand detected.",
                score_type="heuristic_landmarks",
            )

        wrist = landmarks[0]
        mid_mcp = landmarks[9]
        palm_scale = math.hypot(mid_mcp["x"] - wrist["x"], mid_mcp["y"] - wrist["y"])
        if palm_scale < 1e-4:
            return self._unknown_response(
                confidence=0.0,
                message="Gesture not recognised. Hand landmarks collapsed.",
                score_type="heuristic_landmarks",
            )

        # Normalized vertical extension (positive = tip above PIP in image coordinates)
        idx_margin = (landmarks[6]["y"] - landmarks[8]["y"]) / palm_scale
        mid_margin = (landmarks[10]["y"] - landmarks[12]["y"]) / palm_scale
        ring_margin = (landmarks[14]["y"] - landmarks[16]["y"]) / palm_scale
        pinky_margin = (landmarks[18]["y"] - landmarks[20]["y"]) / palm_scale

        ext_thresh = 0.08
        index_up = idx_margin > ext_thresh
        middle_up = mid_margin > ext_thresh
        ring_up = ring_margin > ext_thresh
        pinky_up = pinky_margin > ext_thresh

        thumb_lat = math.hypot(landmarks[4]["x"] - landmarks[5]["x"], landmarks[4]["y"] - landmarks[5]["y"]) / palm_scale
        thumb_up_margin = (landmarks[3]["y"] - landmarks[4]["y"]) / palm_scale
        thumb_out = (thumb_lat > 0.55) or (thumb_up_margin > 0.18 and landmarks[4]["y"] < landmarks[2]["y"])

        pose_tuple = (thumb_out, index_up, middle_up, ring_up, pinky_up)
        pose_map = {
            (False, False, False, False, False): "YES",
            (False, True, False, False, False): "NO",
            (True, False, False, False, False): "GOOD",
            (False, True, True, False, False): "THANK_YOU",
            (True, True, False, False, False): "HELP",
            (False, True, True, True, False): "WATER",
            (False, True, True, True, True): "NAMASTE",
            (True, True, True, True, True): "STOP",
        }

        recognized_key = pose_map.get(pose_tuple)
        if recognized_key is None:
            return self._unknown_response(confidence=0.35, score_type="heuristic_landmarks")

        # Require clear upward thumb when classifying GOOD
        if recognized_key == "GOOD" and thumb_up_margin <= 0.18:
            return self._unknown_response(confidence=0.45, score_type="heuristic_landmarks")

        finger_margins = [
            abs(idx_margin - ext_thresh),
            abs(mid_margin - ext_thresh),
            abs(ring_margin - ext_thresh),
            abs(pinky_margin - ext_thresh),
            max(abs(thumb_lat - 0.55), abs(thumb_up_margin - 0.18)),
        ]
        min_sep = min(finger_margins)
        avg_sep = sum(finger_margins) / len(finger_margins)
        score = 0.58 + min(0.22, min_sep * 0.45) + min(0.14, avg_sep * 0.20)

        if score < self.confidence_threshold:
            return self._unknown_response(confidence=score, score_type="heuristic_landmarks")

        return self._recognized_response(
            gesture_key=recognized_key,
            confidence=score,
            score_type="heuristic_landmarks",
            extra={"audio_cue": self.vocabulary[recognized_key]["phrase_en"]},
        )


sign_service = SignService()
