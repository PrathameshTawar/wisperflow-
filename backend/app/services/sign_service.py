import io
import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional

class SignService:
    def __init__(self):
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(cascade_path) if os.path.exists(cascade_path) else None
        # 8 sample hand gestures for the gesture-to-speech prototype (not full ISL)
        self.vocabulary = {
            "NAMASTE": {
                "phrase_en": "Namaste / Hello",
                "phrase_hi": "नमस्ते (Namaste)",
                "category": "greeting",
                "description": "Both palms pressed together or open palm raised near chest in greeting."
            },
            "THANK_YOU": {
                "phrase_en": "Thank you very much",
                "phrase_hi": "धन्यवाद (Dhanyawad)",
                "category": "politeness",
                "description": "Flat hand moving outward from the chin towards the person."
            },
            "HELP": {
                "phrase_en": "I need help",
                "phrase_hi": "मुझे मदद चाहिए (Mujhe madad chahiye)",
                "category": "emergency",
                "description": "Fist with thumb up supported on the flat palm of the other hand."
            },
            "WATER": {
                "phrase_en": "Water please",
                "phrase_hi": "पानी चाहिए (Paani chahiye)",
                "category": "essential",
                "description": "Three fingers forming 'W' or cupped hand brought near the mouth."
            },
            "YES": {
                "phrase_en": "Yes / Agreed",
                "phrase_hi": "हाँ (Haan)",
                "category": "response",
                "description": "Closed fist nodding up and down like a head nod."
            },
            "NO": {
                "phrase_en": "No / Disagree",
                "phrase_hi": "नहीं (Nahi)",
                "category": "response",
                "description": "Index finger pointing or waving."
            },
            "GOOD": {
                "phrase_en": "Good / Everything is fine",
                "phrase_hi": "बहुत अच्छा (Bahut accha)",
                "category": "expression",
                "description": "Thumbs up raised firmly."
            },
            "STOP": {
                "phrase_en": "Stop / Wait here",
                "phrase_hi": "रुकिए (Rukiye)",
                "category": "safety",
                "description": "Open hand facing forward firmly."
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

    def predict_from_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Gesture-to-speech prototype: masks detected faces first, detects hand skin contour,
        verifies internal texture, and counts convexity defects (extended fingers)
        to map to 8 sample gestures. Not full ISL grammar.
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
                "message": "Hand area too small. Bring hand closer."
            }

        hand_contour = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(hand_contour)
        aspect_ratio = float(w) / h

        # Reject ultra-smooth uniform skin ovals/walls lacking finger/knuckle edge detail inside the contour
        inner_mask = np.zeros_like(mask)
        cv2.drawContours(inner_mask, [hand_contour], -1, 255, -1)
        inner_mask = cv2.erode(inner_mask, kernel, iterations=6)
        interior_lap = cv2.Laplacian(gray, cv2.CV_64F)[inner_mask > 0]
        if interior_lap.size == 0 or float(interior_lap.var()) < 10.0:
            return {
                "recognized": False,
                "gesture": "NO_HAND",
                "phrase_en": "No clear hand gesture detected",
                "phrase_hi": "स्पष्ट हाथ का इशारा नहीं दिखा",
                "confidence": 0.0,
                "message": "Hold your hand clearly in front of the camera."
            }

        hull = cv2.convexHull(hand_contour, returnPoints=False)
        defects_count = 0
        if len(hull) > 3:
            try:
                defects = cv2.convexityDefects(hand_contour, hull)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = hand_contour[s][0]
                        end = hand_contour[e][0]
                        far = hand_contour[f][0]

                        a = np.linalg.norm(end - start)
                        b = np.linalg.norm(far - start)
                        c = np.linalg.norm(end - far)

                        if b * c > 0:
                            cosine_angle = (b**2 + c**2 - a**2) / (2 * b * c)
                            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) * 180 / np.pi
                            if angle <= 85 and d > 3000:
                                defects_count += 1
            except Exception:
                pass

        recognized_key = "HELP"
        confidence = 0.88

        if defects_count >= 3:
            recognized_key = "NAMASTE"
            confidence = 0.94
        elif defects_count == 1:
            recognized_key = "WATER"
            confidence = 0.91
        elif defects_count == 0:
            if aspect_ratio < 0.65:
                recognized_key = "NO"
                confidence = 0.89
            elif aspect_ratio > 1.15:
                recognized_key = "GOOD"
                confidence = 0.92
            else:
                recognized_key = "YES"
                confidence = 0.88
        elif defects_count == 2:
            recognized_key = "THANK_YOU"
            confidence = 0.86

        details = self.vocabulary.get(recognized_key, self.vocabulary["NAMASTE"])

        return {
            "recognized": True,
            "gesture": recognized_key,
            "confidence": confidence,
            "phrase_en": details["phrase_en"],
            "phrase_hi": details["phrase_hi"],
            "category": details["category"],
            "description": details["description"],
            "bbox": [x, y, x + w, y + h],
            "fingers_detected": defects_count + 1 if defects_count > 0 else 1
        }

    def assemble_sentence(self, gesture_sequence: List[str]) -> Dict[str, Any]:
        """
        Takes a sequence of continuous ISL gestures recorded over time
        and composes natural conversational sentences in Hindi and English.
        """
        if not gesture_sequence:
            return {
                "sentence_en": "",
                "sentence_hi": "",
                "gestures": []
            }

        # Filter out duplicates and NO_HAND
        clean_seq = []
        for g in gesture_sequence:
            if g and g != "NO_HAND" and (not clean_seq or clean_seq[-1] != g):
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
        item = self.vocabulary.get(latest, self.vocabulary["NAMASTE"])
        return {
            "sentence_en": item["phrase_en"],
            "sentence_hi": item["phrase_hi"],
            "gestures": clean_seq,
            "is_compound": False
        }

    def predict_from_landmarks(self, landmarks: List[Dict[str, float]]) -> Dict[str, Any]:
        if not landmarks or len(landmarks) < 21:
            return {
                "recognized": False,
                "gesture": "UNKNOWN",
                "message": "Hand landmarks incomplete or no hand detected."
            }

        index_up = landmarks[8]['y'] < landmarks[6]['y']
        middle_up = landmarks[12]['y'] < landmarks[10]['y']
        ring_up = landmarks[16]['y'] < landmarks[14]['y']
        pinky_up = landmarks[20]['y'] < landmarks[18]['y']
        thumb_up = landmarks[4]['y'] < landmarks[3]['y'] and landmarks[4]['y'] < landmarks[8]['y']

        recognized_key = "HELP"
        confidence = 0.90

        if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
            recognized_key = "GOOD"
        elif index_up and middle_up and ring_up and pinky_up:
            recognized_key = "NAMASTE"
        elif index_up and middle_up and not ring_up and not pinky_up:
            recognized_key = "WATER"
        elif not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
            recognized_key = "YES"
        elif index_up and not middle_up and not ring_up and not pinky_up:
            recognized_key = "NO"

        details = self.vocabulary.get(recognized_key, self.vocabulary["NAMASTE"])

        return {
            "recognized": True,
            "gesture": recognized_key,
            "confidence": confidence,
            "phrase_en": details["phrase_en"],
            "phrase_hi": details["phrase_hi"],
            "category": details["category"],
            "description": details["description"],
            "audio_cue": details["phrase_en"]
        }

sign_service = SignService()
