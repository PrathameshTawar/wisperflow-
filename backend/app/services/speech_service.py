import io
import os
import wave
import av
import httpx
import numpy as np
from typing import Dict, Any, Optional
from app.core.config import settings

LANG_TO_SARVAM = {
    "en": "en-IN",
    "hi": "hi-IN",
    "mr": "mr-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "bn": "bn-IN",
}

class SpeechService:
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None
        self.load_error: Optional[str] = None

    def _get_sarvam_key(self) -> Optional[str]:
        key = os.getenv("SARVAM_API_KEY") or settings.SARVAM_API_KEY
        if not key or key.strip() in ("", "your_sarvam_api_key_here"):
            return None
        return key.strip()

    def _to_sarvam_lang(self, language: Optional[str]) -> str:
        if not language or language in ("auto", "unknown"):
            return "unknown"
        if "-" in language:
            return language
        return LANG_TO_SARVAM.get(language.lower(), "hi-IN")

    def _get_model(self):
        """Lazy-loads faster-whisper model on first local transcription request."""
        if self.model is not None:
            return self.model

        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            self.load_error = None
            print(f"SAHAAY Speech: faster-whisper '{self.model_size}' loaded successfully.")
            return self.model
        except Exception as e:
            self.load_error = str(e)
            print(f"SAHAAY Speech load error: {e}")
            raise

    def get_status(self) -> Dict[str, Any]:
        sarvam_active = self._get_sarvam_key() is not None
        return {
            "engine": f"faster-whisper-{self.model_size}",
            "loaded": self.model is not None,
            "sarvam_configured": sarvam_active,
            "sarvam_stt_model": "saarika:v2.5",
            "sarvam_tts_model": "bulbul:v2",
            "sarvam_speaker": "anushka",
            "error": self.load_error,
            "supported_languages": ["en", "hi", "mr", "ta", "te", "bn"]
        }

    def _decode_audio_to_float32(self, audio_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
        """
        Decodes webm/ogg/wav/mp4/mp3 bytes to a 16kHz mono float32 numpy array using PyAV.
        """
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
        chunks = []

        with av.open(io.BytesIO(audio_bytes), mode="r") as container:
            audio_streams = [s for s in container.streams if s.type == "audio"]
            if not audio_streams:
                return np.zeros(0, dtype=np.float32)

            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1))

        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray().reshape(-1))

        if not chunks:
            return np.zeros(0, dtype=np.float32)

        return np.concatenate(chunks).astype(np.float32)

    def _float32_to_wav_bytes(self, audio_np: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Encodes normalized float32 audio array to 16-bit PCM WAV bytes in-memory."""
        clipped = np.clip(audio_np, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()

    def _transcribe_sarvam(self, wav_bytes: bytes, language: Optional[str], api_key: str) -> Dict[str, Any]:
        sarvam_lang = self._to_sarvam_lang(language)
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": api_key},
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={"language_code": sarvam_lang, "model": "saarika:v2.5"},
            )
            resp.raise_for_status()
            payload = resp.json()

        transcript = (payload.get("transcript") or "").strip()
        detected_lang = payload.get("language_code") or sarvam_lang
        return {
            "text": transcript,
            "language": detected_lang.split("-")[0] if detected_lang and detected_lang != "unknown" else "en",
            "language_probability": 1.0,
            "duration": 0.0,
            "engine": "sarvam-saarika:v2.5"
        }

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        engine: str = "auto"
    ) -> Dict[str, Any]:
        """
        Transcribes raw audio bytes (webm/ogg/wav/mp4/mp3).
        Uses Sarvam AI (saarika:v2.5) if configured and engine is 'sarvam' or 'auto',
        with seamless fallback to local faster-whisper.
        """
        if not audio_bytes:
            return {
                "text": "",
                "language": language or "unknown",
                "language_probability": 0.0,
                "duration": 0.0,
                "engine": f"faster-whisper-{self.model_size}"
            }

        audio_np = self._decode_audio_to_float32(audio_bytes, target_sr=16000)
        if audio_np.size == 0:
            return {
                "text": "",
                "language": language or "unknown",
                "language_probability": 0.0,
                "duration": 0.0,
                "engine": f"faster-whisper-{self.model_size}"
            }

        sarvam_key = self._get_sarvam_key()
        if engine in ("auto", "sarvam") and sarvam_key:
            try:
                wav_bytes = self._float32_to_wav_bytes(audio_np, 16000)
                return self._transcribe_sarvam(wav_bytes, language, sarvam_key)
            except Exception as e:
                print(f"Sarvam STT failed, falling back to Whisper: {e}")

        # Local faster-whisper fallback (or explicit 'whisper' engine)
        model = self._get_model()
        lang_arg = None if (not language or language in ("auto", "unknown")) else language.split("-")[0]

        segments, info = model.transcribe(
            audio_np,
            language=lang_arg,
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
        )

        transcript_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        full_text = " ".join(transcript_parts).strip()

        return {
            "text": full_text,
            "language": info.language,
            "language_probability": round(float(info.language_probability), 3),
            "duration": round(float(info.duration), 2),
            "engine": f"faster-whisper-{self.model_size}"
        }

    def synthesize_speech(
        self,
        text: str,
        lang: str = "hi-IN",
        speaker: str = "anushka"
    ) -> Dict[str, Any]:
        """
        Synthesizes natural Indian-language speech via Sarvam AI Bulbul v2 (anushka).
        Returns audio_b64=None and provider='browser-fallback' when SARVAM_API_KEY is unset.
        """
        clean_text = (text or "").strip()[:500]
        if not clean_text:
            return {"audio_b64": None, "provider": "none"}

        sarvam_key = self._get_sarvam_key()
        if not sarvam_key:
            return {
                "audio_b64": None,
                "provider": "browser-fallback",
                "message": "SARVAM_API_KEY not configured; using browser speech synthesis."
            }

        target_lang = self._to_sarvam_lang(lang)
        if target_lang == "unknown":
            target_lang = "hi-IN"

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    "https://api.sarvam.ai/text-to-speech",
                    headers={
                        "api-subscription-key": sarvam_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "inputs": [clean_text],
                        "target_language_code": target_lang,
                        "speaker": speaker.lower(),
                        "model": "bulbul:v2",
                    },
                )
                resp.raise_for_status()
                payload = resp.json()
                audios = payload.get("audios") or []
                if audios:
                    return {
                        "audio_b64": audios[0],
                        "provider": "sarvam-bulbul:v2",
                        "speaker": speaker.lower(),
                        "lang": target_lang
                    }
        except Exception as e:
            print(f"Sarvam TTS error, using browser fallback: {e}")

        return {
            "audio_b64": None,
            "provider": "browser-fallback",
            "message": "Sarvam TTS unavailable; using browser speech synthesis."
        }

speech_service = SpeechService(model_size=os.getenv("WHISPER_MODEL_SIZE", "base"))
