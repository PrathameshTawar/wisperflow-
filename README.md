# SAHAAY AI — Voice-First Visual Assistant & Gesture Demo

A voice-first accessibility prototype built for blind, low-vision, and speech-impaired users in India.

## Primary Flow: Drishti (दृष्टि) — Voice-First Visual Assistant

- **Ask & Hear What It Sees:** Point your laptop or phone camera and hear continuous spoken descriptions every ~2.2 seconds, or ask spoken questions (*"What is in front of me?"*, *"Is there a door?"*).
- **Dual Vision Engine:**
  - **Vision-Language Model (Recommended):** Supports **Google Gemini (`gemini-2.5-flash`)** or **Anthropic Claude (`claude-sonnet-4-5`)** when `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` is set in `backend/.env` for full scene, signboard, door, and stair descriptions.
  - **Local Offline Fallback (`YOLOv8n`):** Runs locally when no VLM key is configured (limited to 80 COCO object classes).
- **Qualitative Distance Only:** Reports spatial sectors (`on your left`, `directly ahead`, `on your right`) and qualitative proximity (`very close`, `nearby`, `far`) paired with optional Web Audio sonar pulses. **Never guesses exact distances in meters or feet.**
- **Label & Medicine Reader (OCR):** Uses VLM when configured, or on-device **Tesseract.js** OCR fallback with entity parsing (dosage, expiry date, MRP, warnings) and raw scanned text display.
- **Currency Reader (Experimental):** Optional color-heuristic INR note check (`₹10`–`₹500`) explicitly marked experimental—never intended as a sole check for financial transactions.

## Secondary Prototype: Samvaad (संवाद) — Gesture-to-Speech Demo (8 Sample Gestures)

- **Scope & Honesty:** This is an **8-gesture demo prototype**—it is **not** full Indian Sign Language (ISL) grammar recognition and does **not** use MediaPipe.
- **How It Works:** Uses **OpenCV** (`YCrCb` skin segmentation + convex hull finger-counting) to classify 8 sample hand poses (*Namaste/Hello, Thank You, Help, Water, Yes, No, Good, Stop*) into bilingual Hindi and English speech, plus a sample gesture-pair phrase builder.

## Multilingual Voice Pipeline

- **Sarvam AI (When `SARVAM_API_KEY` is set):** Uses `saarika:v2.5` (STT) and `bulbul:v2` speaker `anushka` (TTS) for Hindi, Marathi, Tamil, Telugu, Bengali, and English.
- **Local Fallback (No API key):** Uses **`faster-whisper-base` (`int8`)** via **PyAV** (`av`) + browser `speechSynthesis`. Local `whisper-base` performs best on English; Hindi/Marathi accuracy requires setting `SARVAM_API_KEY`.

## Tech Stack

- **Frontend:** React 19, TypeScript, Vite, Lucide Icons, Tesseract.js, Web Audio API, Web Speech API
- **Backend:** Python 3.12, FastAPI, Uvicorn, PyTorch, Ultralytics YOLOv8n, OpenCV, faster-whisper, PyAV, httpx

## Measured Benchmarks & Verification

Measured locally on this Windows workstation (`Python 3.12.10`, CPU mode):

| Component | Measured Result | Notes |
| :--- | :--- | :--- |
| **Backend Test Suite (`pytest -v`)** | **9 / 9 passed** (`13.68s`) | API routes, OCR entity parser, qualitative distance guard, WAV STT/TTS endpoints |
| **Frontend Build (`tsc -b && vite build`)** | **0 errors** (`1.02s`) | Production TypeScript + Vite bundle |
| **YOLOv8n Local Inference Latency (`640×480`)** | **129.6 ms median** (`161.5 ms` p95) | Speed-only benchmark (`ml/evaluation/benchmark.py`) on CPU; covers 80 COCO classes |
| **Tesseract.js OCR (generated test PNGs)** | **594 ms** on 34pt label; tiny 10px print misread `Rs. 45.00` as `Fs 4500` | Synthetic test only — real medicine strip photo testing in progress via `ml/evaluation/scene_eval_log.csv` |

*Field Evaluation (In Progress):* Logging 10 real camera scenes (desk, door, person, chair, bottle, signboard, medicine strip, etc.) in [`ml/evaluation/scene_eval_log.csv`](ml/evaluation/scene_eval_log.csv) alongside user feedback from 3–5 testers.

## Safety & Limitations

- **Drishti is an assistance tool only:** Camera vision can miss potholes, transparent glass doors, or fast-moving vehicles. It must **never** replace a white cane, guide dog, or orientation/mobility training.
- **No external API keys committed:** Without `GEMINI_API_KEY` or `SARVAM_API_KEY` in `backend/.env`, the app runs entirely locally using `YOLOv8n` + `faster-whisper-base` + browser speech synthesis.

## Run Locally

```powershell
# 1. Backend
cd E:\sahaay-ai\backend
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Optional: copy .env.example to .env and add GEMINI_API_KEY + SARVAM_API_KEY
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2. Frontend
cd E:\sahaay-ai\apps\web
npm install
npm run dev -- --host 0.0.0.0 --port 5173

# 3. Tests & Benchmarks
cd E:\sahaay-ai\backend
$env:PYTHONPATH = "."
..\.venv\Scripts\pytest.exe -v
..\.venv\Scripts\python.exe ..\ml\evaluation\benchmark.py
```
