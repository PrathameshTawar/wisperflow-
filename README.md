# SAHAAY AI — Voice-First Visual Assistant & Gesture Demo

A voice-first accessibility prototype built for blind, low-vision, and speech-impaired users in India.

## Primary Flow: Drishti (दृष्टि) — Voice-First Visual Assistant

- **Voice-First Hands-Free Commands (English, Hindi, Marathi):**
  - *"What do you see?"* / *"क्या है?"* / *"काय आहे?"* — opens Drishti (if needed), starts camera, and immediately captures & describes the scene.
  - *"Read this medicine label"* / *"Read this label"* / *"दवाई पढ़ो"* / *"औषध वाच"* — switches to OCR mode and automatically captures & scans the current frame without touching the screen.
  - *"Repeat"* / *"Repeat that"* / *"दोहराओ"* / *"पुन्हा सांग"* — repeats the last spoken vision or OCR guidance aloud.
  - *"Try again"* / *"Retry"* — repeats the last failed or previous vision/OCR scan.
  - *"Change language to Marathi / Hindi / English"* — switches UI/TTS language and confirms aloud.
  - *"Start camera"* / *"Stop"* — starts camera or stops active cameras and audio playback; commands run only on finalized speech transcripts (never half-spoken interim text), and errors are spoken aloud with recovery hints.
- **Dual Vision Engine:**
  - **Vision-Language Model (Recommended):** Supports **Google Gemini (`gemini-2.5-flash`)** or **Anthropic Claude (`claude-sonnet-4-5`)** when `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` is set in `backend/.env` for full scene, signboard, door, and stair descriptions.
  - **Local Offline Fallback (`YOLOv8n`):** Runs locally when no VLM key is configured (limited to 80 COCO object classes).
- **Experimental Spatial & Proximity Cues:** Reports spatial sectors (`on your left`, `directly ahead`, `on your right`) and approximate size-based closeness (`very close`, `nearby`, `far`) paired with optional Web Audio sonar pulses. Bounding-box area is an experimental visual cue—**never physical distance in meters/feet, and never a claim that a path is clear or safe.**
- **Label & Medicine Reader (OCR):** Uses VLM when configured, or on-device **Tesseract.js** (`eng` with contrast-stretch preprocessing) fallback. Reads raw extracted text aloud **before** structured highlights (`dosage`, `expiry`, `mfg`, `price`, `warning`), flags blurry/low-confidence (`<45%`) text as **unclear** instead of guessing, separates `Mfg` from `Exp`, and explicitly states it is **not a medical verification tool**.
- **Currency Reader (Experimental):** Optional color-heuristic INR note check (`₹10`–`₹500`) explicitly marked experimental—never intended as a sole check for financial transactions.

## Secondary Prototype: Samvaad (संवाद) — Gesture-to-Speech Demo (8 Sample Gestures)

- **Scope & Honesty:** This is an **8-gesture demo prototype**—it is **not** complete Indian Sign Language (ISL) translation (real ISL includes grammar, movement, facial expressions, and context) and does **not** use MediaPipe.
- **How It Works:** Uses **OpenCV** (`YCrCb` skin segmentation + convex hull defect count/angle/solidity/aspect ratio) and a 21-point landmark endpoint (`/api/sign/predict`) to distinguish 8 static sample poses (*Namaste/Hello, Thank You, Help, Water, Yes, No, Good, Stop*).
- **Safety & Confirmation:** Uses computed geometric heuristic scores with a `0.60` confidence threshold, returns **`UNKNOWN` (`"Gesture not recognised. Please try again."`)** for unsupported or low-confidence poses (never defaulting to `HELP`), and defaults to **Confirm Before Speak** (showing the recognized gesture as text before converting it to speech).

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
| **Backend Test Suite (`pytest -v`)** | **15 / 15 passed** (`6.03s`) | API routes, all 8 Samvaad poses (landmarks + synthetic silhouettes) + `UNKNOWN` rejection, OCR uncertainty & `Mfg`/`Exp` separation, safety guards, 400 upload guards, CORS origin restriction |
| **Frontend Lint (`npm run lint`)** | **0 warnings, 0 errors** (`716ms`) | `oxlint` across 3 files / 116 rules |
| **Frontend Build (`npm run build`)** | **0 errors** (`717ms`) | `tsc -b && vite build` (2,152 modules) |
| **YOLOv8n Local Inference Latency (`640×480`)** | **47.5 ms median** (`47.9 ms` p95) | Synthetic speed benchmark (`ml/evaluation/benchmark.py`) on CPU; covers 80 COCO classes |
| **OpenCV Gesture Contour Pipeline (`640×480`)** | **6.3 ms median** | Synthetic speed check (`ml/evaluation/benchmark.py`); rejects smooth skin oval as `NO_HAND` |
| **OCR Entity & Uncertainty Parser** | **1.56 ms** | Clear medicine text parsed; low-confidence (`28%`) text flagged `unclear=True` |
| **Speech Pipeline (`faster-whisper-base`, 2s silent WAV)** | **19.1 ms median** | Local CPU execution benchmark (`ml/evaluation/benchmark.py`); does not measure word error rate |

*Pending Field & User Evaluation (Not Yet Measured):* Real-camera 10-scene accuracy ([`ml/evaluation/scene_eval_log.csv`](ml/evaluation/scene_eval_log.csv)), real medicine-strip photo OCR trials, live spoken-command recognition across Hindi/Marathi/English, repeated real-hand Samvaad trials across lighting/backgrounds, and usability feedback from 3–5 participants (including people with visual or hearing impairments) remain next steps requiring live hardware and human testers.

## Safety & Limitations

- **First-Launch Spoken Consent:** On first open (and whenever triggered from the footer **Safety & Consent Notice** button), the app speaks aloud an *"assistance tool only"* warning plus data-sharing notice and requires tapping **Yes, I Agree** or saying **"Yes"** before capturing camera frames.
- **Drishti is an assistance tool only:** Camera vision can miss potholes, transparent glass doors, or fast-moving vehicles. It must **never** replace a white cane, guide dog, or orientation/mobility training.
- **Camera & Audio Privacy:** When `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, or `SARVAM_API_KEY` is configured in `backend/.env`, captured camera frames are sent to Google or Anthropic servers for scene narration, and audio clips are sent to Sarvam AI servers for speech processing. Avoid pointing the camera at sensitive documents or other people's faces while VLM mode is enabled.
- **No external API keys committed:** Without `GEMINI_API_KEY` or `SARVAM_API_KEY` in `backend/.env`, the app runs entirely locally using `YOLOv8n` + `faster-whisper-base` + browser speech synthesis.

## Demo Video, Screenshots & Wispr Flow Note (To Fill After Recording)

- **3-Minute Demo Video:** *[Add recording link after completing 10-scene + medicine + user tests]*
- **Screenshots:** *[Add Drishti live narration and spoken-question screenshots here]*
- **Wispr Flow Workflow:** *[Add genuine note on how Wispr Flow was used during prompt dictation and testing, with screenshots]*

## Run Locally (macOS, Linux & Windows)

Requires **Python 3.12+** and **Node.js 20+**.

```bash
# 1. Clone and enter project directory
git clone <your-repo-url>
cd sahaay-ai   # or cloned folder name

# 2. Backend setup (CPU PyTorch keeps the install compact vs CUDA wheels)
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt

# Optional: configure Gemini/Anthropic VLM and Sarvam AI keys
cp backend/.env.example backend/.env   # Windows: copy backend\.env.example backend\.env

# Run backend API (http://localhost:8000)
PYTHONPATH=backend uvicorn app.main:app --host 0.0.0.0 --port 8000
# Windows PowerShell: $env:PYTHONPATH="backend"; uvicorn app.main:app --host 0.0.0.0 --port 8000
```

In a second terminal, start the frontend (`http://localhost:5173`):

```bash
cd apps/web
npm ci
npm run dev -- --host 0.0.0.0 --port 5173
```

Run automated tests and lint checks:

```bash
# Backend tests
cd backend && PYTHONPATH=. pytest -v

# Frontend lint + production build
cd apps/web && npm run lint && npm run build

# Optional CPU speed benchmark
python ml/evaluation/benchmark.py
```
