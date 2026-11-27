# SAHAAY AI — Voice-First Accessibility Bridge for Bharat

A multilingual, voice-first accessibility platform built for blind, visually impaired, deaf, and hard-of-hearing individuals across India.

## The Pain Point

Millions of people in India with visual, hearing, or communication disabilities face daily barriers in markets, public transport, hospitals, and workplaces:
- Blind and low-vision individuals struggle to identify nearby objects, read medicine strips/signboards, or verify Indian Rupee currency notes independently.
- Deaf and hard-of-hearing individuals struggle to communicate in everyday settings with people who do not know Indian Sign Language (ISL).

## Modules & Features

### 1. Drishti (दृष्टि) — AI Vision, Currency & Medicine OCR
- **Continuous Auto-See & Speak:** Live camera feed analyzes scenes every ~2.2 seconds via **YOLOv8n** (`ultralytics` + PyTorch) and speaks spatial descriptions (*"I see a person directly ahead, very close, and a chair on your left"*).
- **Web Audio Sonar Proximity Cues:** Parking-sensor style audio pulses increase in pitch and speed as obstacles get closer.
- **Indian Rupee (INR) Note Detector:** Dedicated currency reader mode for ₹10, ₹20, ₹50, ₹100, ₹200, and ₹500 notes.
- **Medicine & Document OCR:** On-device **Tesseract.js** OCR paired with backend entity extraction for dosage (`500mg`), expiry date (`EXP`), MRP/price (`₹`), and warning labels.
- **Voice Queries:** Ask *"What do you see?"*, *"Kya hai?"*, *"Currency"*, or *"Medicine"* hands-free.

### 2. Samvaad (संवाद) — Indian Sign Language & Conversational Builder
- **Hand Sign Recognition:** Camera-based hand contour and convexity-defect analysis (`OpenCV`) plus 21-landmark prediction endpoint (`POST /api/sign/predict`) supporting core ISL gestures (*Namaste, Thank You, Help, Water, Yes, No, Good, Stop*).
- **Multi-Sign Sentence Builder:** Assembles consecutive signed gestures into bilingual Hindi & English conversational sentences (`POST /api/sign/sequence-sentence`) with audio readout.

### 3. Multilingual Voice Pipeline
- **Sarvam AI + Faster-Whisper STT:** `POST /api/speech/transcribe` decodes browser audio (`webm`/`ogg`/`wav`/`mp4`) in-memory via PyAV (`av`) and routes to **Sarvam AI (`saarika:v2.5`)** when `SARVAM_API_KEY` is configured, with automatic offline fallback to **`faster-whisper-base`** (`int8`).
- **Sarvam AI + Browser TTS:** `POST /api/speech/tts` uses **Sarvam AI (`bulbul:v2`, speaker `anushka`)** for natural Indian-language speech when configured, falling back to browser `window.speechSynthesis`.
- **Accessibility UI:** High-contrast toggle, large-font toggle, English/Hindi/Marathi/Tamil/Telugu/Bengali switcher, PWA manifest, and native mobile camera photo capture (`capture="environment"` / `capture="user"`).

## Tech Stack
- **Frontend:** React 19, TypeScript, Vite, Lucide Icons, Tesseract.js, Web Speech API, Web Audio API
- **Backend:** Python 3.12, FastAPI, Uvicorn, PyTorch, Ultralytics YOLOv8n, OpenCV, faster-whisper, PyAV, Sarvam AI REST API

## Safety & Limits
- **Drishti is an assistance tool only:** It can miss moving vehicles, potholes, or transparent objects (e.g., glass doors) and must **never** replace a white cane, guide dog, or orientation/mobility training.
- **Prototype scope:** Designed as an accessible prototype; lighting, camera angle, and occlusion affect vision and sign accuracy.

## Setup & Run Locally

### 1. Backend (Python 3.12)
```powershell
cd E:\sahaay-ai\backend
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Optional: copy .env.example to .env and set SARVAM_API_KEY
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. Frontend (Node.js)
```powershell
cd E:\sahaay-ai\apps\web
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Open **[http://localhost:5173/](http://localhost:5173/)** (API Swagger docs at **[http://localhost:8000/docs](http://localhost:8000/docs)**).

### 3. Automated Tests
```powershell
cd E:\sahaay-ai\backend
$env:PYTHONPATH = "."
..\.venv\Scripts\pytest.exe -v
```
Current status: **9 / 9 backend tests passing**, frontend production build (`tsc -b && vite build`) verified with 0 errors.
