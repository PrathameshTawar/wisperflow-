import React, { useState, useEffect, useRef } from 'react';
import { 
  Eye, 
  HandMetal, 
  Mic, 
  MicOff, 
  Volume2, 
  RefreshCw, 
  CheckCircle2, 
  Languages, 
  Sparkles, 
  Maximize2, 
  SunMoon, 
  Layers, 
  ArrowRight,
  Camera,
  CameraOff,
  AlertTriangle,
  Upload,
  Play,
  RotateCcw,
  Smartphone,
  SwitchCamera,
  FileText,
  Radio,
  Pill,
  Trash2
} from 'lucide-react';
import './App.css';

interface HealthData {
  status: string;
  app: string;
  version: string;
  timestamp: string;
  services: {
    api: string;
    drishti_vision: string;
    samvaad_sign: string;
    whisper_speech?: string;
  };
}

interface Detection {
  label: string;
  confidence: number;
  bbox: [number, number, number, number];
  position: string;
  sector: string;
  proximity: string;
  is_obstacle: boolean;
  is_currency?: boolean;
}

interface VisionResponse {
  detections: Detection[];
  obstacle_alert: boolean;
  proximity_score?: number;
  narration: string;
  vision_engine?: string;
  vlm_error?: string | null;
  dimensions: { width: number; height: number };
}

interface OCREntities {
  dosage?: string;
  expiry?: string;
  price?: string;
  warning?: string;
}

interface OCRResponse {
  full_text: string;
  lines: string[];
  entities: OCREntities;
  speech_summary: string;
}

interface SignPrediction {
  recognized: boolean;
  gesture: string;
  confidence: number;
  phrase_en: string;
  phrase_hi: string;
  category: string;
  description: string;
  bbox?: [number, number, number, number];
  fingers_detected?: number;
}

const LANGUAGES = [
  { code: 'en', label: 'English', native: 'English', speechCode: 'en-IN' },
  { code: 'hi', label: 'Hindi', native: 'हिन्दी', speechCode: 'hi-IN' },
  { code: 'ta', label: 'Tamil', native: 'தமிழ்', speechCode: 'ta-IN' },
  { code: 'te', label: 'Telugu', native: 'తెలుగు', speechCode: 'te-IN' },
  { code: 'bn', label: 'Bengali', native: 'বাংলা', speechCode: 'bn-IN' },
  { code: 'mr', label: 'Marathi', native: 'मराठी', speechCode: 'mr-IN' },
];

export function App() {
  const [activeTab, setActiveTab] = useState<'overview' | 'drishti' | 'samvaad'>('overview');
  const [health, setHealth] = useState<HealthData | null>(null);
  const [healthStatus, setHealthStatus] = useState<'loading' | 'online' | 'offline'>('loading');
  const [latency, setLatency] = useState<number | null>(null);
  const [highContrast, setHighContrast] = useState(false);
  const [largeFont, setLargeFont] = useState(false);
  const [selectedLang, setSelectedLang] = useState('en');
  
  // Voice & Mic State (Sarvam AI + Whisper AI + Browser Web Speech)
  const [micEngine, setMicEngine] = useState<'auto' | 'whisper' | 'browser'>('auto');
  const [sarvamAvailable, setSarvamAvailable] = useState(false);
  const [activeSpeechEngine, setActiveSpeechEngine] = useState<string | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [voiceTranscript, setVoiceTranscript] = useState('');
  const [detectedSpeechLang, setDetectedSpeechLang] = useState<string | null>(null);
  const [voiceAnnouncement, setVoiceAnnouncement] = useState('Welcome to SAHAAY AI — India’s voice-first accessibility platform.');
  const [micError, setMicError] = useState<string | null>(null);
  const recognitionRef = useRef<any>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const whisperStopTimerRef = useRef<any>(null);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);
  const isSarvamSpeakingRef = useRef(false);

  // Sonar Proximity Audio
  const [sonarBeepEnabled, setSonarBeepEnabled] = useState(true);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const lastBeepTimeRef = useRef<number>(0);

  // Drishti Vision State
  const [drishtiSubMode, setDrishtiSubMode] = useState<'vision' | 'currency' | 'ocr'>('vision');
  const [autoVisionSpeak, setAutoVisionSpeak] = useState(true);
  const [vlmProvider, setVlmProvider] = useState<string | null>(null);
  const [visionQuestion, setVisionQuestion] = useState('');
  const [isVisionCameraActive, setIsVisionCameraActive] = useState(false);
  const [visionFacingMode, setVisionFacingMode] = useState<'user' | 'environment'>('environment');
  const [visionResult, setVisionResult] = useState<VisionResponse | null>(null);
  const [ocrResult, setOcrResult] = useState<OCRResponse | null>(null);
  const [isAnalyzingVision, setIsAnalyzingVision] = useState(false);
  const [visionCameraError, setVisionCameraError] = useState<string | null>(null);
  const visionVideoRef = useRef<HTMLVideoElement | null>(null);
  const visionCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const isAnalyzingVisionRef = useRef(false);
  const lastSpokenVisionRef = useRef<{ text: string; time: number }>({ text: '', time: 0 });

  // Samvaad Sign State
  const [isSignCameraActive, setIsSignCameraActive] = useState(false);
  const [signFacingMode, setSignFacingMode] = useState<'user' | 'environment'>('user');
  const [signResult, setSignResult] = useState<SignPrediction | null>(null);
  const [gestureSequence, setGestureSequence] = useState<string[]>([]);
  const [composedSentence, setComposedSentence] = useState<{ en: string; hi: string } | null>(null);
  const [isAnalyzingSign, setIsAnalyzingSign] = useState(false);
  const [signCameraError, setSignCameraError] = useState<string | null>(null);
  const [autoSignTrack, setAutoSignTrack] = useState(false);
  const [selectedSimGesture, setSelectedSimGesture] = useState<string>('NAMASTE');
  const signVideoRef = useRef<HTMLVideoElement | null>(null);
  const signCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const signIntervalRef = useRef<any>(null);

  // Accessibility Modes
  useEffect(() => {
    if (highContrast) {
      document.body.classList.add('high-contrast');
    } else {
      document.body.classList.remove('high-contrast');
    }
  }, [highContrast]);

  useEffect(() => {
    if (largeFont) {
      document.body.classList.add('large-font');
    } else {
      document.body.classList.remove('large-font');
    }
  }, [largeFont]);

  // Fetch Health Check from Backend
  const checkHealth = async () => {
    setHealthStatus('loading');
    const startTime = performance.now();
    try {
      const response = await fetch('/api/health');
      if (response.ok) {
        const data: HealthData = await response.json();
        setHealth(data);
        setHealthStatus('online');
        setLatency(Math.round(performance.now() - startTime));
      } else {
        setHealthStatus('offline');
      }
      // Check Sarvam / Whisper Speech & Vision VLM capabilities
      const [speechRes, visionRes] = await Promise.all([
        fetch('/api/speech/status'),
        fetch('/api/vision/status')
      ]);
      if (speechRes.ok) {
        const sData = await speechRes.json();
        setSarvamAvailable(Boolean(sData.sarvam_configured));
      }
      if (visionRes.ok) {
        const vData = await visionRes.json();
        setVlmProvider(vData.vlm_provider || null);
      }
    } catch {
      setHealthStatus('offline');
      setHealth(null);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // Web Audio Sonar Proximity Beep Generator for the Blind
  const playSonarBeep = (proximityScore: number) => {
    if (!sonarBeepEnabled || proximityScore <= 0.05) return;

    try {
      if (!audioCtxRef.current) {
        const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
        audioCtxRef.current = new AudioContextClass();
      }
      const ctx = audioCtxRef.current;
      if (ctx.state === 'suspended') {
        ctx.resume();
      }

      const now = performance.now();
      // Frequency: 300Hz (far) to 850Hz (very close)
      const freq = 300 + (proximityScore * 550);
      // Min interval between beeps: 250ms (emergency) to 1200ms (far)
      const minInterval = Math.max(250, 1200 - (proximityScore * 950));

      if (now - lastBeepTimeRef.current >= minInterval) {
        lastBeepTimeRef.current = now;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();

        osc.type = proximityScore > 0.6 ? 'sawtooth' : 'sine';
        osc.frequency.setValueAtTime(freq, ctx.currentTime);

        gain.gain.setValueAtTime(0.15, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.12);

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start();
        osc.stop(ctx.currentTime + 0.12);
      }
    } catch (e) {
      console.warn("Sonar audio unavailable:", e);
    }
  };

  // Voice Readout Helper (Sarvam AI Bulbul Anushka when key set, with Browser TTS fallback)
  const speakBrowserFallback = (text: string, speechCode = 'en-IN') => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = speechCode;
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  };

  const speakText = async (text: string) => {
    if (!text) return;
    setVoiceAnnouncement(text);

    // Stop any previous browser or Sarvam audio
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }
    isSarvamSpeakingRef.current = false;
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }

    const langConfig = LANGUAGES.find(l => l.code === selectedLang) || LANGUAGES[0];

    if (sarvamAvailable) {
      try {
        isSarvamSpeakingRef.current = true;
        const res = await fetch('/api/speech/tts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text,
            lang: langConfig.speechCode,
            speaker: 'anushka'
          })
        });
        if (res.ok) {
          const data = await res.json();
          if (data.audio_b64) {
            const audio = new Audio(`data:audio/wav;base64,${data.audio_b64}`);
            currentAudioRef.current = audio;
            audio.onended = () => {
              isSarvamSpeakingRef.current = false;
              currentAudioRef.current = null;
            };
            audio.onerror = () => {
              isSarvamSpeakingRef.current = false;
              speakBrowserFallback(text, langConfig.speechCode);
            };
            await audio.play();
            return;
          }
        }
      } catch {
        // Fallback below
      }
      isSarvamSpeakingRef.current = false;
    }

    speakBrowserFallback(text, langConfig.speechCode);
  };

  // Camera Stream Request
  const requestCameraStream = async (facing: 'user' | 'environment') => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("SECURE_CONTEXT_REQUIRED");
    }

    try {
      return await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: facing },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });
    } catch {
      return await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }
  };

  // Drishti Camera Management
  const startVisionCamera = async (facing = visionFacingMode) => {
    setVisionCameraError(null);
    try {
      const stream = await requestCameraStream(facing);
      if (visionVideoRef.current) {
        visionVideoRef.current.srcObject = stream;
        await visionVideoRef.current.play();
        setIsVisionCameraActive(true);
        speakText('Camera activated for Drishti Vision.');
      }
    } catch (err: any) {
      console.error("Camera error:", err);
      let msg = "Could not access live camera.";
      if (err.message === "SECURE_CONTEXT_REQUIRED") {
        msg = "Mobile live video stream is blocked by browser security over HTTP. Tap 'Take Phone Camera Photo' below to use your phone's camera directly!";
      } else if (err.name === 'NotAllowedError') {
        msg = "Camera permission was denied. Please allow camera permissions in your browser.";
      } else if (err.name === 'NotFoundError') {
        msg = "No camera found. Use 'Take Phone Camera Photo' or 'Upload Image'.";
      } else if (err.name === 'NotReadableError') {
        msg = "Camera is already in use by another app. Please close other camera apps and retry.";
      }
      setVisionCameraError(msg);
      speakText(msg);
    }
  };

  const stopVisionCamera = () => {
    if (visionVideoRef.current && visionVideoRef.current.srcObject) {
      const stream = visionVideoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(t => t.stop());
      visionVideoRef.current.srcObject = null;
    }
    setIsVisionCameraActive(false);
    setVisionResult(null);
    speakText('Drishti camera stopped.');
  };

  const flipVisionCamera = async () => {
    const nextMode = visionFacingMode === 'environment' ? 'user' : 'environment';
    setVisionFacingMode(nextMode);
    stopVisionCamera();
    setTimeout(() => startVisionCamera(nextMode), 300);
  };

  // Samvaad Camera Management
  const startSignCamera = async (facing = signFacingMode) => {
    setSignCameraError(null);
    try {
      const stream = await requestCameraStream(facing);
      if (signVideoRef.current) {
        signVideoRef.current.srcObject = stream;
        await signVideoRef.current.play();
        setIsSignCameraActive(true);
        speakText('Camera activated for Samvaad Sign Language.');
      }
    } catch (err: any) {
      console.error("Sign camera error:", err);
      let msg = "Could not access live camera.";
      if (err.message === "SECURE_CONTEXT_REQUIRED") {
        msg = "Mobile live video is blocked by browser security over HTTP. Tap 'Snap Hand Photo with Phone' below to take a photo directly!";
      } else if (err.name === 'NotAllowedError') {
        msg = "Camera permission was denied. Please allow camera access in your browser.";
      }
      setSignCameraError(msg);
      speakText(msg);
    }
  };

  const stopSignCamera = () => {
    if (signIntervalRef.current) {
      clearInterval(signIntervalRef.current);
      signIntervalRef.current = null;
    }
    setAutoSignTrack(false);
    if (signVideoRef.current && signVideoRef.current.srcObject) {
      const stream = signVideoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(t => t.stop());
      signVideoRef.current.srcObject = null;
    }
    setIsSignCameraActive(false);
    setSignResult(null);
    speakText('Sign language camera stopped.');
  };

  const flipSignCamera = async () => {
    const nextMode = signFacingMode === 'user' ? 'environment' : 'user';
    setSignFacingMode(nextMode);
    stopSignCamera();
    setTimeout(() => startSignCamera(nextMode), 300);
  };

  // Drishti VLM / YOLO Scene & Experimental Currency Analysis
  const captureAndAnalyzeVision = async (forceSpeak = false, questionOverride?: string) => {
    if (!visionVideoRef.current || !isVisionCameraActive || isAnalyzingVisionRef.current) return;
    if (!visionVideoRef.current.videoWidth) return;

    isAnalyzingVisionRef.current = true;
    setIsAnalyzingVision(true);
    const video = visionVideoRef.current;
    const hiddenCanvas = document.createElement('canvas');
    hiddenCanvas.width = video.videoWidth || 640;
    hiddenCanvas.height = video.videoHeight || 480;
    const ctx = hiddenCanvas.getContext('2d');
    if (!ctx) {
      isAnalyzingVisionRef.current = false;
      setIsAnalyzingVision(false);
      return;
    }

    ctx.drawImage(video, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
    const base64Data = hiddenCanvas.toDataURL('image/jpeg', 0.8);
    const apiMode = drishtiSubMode === 'currency' ? 'currency' : 'scene';
    const activeQ = (questionOverride !== undefined ? questionOverride : visionQuestion).trim() || undefined;

    try {
      const res = await fetch('/api/vision/detect-base64', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: base64Data,
          mode: apiMode,
          question: activeQ,
          language: selectedLang
        })
      });

      if (res.ok) {
        const data: VisionResponse = await res.json();
        setVisionResult(data);
        drawVisionBoxes(data.detections, hiddenCanvas.width, hiddenCanvas.height);

        if (data.proximity_score) {
          playSonarBeep(data.proximity_score);
        }

        if (data.narration) {
          const now = Date.now();
          const last = lastSpokenVisionRef.current;
          const isSpeaking = isSarvamSpeakingRef.current || ('speechSynthesis' in window && window.speechSynthesis.speaking);
          const changed = data.narration !== last.text;
          const cooldownPassed = now - last.time > 7000;

          if (forceSpeak || (!isSpeaking && (changed || cooldownPassed))) {
            speakText(data.narration);
            lastSpokenVisionRef.current = { text: data.narration, time: now };
          }
        }
      }
    } catch (e) {
      console.error("Vision analysis failed", e);
    } finally {
      isAnalyzingVisionRef.current = false;
      setIsAnalyzingVision(false);
    }
  };

  // Automatic Continuous See & Speak Loop when Drishti Camera is Active
  useEffect(() => {
    if (!isVisionCameraActive || !autoVisionSpeak || drishtiSubMode === 'ocr') return;

    // Run once shortly after camera starts, then every 2.2s
    const initialTimer = setTimeout(() => {
      captureAndAnalyzeVision(true);
    }, 600);

    const interval = setInterval(() => {
      captureAndAnalyzeVision(false);
    }, 2200);

    return () => {
      clearTimeout(initialTimer);
      clearInterval(interval);
    };
  }, [isVisionCameraActive, autoVisionSpeak, drishtiSubMode]);

  const drawVisionBoxes = (detections: Detection[], origW: number, origH: number) => {
    const canvas = visionCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / origW;
    const scaleY = canvas.height / origH;

    detections.forEach(det => {
      const [x1, y1, x2, y2] = det.bbox;
      const rx = x1 * scaleX;
      const ry = y1 * scaleY;
      const rw = (x2 - x1) * scaleX;
      const rh = (y2 - y1) * scaleY;

      ctx.strokeStyle = det.is_obstacle ? '#ef4444' : det.is_currency ? '#10b981' : '#06b6d4';
      ctx.lineWidth = 3;
      ctx.strokeRect(rx, ry, rw, rh);

      ctx.fillStyle = det.is_obstacle ? 'rgba(239, 68, 68, 0.85)' : 'rgba(6, 182, 212, 0.85)';
      const labelText = `${det.label} (${Math.round(det.confidence * 100)}%) • ${det.position}`;
      ctx.font = '14px Inter, sans-serif';
      const textWidth = ctx.measureText(labelText).width;
      ctx.fillRect(rx, Math.max(0, ry - 24), textWidth + 12, 24);

      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, rx + 6, Math.max(16, ry - 7));
    });
  };

  // Document & Medicine Reader: uses VLM when configured, or local Tesseract.js OCR fallback
  const runDocumentOCR = async (imageSource: string | File) => {
    setIsAnalyzingVision(true);
    setOcrResult(null);
    speakText("Scanning medicine or document text. Please hold steady...");

    try {
      if (vlmProvider) {
        let res: Response;
        if (typeof imageSource === 'string') {
          res = await fetch('/api/vision/detect-base64', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_base64: imageSource, mode: 'ocr', language: selectedLang })
          });
        } else {
          const fd = new FormData();
          fd.append('file', imageSource);
          fd.append('mode', 'ocr');
          fd.append('language', selectedLang);
          res = await fetch('/api/vision/detect', { method: 'POST', body: fd });
        }
        if (res.ok) {
          const vData: VisionResponse = await res.json();
          const summary = vData.narration || "Could not read text clearly.";
          setOcrResult({
            full_text: summary,
            lines: [summary],
            entities: {},
            speech_summary: summary
          });
          speakText(summary);
          setIsAnalyzingVision(false);
          return;
        }
      }

      const { createWorker } = await import('tesseract.js');
      const worker = await createWorker('eng');
      const ret = await worker.recognize(imageSource);
      await worker.terminate();

      const extractedText = ret.data.text;
      if (!extractedText.trim()) {
        speakText("No clear text detected. Local OCR needs flat, well-lit, large print.");
        setIsAnalyzingVision(false);
        return;
      }

      const res = await fetch('/api/vision/analyze-ocr-text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ raw_text: extractedText })
      });

      if (res.ok) {
        const data: OCRResponse = await res.json();
        setOcrResult(data);
        speakText(data.speech_summary);
      }
    } catch (err: any) {
      console.error("OCR error:", err);
      speakText("Could not complete text scan.");
    } finally {
      setIsAnalyzingVision(false);
    }
  };

  // Samvaad Live Sign Recognition & Sentence Composition
  const captureAndAnalyzeSign = async () => {
    if (!signVideoRef.current || !isSignCameraActive || isAnalyzingSign) return;

    setIsAnalyzingSign(true);
    const video = signVideoRef.current;
    const hiddenCanvas = document.createElement('canvas');
    hiddenCanvas.width = video.videoWidth || 640;
    hiddenCanvas.height = video.videoHeight || 480;
    const ctx = hiddenCanvas.getContext('2d');
    if (!ctx) {
      setIsAnalyzingSign(false);
      return;
    }

    ctx.drawImage(video, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
    const base64Data = hiddenCanvas.toDataURL('image/jpeg', 0.8);

    try {
      const res = await fetch('/api/sign/detect-frame-base64', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: base64Data })
      });

      if (res.ok) {
        const data: SignPrediction = await res.json();
        setSignResult(data);
        drawSignBox(data, hiddenCanvas.width, hiddenCanvas.height);

        if (data.recognized && data.gesture !== 'NO_HAND') {
          // Add to sequence queue if new
          handleNewRecognizedSign(data.gesture);
        }
      }
    } catch (e) {
      console.error("Sign analysis failed", e);
    } finally {
      setIsAnalyzingSign(false);
    }
  };

  const handleNewRecognizedSign = async (gestureKey: string) => {
    setGestureSequence(prev => {
      const updated = prev[prev.length - 1] === gestureKey ? prev : [...prev, gestureKey];
      // Keep last 4 gestures
      const trimmed = updated.slice(-4);
      // Fetch assembled sentence
      fetch('/api/sign/sequence-sentence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gestures: trimmed })
      })
      .then(r => r.json())
      .then(sent => {
        setComposedSentence({ en: sent.sentence_en, hi: sent.sentence_hi });
        speakText(`${sent.sentence_en}. Hindi: ${sent.sentence_hi}`);
      })
      .catch(console.error);

      return trimmed;
    });
  };

  const drawSignBox = (data: SignPrediction, origW: number, origH: number) => {
    const canvas = signCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (data.bbox) {
      const [x1, y1, x2, y2] = data.bbox;
      const scaleX = canvas.width / origW;
      const scaleY = canvas.height / origH;
      const rx = x1 * scaleX;
      const ry = y1 * scaleY;
      const rw = (x2 - x1) * scaleX;
      const rh = (y2 - y1) * scaleY;

      ctx.strokeStyle = '#ec4899';
      ctx.lineWidth = 4;
      ctx.strokeRect(rx, ry, rw, rh);

      const labelText = `ISL Sign: ${data.phrase_en} (${Math.round(data.confidence * 100)}%)`;
      ctx.fillStyle = 'rgba(236, 72, 153, 0.9)';
      ctx.font = '15px Inter, sans-serif';
      const textWidth = ctx.measureText(labelText).width;
      ctx.fillRect(rx, Math.max(0, ry - 28), textWidth + 14, 28);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, rx + 7, Math.max(20, ry - 9));
    }
  };

  const toggleAutoSignTracking = () => {
    if (autoSignTrack) {
      if (signIntervalRef.current) clearInterval(signIntervalRef.current);
      signIntervalRef.current = null;
      setAutoSignTrack(false);
      speakText('Continuous sign recognition paused.');
    } else {
      setAutoSignTrack(true);
      speakText('Continuous sign sentence builder active. Hold hand in front of camera.');
      signIntervalRef.current = setInterval(() => {
        captureAndAnalyzeSign();
      }, 1200);
    }
  };

  // Shared Multilingual Voice Command Parser (English, Hindi, Marathi)
  const executeVoiceCommand = (transcript: string) => {
    const cmd = transcript.toLowerCase();
    if (!cmd.trim()) return;

    if (cmd.includes('what do you see') || cmd.includes('in front') || cmd.includes('describe') || cmd.includes('kya hai') || cmd.includes('saamne') || cmd.includes('क्या है') || cmd.includes('सामने') || cmd.includes('काय आहे')) {
      setActiveTab('drishti');
      setDrishtiSubMode('vision');
      setVisionQuestion(transcript);
      if (!isVisionCameraActive) {
        startVisionCamera();
      } else {
        captureAndAnalyzeVision(true, transcript);
      }
    } else if (cmd.includes('currency') || cmd.includes('rupee') || cmd.includes('note') || cmd.includes('paisa') || cmd.includes('नोट') || cmd.includes('रुपये')) {
      setActiveTab('drishti');
      setDrishtiSubMode('currency');
      if (isVisionCameraActive) {
        setTimeout(() => captureAndAnalyzeVision(true), 200);
      } else {
        speakText('Switched to experimental currency check mode. Please verify notes by touch as well.');
      }
    } else if (cmd.includes('medicine') || cmd.includes('ocr') || cmd.includes('dawai') || cmd.includes('दवाई') || cmd.includes('पढ़ो')) {
      setActiveTab('drishti');
      setDrishtiSubMode('ocr');
      speakText('Switched to Medicine and Document Reader mode.');
    } else if (cmd.includes('drishti') || cmd.includes('vision') || cmd.includes('camera') || cmd.includes('eye') || cmd.includes('दृष्टि')) {
      setActiveTab('drishti');
      speakText('Opening Drishti Visual Assistant.');
    } else if (cmd.includes('samvaad') || cmd.includes('sign') || cmd.includes('hand') || cmd.includes('gesture') || cmd.includes('संवाद') || cmd.includes('इशारा')) {
      setActiveTab('samvaad');
      speakText('Opening Samvaad 8-gesture speech demo.');
    } else if (cmd.includes('contrast')) {
      setHighContrast(prev => !prev);
      speakText('Toggled high contrast mode.');
    } else if (cmd.includes('text') || cmd.includes('font') || cmd.includes('large')) {
      setLargeFont(prev => !prev);
      speakText('Toggled large font mode.');
    } else if (cmd.includes('stop') || cmd.includes('ruko') || cmd.includes('बंद')) {
      stopVisionCamera();
      stopSignCamera();
      speakText('Stopped cameras.');
    } else if (cmd.includes('read') || cmd.includes('summary')) {
      speakText('SAHAAY AI platform is active and ready.');
    } else if (activeTab === 'drishti' && isVisionCameraActive) {
      // Any other spoken question while viewing Drishti camera asks the scene analyzer directly
      setVisionQuestion(transcript);
      captureAndAnalyzeVision(true, transcript);
    }
  };

  const stopWhisperRecording = () => {
    if (whisperStopTimerRef.current) {
      clearTimeout(whisperStopTimerRef.current);
      whisperStopTimerRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    setIsListening(false);
  };

  // Real Microphone: Sarvam AI / Whisper AI backend (default 'auto') or Browser SpeechRecognition
  const toggleListening = async () => {
    if (isListening) {
      if (micEngine !== 'browser') {
        stopWhisperRecording();
      } else {
        if (recognitionRef.current) recognitionRef.current.stop();
        setIsListening(false);
        speakText('Voice microphone stopped.');
      }
      return;
    }

    setMicError(null);

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const msg = "Microphone requires HTTPS or localhost.";
      setMicError(msg);
      speakText(msg);
      return;
    }

    // 1. Backend Speech Mode (Sarvam AI saarika:v2.5 when key set, else faster-whisper)
    if (micEngine !== 'browser') {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioChunksRef.current = [];

        const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus'
          : MediaRecorder.isTypeSupported('audio/mp4')
            ? 'audio/mp4'
            : '';
        const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
        mediaRecorderRef.current = recorder;

        recorder.ondataavailable = (ev) => {
          if (ev.data && ev.data.size > 0) {
            audioChunksRef.current.push(ev.data);
          }
        };

        recorder.onstop = async () => {
          stream.getTracks().forEach(t => t.stop());
          const audioBlob = new Blob(audioChunksRef.current, { type: recorder.mimeType || 'audio/webm' });
          if (audioBlob.size === 0) return;

          setIsTranscribing(true);
          try {
            const formData = new FormData();
            formData.append('file', audioBlob, 'voice.webm');
            formData.append('language', selectedLang === 'en' ? 'auto' : selectedLang);
            formData.append('engine', micEngine);

            const res = await fetch('/api/speech/transcribe', {
              method: 'POST',
              body: formData
            });

            if (res.ok) {
              const data = await res.json();
              const text = (data.text || '').trim();
              setDetectedSpeechLang(data.language || null);
              setActiveSpeechEngine(data.engine || null);
              if (text) {
                setVoiceTranscript(text);
                executeVoiceCommand(text);
              } else {
                speakText("I didn't catch any speech. Please try speaking closer to the mic.");
              }
            } else {
              const errData = await res.json().catch(() => ({}));
              setMicError(errData.detail || "Speech transcription failed.");
            }
          } catch (err: any) {
            setMicError(`Speech upload failed: ${err.message}`);
          } finally {
            setIsTranscribing(false);
          }
        };

        recorder.start();
        setIsListening(true);
        // Auto-stop after 6 seconds so user doesn't have to tap twice if hands-free
        whisperStopTimerRef.current = setTimeout(() => {
          stopWhisperRecording();
        }, 6000);
      } catch (err: any) {
        const msg = "Microphone permission is blocked. Please allow microphone access in your browser.";
        setMicError(msg);
        speakText(msg);
      }
      return;
    }

    // 2. Browser Web Speech API Mode
    try {
      const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioStream.getTracks().forEach(t => t.stop());
    } catch {
      const msg = "Microphone permission is blocked. Please allow microphone access in your browser.";
      setMicError(msg);
      speakText(msg);
      return;
    }

    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      const err = "Browser live speech is unsupported here—switch Mic Engine to Whisper AI.";
      setMicError(err);
      speakText(err);
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      recognitionRef.current = recognition;
      recognition.continuous = true;
      recognition.interimResults = true;
      const langConfig = LANGUAGES.find(l => l.code === selectedLang) || LANGUAGES[0];
      recognition.lang = langConfig.speechCode;

      recognition.onstart = () => {
        setIsListening(true);
        speakText('Browser microphone active. I am listening.');
      };

      recognition.onresult = (event: any) => {
        let currentTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          currentTranscript += event.results[i][0].transcript;
        }
        setVoiceTranscript(currentTranscript);
        executeVoiceCommand(currentTranscript);
      };

      recognition.onerror = (event: any) => {
        if (event.error === 'not-allowed') {
          setMicError("Microphone permission denied. Allow mic access in browser.");
        } else if (event.error !== 'no-speech') {
          setMicError(`Voice error: ${event.error}`);
        }
        setIsListening(false);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognition.start();
    } catch (err: any) {
      setMicError(`Unable to start speech recognition: ${err.message}`);
      setIsListening(false);
    }
  };

  // Direct Mobile Photo Capture Handlers
  const handleVisionPhoto = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (drishtiSubMode === 'ocr') {
      runDocumentOCR(file);
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('mode', drishtiSubMode === 'currency' ? 'currency' : 'scene');
    formData.append('language', selectedLang);
    if (visionQuestion.trim()) {
      formData.append('question', visionQuestion.trim());
    }
    setIsAnalyzingVision(true);

    try {
      const res = await fetch('/api/vision/detect', {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        const data: VisionResponse = await res.json();
        setVisionResult(data);
        if (data.proximity_score) {
          playSonarBeep(data.proximity_score);
        }
        if (data.narration) {
          speakText(data.narration);
        }
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsAnalyzingVision(false);
    }
  };

  const handleSignPhoto = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    setIsAnalyzingSign(true);

    try {
      const res = await fetch('/api/sign/detect-frame', {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        const data: SignPrediction = await res.json();
        setSignResult(data);
        if (data.recognized && data.gesture !== 'NO_HAND') {
          handleNewRecognizedSign(data.gesture);
        }
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsAnalyzingSign(false);
    }
  };

  // Simulator for quick testing
  const testSimGesture = async (gestureKey: string) => {
    setSelectedSimGesture(gestureKey);
    const mockLandmarks = Array.from({ length: 21 }, (_, i) => ({
      x: 0.5,
      y: gestureKey === 'GOOD' ? (i === 4 ? 0.2 : 0.8) : 0.4,
      z: 0.0
    }));

    try {
      const res = await fetch('/api/sign/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ landmarks: mockLandmarks })
      });
      if (res.ok) {
        const data: SignPrediction = await res.json();
        setSignResult(data);
        handleNewRecognizedSign(gestureKey);
      }
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header" role="banner">
        <div className="brand-section">
          <div className="logo-badge" aria-hidden="true">
            <Sparkles size={24} />
          </div>
          <div>
            <h1 className="brand-title">SAHAAY AI</h1>
            <div className="brand-sub">सहाय एआई • Voice-First Accessibility Platform</div>
          </div>
        </div>

        <div className="header-controls" role="toolbar" aria-label="Accessibility and Language Settings">
          {/* Sonar Audio Pulse Toggle */}
          <button 
            className={`icon-btn ${sonarBeepEnabled ? 'active' : ''}`}
            onClick={() => {
              setSonarBeepEnabled(!sonarBeepEnabled);
              speakText(!sonarBeepEnabled ? "Sonar proximity beeps enabled." : "Sonar beeps muted.");
            }}
            title="Toggle Sonar Obstacle Beep"
          >
            <Radio size={18} />
            <span>Sonar: {sonarBeepEnabled ? 'ON' : 'OFF'}</span>
          </button>

          {/* Language Selector */}
          <button
            className="icon-btn"
            aria-label={`Current language: ${selectedLang}`}
            onClick={() => {
              const nextIdx = (LANGUAGES.findIndex(l => l.code === selectedLang) + 1) % LANGUAGES.length;
              setSelectedLang(LANGUAGES[nextIdx].code);
              speakText(`Language changed to ${LANGUAGES[nextIdx].label}`);
            }}
          >
            <Languages size={18} />
            <span>{LANGUAGES.find(l => l.code === selectedLang)?.native}</span>
          </button>

          {/* High Contrast Toggle */}
          <button 
            className={`icon-btn ${highContrast ? 'active' : ''}`}
            onClick={() => {
              setHighContrast(!highContrast);
              speakText(!highContrast ? "High contrast mode enabled." : "High contrast mode disabled.");
            }}
            aria-label="Toggle High Contrast"
          >
            <SunMoon size={18} />
            <span>Contrast</span>
          </button>

          {/* Large Font Toggle */}
          <button 
            className={`icon-btn ${largeFont ? 'active' : ''}`}
            onClick={() => {
              setLargeFont(!largeFont);
              speakText(!largeFont ? "Large text enabled." : "Standard text enabled.");
            }}
            aria-label="Toggle Large Font Size"
          >
            <Maximize2 size={18} />
            <span>Text</span>
          </button>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="nav-tabs" aria-label="Module Tabs">
        <button 
          className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          <Layers size={18} />
          <span>Platform Overview</span>
        </button>
        <button 
          className={`tab-btn ${activeTab === 'drishti' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('drishti');
            speakText("Navigated to Drishti Visual Assistant.");
          }}
        >
          <Eye size={18} />
          <span>Drishti: Visual Assistant</span>
        </button>
        <button 
          className={`tab-btn ${activeTab === 'samvaad' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('samvaad');
            speakText("Navigated to Samvaad 8-gesture speech demo.");
          }}
        >
          <HandMetal size={18} />
          <span>Samvaad: Gesture Demo (8 Signs)</span>
        </button>
      </nav>

      {/* Main Content */}
      <main id="main-content" role="main">
        {/* OVERVIEW TAB */}
        {activeTab === 'overview' && (
          <section className="hero-section">
            <div className="hero-tag">
              <Sparkles size={16} />
              <span>Voice-First Visual Assistant & Gesture Prototype</span>
            </div>

            <h2 className="hero-title">
              SAHAAY AI — Ask & Hear What Is In Front of You
            </h2>
            <p className="hero-description">
              A voice-first accessibility prototype featuring <strong>Drishti</strong> (live camera scene description &amp; audio proximity cues) alongside an early <strong>Samvaad</strong> 8-gesture speech demo.
            </p>

            {/* Voice Command Bar */}
            <div className="voice-action-bar">
              <button 
                className={`primary-voice-btn ${isListening ? 'listening' : ''}`}
                onClick={toggleListening}
                disabled={isTranscribing}
                aria-label={isListening ? "Stop voice listening" : "Start voice command"}
              >
                {isListening ? <MicOff size={22} /> : <Mic size={22} />}
                <span>
                  {isTranscribing
                    ? "Transcribing speech..."
                    : isListening
                      ? (micEngine !== 'browser' ? "Recording... (Tap to send)" : "Listening... (Tap to stop)")
                      : `Speak to SAHAAY (${micEngine === 'auto' ? (sarvamAvailable ? 'Sarvam + Whisper' : 'Whisper AI') : micEngine === 'whisper' ? 'Whisper AI' : 'Browser'})`}
                </span>
              </button>

              <button
                className={`icon-btn ${micEngine !== 'browser' ? 'active' : ''}`}
                onClick={() => {
                  const order: Array<'auto' | 'whisper' | 'browser'> = ['auto', 'whisper', 'browser'];
                  const next = order[(order.indexOf(micEngine) + 1) % order.length];
                  setMicEngine(next);
                  const label = next === 'auto' ? 'Auto (Sarvam plus Whisper)' : next === 'whisper' ? 'Whisper AI' : 'Browser Speech';
                  speakText(`Switched voice engine to ${label}.`);
                }}
                style={{ padding: '0.875rem 1.1rem' }}
                title="Cycle between Auto (Sarvam + Whisper), Whisper AI, and Browser Speech"
              >
                <span>Engine: {micEngine === 'auto' ? 'Auto (Sarvam/Whisper)' : micEngine === 'whisper' ? 'Whisper AI' : 'Browser'}</span>
              </button>

              <button 
                className="icon-btn"
                onClick={() => speakText("SAHAAY AI is active. Choose Drishti for vision and medicine identification or Samvaad for sign sentence communication.")}
                style={{ padding: '0.875rem 1.25rem' }}
              >
                <Volume2 size={20} />
                <span>Read Screen Aloud</span>
              </button>
            </div>

            {/* Mic Error Banner */}
            {micError && (
              <div style={{ maxWidth: '680px', margin: '0 auto 1rem', padding: '0.75rem 1rem', borderRadius: '10px', background: 'rgba(239, 68, 68, 0.15)', border: '1px solid #ef4444', color: '#fca5a5', fontSize: '0.9rem' }}>
                <AlertTriangle size={16} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
                {micError}
              </div>
            )}

            {/* Live Voice Transcript Bar */}
            {voiceTranscript && (
              <div 
                style={{ 
                  maxWidth: '680px', 
                  margin: '0 auto 1.5rem', 
                  padding: '0.75rem 1.25rem', 
                  borderRadius: '10px', 
                  background: 'rgba(6, 182, 212, 0.1)',
                  border: '1px solid var(--accent-cyan)',
                  fontSize: '0.95rem',
                  color: '#e0f2fe'
                }}
              >
                <strong>{activeSpeechEngine || (micEngine === 'browser' ? 'Browser Transcript' : 'AI Transcript')}{detectedSpeechLang ? ` (${detectedSpeechLang.toUpperCase()})` : ''}: </strong>
                <span>"{voiceTranscript}"</span>
              </div>
            )}

            {/* Status Announcement Bar */}
            <div 
              style={{ 
                maxWidth: '680px', 
                margin: '0 auto 2.5rem', 
                padding: '0.75rem 1.25rem', 
                borderRadius: '10px', 
                background: 'rgba(255, 255, 255, 0.04)',
                border: '1px dashed var(--border-color)',
                fontSize: '0.9rem',
                color: 'var(--text-secondary)'
              }}
              aria-live="polite"
            >
              <strong>Voice Audio Status: </strong>
              <span>{voiceAnnouncement}</span>
            </div>

            {/* Backend Health Badge */}
            <div className="health-status-card" aria-label="Backend Server Status">
              <div className="health-info">
                <span className={`status-dot ${healthStatus}`} />
                <div>
                  <div className="health-text-primary">
                    FastAPI Backend: {healthStatus === 'online' ? 'Connected & Healthy' : healthStatus === 'loading' ? 'Checking connection...' : 'Offline (Start backend server)'}
                  </div>
                  <div className="health-text-secondary">
                    {healthStatus === 'online' && health ? (
                      <>Endpoint: <code>/api/health</code> • Latency: {latency}ms • v{health.version}</>
                    ) : (
                      'Run uvicorn in backend/'
                    )}
                  </div>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                {health && (
                  <div className="health-services-pills">
                    <span className="service-pill">
                      <CheckCircle2 size={12} color="#10b981" /> Vision: {vlmProvider ? `${vlmProvider.toUpperCase()} VLM + YOLOv8n` : 'YOLOv8n Local (80 classes)'}
                    </span>
                    <span className="service-pill">
                      <CheckCircle2 size={12} color="#10b981" /> Samvaad: 8-Gesture Demo
                    </span>
                    <span className="service-pill">
                      <CheckCircle2 size={12} color="#10b981" /> Speech: {sarvamAvailable ? 'Sarvam AI + Whisper' : 'Whisper-base (English best)'}
                    </span>
                  </div>
                )}
                <button className="icon-btn" onClick={checkHealth} title="Refresh Status">
                  <RefreshCw size={16} />
                </button>
              </div>
            </div>

            {/* Modules Grid */}
            <div className="modules-container">
              <div className="modules-grid">
                <article className="module-card">
                  <div className="module-icon drishti-icon">
                    <Eye size={28} />
                  </div>
                  <div className="module-header">
                    <h3 className="module-title">Drishti (Primary Flow)</h3>
                    <div className="module-native-title">दृष्टि • Visual Assistant &amp; Audio Guidance</div>
                  </div>
                  <p className="module-description">
                    Continuous live camera narration ({vlmProvider ? `${vlmProvider.toUpperCase()} VLM` : 'YOLOv8n local + VLM-ready'}), spatial orientation (left / ahead / right), qualitative closeness (very close / nearby / far), sonar proximity cues, and medicine/document reading.
                  </p>
                  <button className="module-cta-btn" onClick={() => setActiveTab('drishti')}>
                    <span>Open Drishti Visual Assistant</span>
                    <ArrowRight size={16} />
                  </button>
                </article>

                <article className="module-card">
                  <div className="module-icon samvaad-icon">
                    <HandMetal size={28} />
                  </div>
                  <div className="module-header">
                    <h3 className="module-title">Samvaad (Prototype Demo)</h3>
                    <div className="module-native-title">संवाद • Gesture-to-Speech (8 Sample Gestures)</div>
                  </div>
                  <p className="module-description">
                    Experimental OpenCV hand-contour prototype mapping 8 sample hand gestures into bilingual Hindi &amp; English speech (not full ISL grammar).
                  </p>
                  <button className="module-cta-btn" onClick={() => setActiveTab('samvaad')}>
                    <span>Open 8-Gesture Demo</span>
                    <ArrowRight size={16} />
                  </button>
                </article>
              </div>
            </div>
          </section>
        )}

        {/* DRISHTI VISION & OCR STUDIO TAB */}
        {activeTab === 'drishti' && (
          <section className="studio-container">
            <div className="studio-card">
              <div className="studio-header">
                <div className="studio-title-area">
                  <Eye size={24} color="var(--accent-cyan)" />
                  <div>
                    <h2 style={{ fontSize: '1.25rem', fontWeight: 700 }}>Drishti Visual Assistant</h2>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                      Engine: {visionResult?.vision_engine || (vlmProvider ? `${vlmProvider.toUpperCase()} VLM + YOLOv8n` : 'YOLOv8n Local (80 classes — add GEMINI_API_KEY for full scene narration)')}
                    </div>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                  {/* Mode Selector */}
                  <button 
                    className={`icon-btn ${drishtiSubMode === 'vision' ? 'active' : ''}`}
                    onClick={() => {
                      setDrishtiSubMode('vision');
                      speakText("Switched to Live Scene mode. I will describe what I see.");
                    }}
                  >
                    <Eye size={16} />
                    <span>Live Scene</span>
                  </button>

                  <button 
                    className={`icon-btn ${drishtiSubMode === 'currency' ? 'active' : ''}`}
                    onClick={() => {
                      setDrishtiSubMode('currency');
                      speakText("Switched to experimental currency check mode. Please verify notes by touch as well.");
                    }}
                  >
                    <Sparkles size={16} />
                    <span>Currency (Experimental)</span>
                  </button>

                  <button 
                    className={`icon-btn ${drishtiSubMode === 'ocr' ? 'active' : ''}`}
                    onClick={() => {
                      setDrishtiSubMode('ocr');
                      speakText("Switched to Document and Label Reader mode.");
                    }}
                  >
                    <FileText size={16} />
                    <span>Read Label / OCR</span>
                  </button>

                  {isVisionCameraActive && drishtiSubMode !== 'ocr' && (
                    <button
                      className={`icon-btn ${autoVisionSpeak ? 'active' : ''}`}
                      onClick={() => {
                        setAutoVisionSpeak(!autoVisionSpeak);
                        speakText(!autoVisionSpeak ? "Auto-See and Speak turned on." : "Auto-See and Speak paused.");
                      }}
                      title="Automatically describe what camera sees every 2 seconds"
                    >
                      <Volume2 size={16} />
                      <span>Auto-Speak: {autoVisionSpeak ? 'ON' : 'OFF'}</span>
                    </button>
                  )}

                  {isVisionCameraActive && (
                    <button className="icon-btn" onClick={flipVisionCamera} title="Flip Front / Back Camera">
                      <SwitchCamera size={18} />
                      <span>Flip Camera</span>
                    </button>
                  )}

                  {!isVisionCameraActive ? (
                    <button className="icon-btn active" onClick={() => startVisionCamera()}>
                      <Camera size={18} />
                      <span>Start Camera</span>
                    </button>
                  ) : (
                    <button className="icon-btn" onClick={stopVisionCamera}>
                      <CameraOff size={18} />
                      <span>Stop Camera</span>
                    </button>
                  )}
                </div>
              </div>

              {/* Safety Boundary Notice */}
              <div
                role="note"
                aria-label="Safety boundary notice"
                style={{
                  padding: '0.65rem 1.5rem',
                  background: 'rgba(245, 158, 11, 0.12)',
                  borderBottom: '1px solid rgba(245, 158, 11, 0.35)',
                  color: '#fde68a',
                  fontSize: '0.85rem',
                  lineHeight: '1.4',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}
              >
                <AlertTriangle size={16} style={{ flexShrink: 0, color: '#f59e0b' }} />
                <span>
                  <strong>Safety Boundary:</strong> Drishti is an AI assistance tool only. It can miss moving vehicles, potholes, or transparent objects—never use it as a replacement for a white cane, guide dog, or mobility training.
                </span>
              </div>

              {/* Experimental Currency Warning */}
              {drishtiSubMode === 'currency' && (
                <div
                  role="alert"
                  style={{
                    padding: '0.65rem 1.5rem',
                    background: 'rgba(239, 68, 68, 0.14)',
                    borderBottom: '1px solid rgba(239, 68, 68, 0.4)',
                    color: '#fecaca',
                    fontSize: '0.85rem'
                  }}
                >
                  <strong>Experimental Warning:</strong> Color-based currency guessing is unreliable under varied lighting and can misidentify ₹10/₹50/₹500 notes. Never rely on this alone for money—verify by tactile markings or human assistance.
                </div>
              )}

              {/* OCR Limitation Notice */}
              {drishtiSubMode === 'ocr' && !vlmProvider && (
                <div
                  role="note"
                  style={{
                    padding: '0.65rem 1.5rem',
                    background: 'rgba(6, 182, 212, 0.1)',
                    borderBottom: '1px solid rgba(6, 182, 212, 0.3)',
                    color: '#a5f3fc',
                    fontSize: '0.85rem'
                  }}
                >
                  <strong>OCR Notice:</strong> Running local Tesseract.js OCR (no VLM key set). Works best on large, flat, well-lit text; small print (like prices or foil strips) may misread—always inspect raw lines below.
                </div>
              )}

              {/* Camera Error Alert */}
              {visionCameraError && (
                <div style={{ padding: '0.875rem 1.5rem', background: 'rgba(239, 68, 68, 0.15)', borderBottom: '1px solid #ef4444', color: '#fca5a5', fontSize: '0.9rem', lineHeight: '1.4' }}>
                  <AlertTriangle size={18} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
                  {visionCameraError}
                </div>
              )}

              {/* Video Viewport */}
              <div className="video-view-wrapper">
                {visionResult?.obstacle_alert && (
                  <div className="obstacle-alert-banner">
                    <AlertTriangle size={18} />
                    <span>CAUTION: Close Obstacle Detected!</span>
                  </div>
                )}

                <video 
                  ref={visionVideoRef} 
                  className="camera-feed" 
                  autoPlay
                  playsInline 
                  muted 
                  style={{ display: isVisionCameraActive ? 'block' : 'none' }}
                />
                <canvas ref={visionCanvasRef} className="bbox-canvas" />

                {!isVisionCameraActive && (
                  <div className="camera-placeholder">
                    {drishtiSubMode === 'ocr' ? (
                      <Pill size={48} style={{ opacity: 0.3, marginBottom: '1rem', color: '#06b6d4' }} />
                    ) : (
                      <Camera size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
                    )}
                    <p style={{ fontSize: '1.1rem', fontWeight: 600 }}>
                      {drishtiSubMode === 'ocr' ? 'Medicine & Document Reader' : 'Drishti Vision Camera is currently off'}
                    </p>
                    <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginTop: '0.35rem', maxWidth: '480px', margin: '0.35rem auto 0' }}>
                      {drishtiSubMode === 'ocr' 
                        ? 'Hold medicine box, prescription, or grocery item in front of your camera, or tap "Scan Medicine Photo".' 
                        : 'Click "Start Camera" for live video, or tap "Take Phone Camera Photo" below.'}
                    </p>
                  </div>
                )}
              </div>

              {/* VLM Fallback Error Notice */}
              {visionResult?.vlm_error && (
                <div style={{ padding: '0.6rem 1.5rem', background: 'rgba(239, 68, 68, 0.12)', borderTop: '1px solid rgba(239, 68, 68, 0.35)', color: '#fca5a5', fontSize: '0.82rem' }}>
                  <strong>VLM API Notice (fell back to YOLOv8n): </strong>
                  <span>{visionResult.vlm_error}</span>
                </div>
              )}

              {/* OCR Entity Highlights Bar */}
              {ocrResult?.entities && (
                <div className="entity-cards-container">
                  {ocrResult.entities.dosage && (
                    <div className="entity-pill-card">
                      <div className="entity-label">Dosage Detected</div>
                      <div className="entity-value">{ocrResult.entities.dosage}</div>
                    </div>
                  )}
                  {ocrResult.entities.expiry && (
                    <div className="entity-pill-card">
                      <div className="entity-label">Expiry Date</div>
                      <div className="entity-value">{ocrResult.entities.expiry}</div>
                    </div>
                  )}
                  {ocrResult.entities.price && (
                    <div className="entity-pill-card">
                      <div className="entity-label">Price</div>
                      <div className="entity-value">{ocrResult.entities.price}</div>
                    </div>
                  )}
                  {ocrResult.entities.warning && (
                    <div className="entity-pill-card" style={{ borderColor: '#ef4444' }}>
                      <div className="entity-label" style={{ color: '#ef4444' }}>Notice / Warning</div>
                      <div className="entity-value" style={{ fontSize: '0.9rem' }}>{ocrResult.entities.warning}</div>
                    </div>
                  )}
                </div>
              )}

              {/* Raw OCR Text Verification Box */}
              {ocrResult?.full_text && (
                <div style={{ padding: '0.75rem 1.75rem', background: 'rgba(255,255,255,0.03)', borderTop: '1px solid var(--border-color)', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  <strong>Raw Scanned Text: </strong>
                  <span style={{ fontFamily: 'monospace' }}>{ocrResult.full_text}</span>
                </div>
              )}

              {/* Voice Narration Output Bar */}
              {(visionResult || ocrResult) && (
                <div className="narration-box" role="status">
                  <Volume2 size={24} color="var(--accent-cyan)" />
                  <div>
                    <strong>Voice Guidance: </strong>
                    <span>{ocrResult?.speech_summary || visionResult?.narration}</span>
                  </div>
                </div>
              )}

              {/* Optional Question Input for VLM Scene QA */}
              {drishtiSubMode === 'vision' && (
                <div style={{ padding: '0.75rem 1.75rem', borderTop: '1px solid var(--border-color)', display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center', background: 'rgba(255,255,255,0.01)' }}>
                  <input
                    type="text"
                    value={visionQuestion}
                    onChange={(e) => setVisionQuestion(e.target.value)}
                    placeholder="Ask a question about the scene (e.g. 'Is there a door?' or 'Read that sign')..."
                    aria-label="Ask a question about what the camera sees"
                    style={{
                      flex: 1,
                      minWidth: '220px',
                      padding: '0.55rem 0.85rem',
                      borderRadius: '8px',
                      border: '1px solid var(--border-color)',
                      background: 'rgba(15, 23, 42, 0.7)',
                      color: '#fff',
                      fontSize: '0.9rem'
                    }}
                  />
                  {visionQuestion && (
                    <button className="icon-btn" onClick={() => setVisionQuestion('')}>
                      Clear
                    </button>
                  )}
                </div>
              )}

              {/* Controls Footer */}
              <div className="studio-controls">
                <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap' }}>
                  {isVisionCameraActive && (
                    <button 
                      className="primary-voice-btn" 
                      onClick={() => {
                        if (drishtiSubMode === 'ocr' && visionVideoRef.current) {
                          const v = visionVideoRef.current;
                          const c = document.createElement('canvas');
                          c.width = v.videoWidth;
                          c.height = v.videoHeight;
                          const cx = c.getContext('2d');
                          if (cx) {
                            cx.drawImage(v, 0, 0);
                            runDocumentOCR(c.toDataURL('image/jpeg'));
                          }
                        } else {
                          captureAndAnalyzeVision(true);
                        }
                      }}
                      disabled={isAnalyzingVision}
                      style={{ padding: '0.6rem 1.25rem', fontSize: '0.95rem' }}
                    >
                      <Play size={16} />
                      <span>{isAnalyzingVision ? "Analyzing..." : drishtiSubMode === 'ocr' ? "Scan Text Now" : "Analyze Current View"}</span>
                    </button>
                  )}

                  {/* Native Mobile Camera Snap Button */}
                  <label className="primary-voice-btn" style={{ cursor: 'pointer', background: 'linear-gradient(135deg, #059669, #10b981)', padding: '0.6rem 1.25rem', fontSize: '0.95rem' }}>
                    <Smartphone size={18} />
                    <span>{drishtiSubMode === 'ocr' ? 'Snap Medicine Photo' : 'Take Phone Camera Photo'}</span>
                    <input 
                      type="file" 
                      accept="image/*" 
                      capture="environment" 
                      onChange={handleVisionPhoto} 
                      style={{ display: 'none' }} 
                    />
                  </label>

                  <label className="icon-btn" style={{ cursor: 'pointer' }}>
                    <Upload size={16} />
                    <span>Upload Image</span>
                    <input type="file" accept="image/*" onChange={handleVisionPhoto} style={{ display: 'none' }} />
                  </label>

                  <button
                    className={`icon-btn ${isListening ? 'active' : ''}`}
                    onClick={toggleListening}
                    disabled={isTranscribing}
                    title="Ask Drishti by voice using Whisper AI"
                  >
                    {isListening ? <MicOff size={16} /> : <Mic size={16} />}
                    <span>{isTranscribing ? "Transcribing..." : isListening ? "Stop & Send" : "Ask by Voice"}</span>
                  </button>
                </div>

                <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  {drishtiSubMode === 'ocr' 
                    ? (ocrResult ? `Lines Read: ${ocrResult.lines.length}` : 'Ready to scan text') 
                    : `Detected Objects: ${visionResult?.detections.length || 0}`}
                </div>
              </div>
            </div>
          </section>
        )}

        {/* SAMVAAD SIGN STUDIO TAB */}
        {activeTab === 'samvaad' && (
          <section className="studio-container">
            <div className="studio-card">
              <div className="studio-header">
                <div className="studio-title-area">
                  <HandMetal size={24} color="#ec4899" />
                  <div>
                    <h2 style={{ fontSize: '1.25rem', fontWeight: 700 }}>Samvaad Gesture-to-Speech Demo</h2>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>8 Sample Hand Gestures via OpenCV Contour Finger-Counting (Not Full ISL)</div>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                  {isSignCameraActive && (
                    <button className="icon-btn" onClick={flipSignCamera} title="Flip Front / Back Camera">
                      <SwitchCamera size={18} />
                      <span>Flip Camera</span>
                    </button>
                  )}

                  {!isSignCameraActive ? (
                    <button className="icon-btn active" onClick={() => startSignCamera()}>
                      <Camera size={18} />
                      <span>Start Hand Camera</span>
                    </button>
                  ) : (
                    <button className="icon-btn" onClick={stopSignCamera}>
                      <CameraOff size={18} />
                      <span>Stop Camera</span>
                    </button>
                  )}
                </div>
              </div>

              {/* Honest Demo Scope Banner */}
              <div
                role="note"
                style={{
                  padding: '0.6rem 1.5rem',
                  background: 'rgba(236, 72, 153, 0.1)',
                  borderBottom: '1px solid rgba(236, 72, 153, 0.3)',
                  color: '#fbcfe8',
                  fontSize: '0.85rem'
                }}
              >
                <strong>Demo Scope Notice:</strong> This is a gesture-to-speech prototype recognizing 8 simple sample hand poses (open palm, V-sign, fist, etc.) via OpenCV contour defects—it is not full Indian Sign Language (ISL) grammar.
              </div>

              {/* Sign Camera Error Alert */}
              {signCameraError && (
                <div style={{ padding: '0.875rem 1.5rem', background: 'rgba(239, 68, 68, 0.15)', borderBottom: '1px solid #ef4444', color: '#fca5a5', fontSize: '0.9rem', lineHeight: '1.4' }}>
                  <AlertTriangle size={18} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
                  {signCameraError}
                </div>
              )}

              {/* Video Viewport for Hand Tracking */}
              <div className="video-view-wrapper">
                <video 
                  ref={signVideoRef} 
                  className="camera-feed" 
                  autoPlay
                  playsInline 
                  muted 
                  style={{ display: isSignCameraActive ? 'block' : 'none' }}
                />
                <canvas ref={signCanvasRef} className="bbox-canvas" />

                {!isSignCameraActive && (
                  <div className="camera-placeholder">
                    <HandMetal size={48} style={{ opacity: 0.3, marginBottom: '1rem', color: '#ec4899' }} />
                    <p style={{ fontSize: '1.1rem', fontWeight: 600 }}>Hand Sign Camera is currently off</p>
                    <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginTop: '0.35rem', maxWidth: '480px', margin: '0.35rem auto 0' }}>
                      Click <strong>"Start Hand Camera"</strong> or tap <strong>"Snap Hand Photo with Phone"</strong> to record gestures and compose sentences.
                    </p>
                  </div>
                )}
              </div>

              {/* Gesture Sequence Breadcrumbs */}
              {gestureSequence.length > 0 && (
                <div className="sentence-queue-bar">
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginRight: '0.25rem' }}>
                    Signed Sequence:
                  </span>
                  {gestureSequence.map((g, idx) => (
                    <React.Fragment key={idx}>
                      <span className="sign-chip">{g}</span>
                      {idx < gestureSequence.length - 1 && <span style={{ color: '#ec4899' }}>➔</span>}
                    </React.Fragment>
                  ))}
                  <button 
                    className="icon-btn" 
                    onClick={() => {
                      setGestureSequence([]);
                      setComposedSentence(null);
                      speakText("Sign sequence reset.");
                    }}
                    style={{ marginLeft: 'auto', padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}
                    title="Clear Sequence"
                  >
                    <Trash2 size={12} />
                    <span>Reset</span>
                  </button>
                </div>
              )}

              {/* Continuous Assembled Sentence Banner */}
              <div style={{ padding: '1.75rem', textAlign: 'center', background: 'rgba(236, 72, 153, 0.05)', borderBottom: '1px solid var(--border-color)' }}>
                <span style={{ fontSize: '0.85rem', color: '#f472b6', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 700 }}>
                  Assembled Conversational Translation
                </span>
                <h3 style={{ fontSize: '2.25rem', fontWeight: 800, marginTop: '0.35rem', color: '#fff' }}>
                  {composedSentence?.hi || signResult?.phrase_hi || "नमस्ते (Namaste)"}
                </h3>
                <p style={{ fontSize: '1.25rem', color: 'var(--accent-cyan)', marginTop: '0.25rem', fontWeight: 600 }}>
                  "{composedSentence?.en || signResult?.phrase_en || "Namaste / Hello"}"
                </p>

                <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'center', gap: '0.75rem' }}>
                  <button 
                    className="icon-btn"
                    onClick={() => speakText(composedSentence?.en || signResult?.phrase_en || "Namaste, Hello")}
                  >
                    <Volume2 size={18} />
                    <span>Pronounce English</span>
                  </button>
                  <button 
                    className="icon-btn"
                    onClick={() => speakText(composedSentence?.hi || signResult?.phrase_hi || "नमस्ते")}
                  >
                    <Volume2 size={18} />
                    <span>Pronounce Hindi</span>
                  </button>
                </div>
              </div>

              {/* Camera Actions for Samvaad */}
              <div className="studio-controls">
                <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap' }}>
                  {isSignCameraActive && (
                    <>
                      <button 
                        className="primary-voice-btn" 
                        onClick={captureAndAnalyzeSign}
                        disabled={isAnalyzingSign}
                        style={{ padding: '0.6rem 1.25rem', fontSize: '0.95rem' }}
                      >
                        <Play size={16} />
                        <span>{isAnalyzingSign ? "Reading Hand..." : "Add Gesture to Sentence"}</span>
                      </button>

                      <button 
                        className={`icon-btn ${autoSignTrack ? 'active' : ''}`}
                        onClick={toggleAutoSignTracking}
                        style={{ padding: '0.6rem 1.25rem' }}
                      >
                        <RotateCcw size={16} className={autoSignTrack ? 'spin' : ''} />
                        <span>{autoSignTrack ? "Stop Continuous Read" : "Live Auto-Sentence"}</span>
                      </button>
                    </>
                  )}

                  {/* Native Mobile Camera Snap Button */}
                  <label className="primary-voice-btn" style={{ cursor: 'pointer', background: 'linear-gradient(135deg, #db2777, #ec4899)', padding: '0.6rem 1.25rem', fontSize: '0.95rem' }}>
                    <Smartphone size={18} />
                    <span>Snap Hand Photo with Phone</span>
                    <input 
                      type="file" 
                      accept="image/*" 
                      capture="user" 
                      onChange={handleSignPhoto} 
                      style={{ display: 'none' }} 
                    />
                  </label>

                  <label className="icon-btn" style={{ cursor: 'pointer' }}>
                    <Upload size={16} />
                    <span>Upload Hand Photo</span>
                    <input type="file" accept="image/*" onChange={handleSignPhoto} style={{ display: 'none' }} />
                  </label>
                </div>
              </div>

              {/* Quick Test Simulator */}
              <div style={{ padding: '1.25rem 1.5rem 0.5rem', fontWeight: 600, fontSize: '0.95rem' }}>
                Tap to Test Sample Gesture Pairs (e.g. NAMASTE + HELP, or WATER + HELP):
              </div>
              <div className="vocab-cards-grid">
                {[
                  { key: 'NAMASTE', hi: 'नमस्ते', en: 'Namaste / Hello' },
                  { key: 'HELP', hi: 'मदद', en: 'I Need Help' },
                  { key: 'WATER', hi: 'पानी', en: 'Water Please' },
                  { key: 'THANK_YOU', hi: 'धन्यवाद', en: 'Thank You' },
                  { key: 'YES', hi: 'हाँ', en: 'Yes / Agreed' },
                  { key: 'NO', hi: 'नहीं', en: 'No / Disagree' },
                  { key: 'GOOD', hi: 'बहुत अच्छा', en: 'Good / Super' },
                  { key: 'STOP', hi: 'रुकिए', en: 'Stop / Wait' },
                ].map(item => (
                  <div 
                    key={item.key} 
                    className={`vocab-item-card ${selectedSimGesture === item.key ? 'active' : ''}`}
                    onClick={() => testSimGesture(item.key)}
                    role="button"
                    tabIndex={0}
                  >
                    <div className="vocab-hi">{item.hi}</div>
                    <div className="vocab-en">{item.en}</div>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer" role="contentinfo">
        <p>SAHAAY AI — FastAPI, PyTorch, YOLOv8n, OpenCV, Faster-Whisper / Sarvam AI, Tesseract.js, React & TypeScript</p>
        <p style={{ marginTop: '0.25rem' }}>Assistance prototype only — not a substitute for mobility aids or professional interpreters</p>
      </footer>
    </div>
  );
}

export default App;
