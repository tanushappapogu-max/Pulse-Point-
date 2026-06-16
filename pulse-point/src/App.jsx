import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Camera, Loader2, ScanLine, Square, Mic, Settings as SettingsIcon, Flashlight, FlashlightOff } from 'lucide-react';

import { COCO_LABELS, resolveCocoTarget, findClosestCocoLabel, TARGET_ALIASES, normalizeTargetText } from './detection/coco.js';
import { loadYoloModel, runYolo } from './detection/yolo.js';
import { BoxTracker } from './detection/tracker.js';
import { KNOWN_OBJECTS, suggestObjects } from './detection/objectList.js';

import { computeGuidance } from './guidance/compute.js';
import { Haptics } from './guidance/haptics.js';
import { Speaker } from './guidance/speech.js';

import { getWideCameraStream, setWidestZoom, hasTorchSupport, setTorch, captureJpeg, stopStream } from './lib/camera.js';
import { startListening, isVoiceSupported, extractTarget } from './lib/voice.js';
import { loadSettings, saveSettings } from './lib/settings.js';

import Announcer from './ui/Announcer.jsx';
import SettingsSheet from './ui/SettingsSheet.jsx';

const HEAVY_COOLDOWN_MS = 2500;
const NO_FIND_SECONDS = 3;
const ADAPTIVE_FPS_MIN = 4;
const ADAPTIVE_FPS_MAX = 15;
const ADAPTIVE_FPS_INITIAL = 10;
const ADAPTIVE_SLACK_MS = 25;

const SENSITIVITY_PROFILES = {
  gentle: { hapticGap: 720, announceGap: 1700 },
  medium: { hapticGap: 520, announceGap: 1200 },
  sharp:  { hapticGap: 360, announceGap: 800 },
};

const URGENT_SIGNALS = new Set(['reach', 'closer', 'lost']);

export default function App() {
  const videoRef          = useRef(null);
  const canvasRef         = useRef(null);
  const modelRef          = useRef(null);
  const streamRef         = useRef(null);
  const loopRef           = useRef(null);
  const startInFlightRef  = useRef(false);
  const detectingRef      = useRef(false);
  const pauseDetectRef    = useRef(false);
  const lastLightRunRef   = useRef(0);
  const lastHeavyRunRef   = useRef(0);
  const lastPredsRef      = useRef([]);
  const prevAreaRef       = useRef(0);
  const foundOnceRef      = useRef(false);
  const noFindFramesRef   = useRef(0);
  const suggestions       = useRef([]);
  const localTargetRef    = useRef(null);
  const targetRef         = useRef('');
  const isRunningRef      = useRef(false);
  const lastAnnouncedSignalRef = useRef('');
  const lastAnnouncedTimeRef   = useRef(0);

  const lightFpsRef       = useRef(ADAPTIVE_FPS_INITIAL);
  const inferenceWindowRef = useRef([]);

  const trackerRef = useRef(null);
  if (!trackerRef.current) trackerRef.current = new BoxTracker();

  const hapticsRef = useRef(null);
  if (!hapticsRef.current) hapticsRef.current = new Haptics();
  const speakerRef = useRef(null);
  if (!speakerRef.current) speakerRef.current = new Speaker();

  const [target,        setTargetState]   = useState('');
  const [draftTarget,   setDraftTarget]   = useState('');
  const [status,        setStatus]        = useState('ready');
  const [signal,        setSignal]        = useState('looking');
  const [match,         setMatch]         = useState(null);
  const [error,         setError]         = useState('');
  const [isRunning,     setIsRunning]     = useState(false);
  const [hapticsAvail,  setHapticsAvail]  = useState(true);
  const [isListening,   setIsListening]   = useState(false);
  const [mode,          setMode]          = useState('normal');
  const [settings,      setSettings]      = useState(() => loadSettings());
  const [settingsOpen,  setSettingsOpen]  = useState(false);
  const [torchOn,       setTorchOn]       = useState(false);
  const [torchAvail,    setTorchAvail]    = useState(false);
  const [announcement,  setAnnouncement]  = useState('');
  const [announcementUrgent, setAnnouncementUrgent] = useState(false);

  const canVoice = useMemo(() => isVoiceSupported(), []);
  const speechAvail = useMemo(() => speakerRef.current.isAvailable(), []);

  useEffect(() => {
    setHapticsAvail(hapticsRef.current.isAvailable());
    return () => stopScanner();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    hapticsRef.current.setEnabled(settings.haptics);
    speakerRef.current.setEnabled(settings.speech);
    speakerRef.current.setRate(settings.speechRate);
    saveSettings(settings);
  }, [settings]);

  function setTarget(t) {
    targetRef.current = t;
    setTargetState(t);
    setDraftTarget(t);
    foundOnceRef.current  = false;
    noFindFramesRef.current = 0;
    lastLightRunRef.current = 0;
    lastHeavyRunRef.current = 0;
    lastPredsRef.current = [];
    aiBoxRef.current  = null;
    aiInFlightRef.current = false;
    localTargetRef.current = null;
    trackerRef.current.reset();
    lastAnnouncedSignalRef.current = '';
    setAiLabel('');
    resolveLocalTarget(t);
  }

  function resolveLocalTarget(tgt) {
    if (!tgt) return;
    const direct = resolveCocoTarget(tgt);
    if (direct) {
      localTargetRef.current = { label: direct, score: 1, source: 'alias' };
      return;
    }
    const closest = findClosestCocoLabel(tgt);
    if (closest?.label) {
      localTargetRef.current = { label: closest.label, score: closest.score, source: 'fuzzy' };
    }
  }

  function submitTypedTarget() {
    const text = draftTarget.trim();
    if (!text) return;
    setError('');
    setTarget(text);
    setMode('normal');
    if (!isRunningRef.current) startScanner();
    else setStatus('looking');
  }

  function handleScanClick() {
    if (isRunningRef.current) { stopScanner(); return; }
    if (draftTarget.trim()) { submitTypedTarget(); return; }
    startScanner();
  }

  function handleTypedKeyDown(e) {
    if (e.key === 'Enter') { e.preventDefault(); submitTypedTarget(); }
  }

  function startVoice() {
    if (!canVoice) {
      setError('Voice input is not supported on this device. Type below.');
      return;
    }
    setError('');
    setIsListening(true);
    if (hapticsRef.current.isAvailable()) navigator.vibrate?.([80]);
    startListening({
      onResult: handleVoice,
      onError: (err) => {
        setIsListening(false);
        if (err === 'not-allowed' || err === 'service-not-allowed') {
          setError('Microphone blocked. Allow microphone access in browser settings.');
        } else if (err === 'no-speech') {
          setError('No speech heard. Tap to retry.');
        } else if (err === 'network') {
          setError('Voice service unavailable. Check connection or type below.');
        } else if (err === 'not-supported') {
          setError('Voice input is not supported in this browser. Type below.');
        } else {
          setError('Voice input failed. Type below.');
        }
      },
      onEnd: () => setIsListening(false),
    });
  }

  function handleVoice(text) {
    setIsListening(false);
    const extracted = extractTarget(text) || text.trim();
    if (extracted) {
      setTarget(extracted);
      setMode('normal');
      if (!isRunningRef.current) startScanner();
      else setStatus('looking');
    }
  }

  async function startScanner() {
    if (isRunningRef.current || startInFlightRef.current) return;
    startInFlightRef.current = true;
    setError('');
    setStatus('camera');
    setMatch(null);
    setSignal('looking');
    prevAreaRef.current     = 0;
    foundOnceRef.current    = false;
    noFindFramesRef.current = 0;
    lastLightRunRef.current = 0;
    lastHeavyRunRef.current = 0;
    lastPredsRef.current    = [];
    aiBoxRef.current        = null;
    aiInFlightRef.current   = false;
    trackerRef.current.reset();
    lastAnnouncedSignalRef.current = '';

    try {
      const stream = await getWideCameraStream();
      streamRef.current        = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      await setWidestZoom(stream);
      setTorchAvail(hasTorchSupport(stream));
      setTorchOn(false);

      isRunningRef.current = true;
      setIsRunning(true);
      setStatus('loading');
      setAnnouncement('Loading object detection model…');
      setAnnouncementUrgent(false);
      hapticsRef.current.fire('looking', true);

      if (!modelRef.current) {
        modelRef.current = await loadYoloModel();
      }

      // race: user pressed Stop while model loaded — bail out
      if (!streamRef.current) return;

      setStatus('looking');
      setAnnouncement(target ? `Looking for ${target}.` : 'Camera active. Say or type a target.');
      detect();
    } catch (err) {
      setStatus('blocked');
      setError(err?.message || 'Camera blocked. Allow camera access and try again.');
      isRunningRef.current = false;
      setIsRunning(false);
    } finally {
      startInFlightRef.current = false;
    }
  }

  function stopScanner() {
    if (loopRef.current) cancelAnimationFrame(loopRef.current);
    loopRef.current = null;
    hapticsRef.current.cancel();
    speakerRef.current.cancel();
    stopStream(streamRef.current);
    streamRef.current = null;
    pauseDetectRef.current = false;
    detectingRef.current = false;
    isRunningRef.current = false;
    setIsRunning(false);
    setStatus('ready');
    aiBoxRef.current = null;
    trackerRef.current.reset();
    setTorchOn(false);
    setTorchAvail(false);
  }

  async function toggleTorch() {
    const ok = await setTorch(streamRef.current, !torchOn);
    if (ok) setTorchOn(t => !t);
  }

  function recordInferenceTime(ms) {
    const w = inferenceWindowRef.current;
    w.push(ms);
    if (w.length > 5) w.shift();
    const avg = w.reduce((s, n) => s + n, 0) / w.length;
    const targetFps = 1000 / (avg + ADAPTIVE_SLACK_MS);
    lightFpsRef.current = Math.max(ADAPTIVE_FPS_MIN, Math.min(ADAPTIVE_FPS_MAX, targetFps));
  }

  async function detect() {
    if (detectingRef.current) {
      if (streamRef.current && !pauseDetectRef.current) {
        loopRef.current = requestAnimationFrame(detect);
      }
      return;
    }

    detectingRef.current = true;
    try {
      const video = videoRef.current;
      const model = modelRef.current;

      if (!video || !model || video.readyState < 2) return;
      if (pauseDetectRef.current) return;

      const tgt = targetRef.current;
      if (tgt && !localTargetRef.current) resolveLocalTarget(tgt);

      const frame = { width: video.videoWidth || 640, height: video.videoHeight || 480 };
      const now = performance.now();

      const lightInterval = 1000 / lightFpsRef.current;
      const ranLight = now - lastLightRunRef.current >= lightInterval;

      let predictions = lastPredsRef.current;
      if (ranLight) {
        lastLightRunRef.current = now;
        const t0 = performance.now();
        try {
          predictions = await runYolo(video, model);
          lastPredsRef.current = predictions;
          recordInferenceTime(performance.now() - t0);
        } catch (e) {
          // eslint-disable-next-line no-console
          console.warn('YOLO inference failed', e);
          predictions = lastPredsRef.current;
        }
      }

      const directLabel = resolveCocoTarget(tgt);
      const localInfo = localTargetRef.current;
      const mappedLabel = localInfo?.label || directLabel;
      const priorTrack = trackerRef.current.predict(now);
      const cocoMatchRaw = mappedLabel ? findTarget(predictions, mappedLabel, priorTrack?.bbox, frame) : null;
      const cocoMatch = cocoMatchRaw ? {
        ...cocoMatchRaw,
        displayClass: mappedLabel !== tgt ? tgt : cocoMatchRaw.class,
        source: cocoMatchRaw,
      } : null;

      let freshMatch = cocoMatch;

      if (freshMatch && ranLight) {
        trackerRef.current.update(
          freshMatch.bbox,
          freshMatch.score,
          freshMatch.displayClass || freshMatch.class,
          now,
          false,
        );
      }

      const predicted = trackerRef.current.predict(now);
      const displayMatch = predicted ? {
        class: predicted.label,
        displayClass: predicted.label,
        bbox: predicted.bbox,
        score: predicted.confidence,
        fromAi: false,
        ageMs: predicted.ageMs,
      } : null;

      draw(predictions, displayMatch);

      if (!displayMatch) {
        if (ranLight) noFindFramesRef.current++;
        setMatch(null);
        if (tgt) {
          setStatus('looking');
          setSignal('looking');
          hapticsRef.current.fire('looking');
          throttledAnnounce(`Looking for ${tgt}.`, false);
        }
        prevAreaRef.current = 0;
      } else {
        if (ranLight && freshMatch) noFindFramesRef.current = 0;
        const g = computeGuidance(displayMatch, frame, prevAreaRef.current);
        prevAreaRef.current = g.area;

        setMatch({
          name: displayMatch.displayClass || displayMatch.class,
          score: displayMatch.score,
          direction: g.direction,
          distance: g.distance,
          distanceMeters: g.distanceMeters,
          fromAi: false,
        });
        setStatus(g.status);
        setSignal(g.signal);

        if (!foundOnceRef.current) {
          foundOnceRef.current = true;
          hapticsRef.current.fire('found', true);
          setAnnouncement(g.sentence);
          setAnnouncementUrgent(true);
          speakerRef.current.say(g.speechPhrase, { urgent: true, force: true });
          lastAnnouncedSignalRef.current = g.signal;
          lastAnnouncedTimeRef.current = now;
        } else {
          hapticsRef.current.fire(g.signal);
          maybeAnnounce(g, now);
        }
      }
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error('detect loop error', e);
    } finally {
      detectingRef.current = false;
      if (streamRef.current && !pauseDetectRef.current) {
        loopRef.current = requestAnimationFrame(detect);
      } else {
        // null out so autopilot resume can correctly relaunch the loop
        loopRef.current = null;
      }
    }
  }

  function maybeAnnounce(g, now) {
    const profile = SENSITIVITY_PROFILES[settings.sensitivity] || SENSITIVITY_PROFILES.medium;
    const signalChanged = g.signal !== lastAnnouncedSignalRef.current;
    const timeOk = now - lastAnnouncedTimeRef.current >= profile.announceGap;
    
    // Only announce if: truly urgent, or enough time has passed (not on every direction change)
    if (!URGENT_SIGNALS.has(g.signal) && !timeOk) return;
    
    if (URGENT_SIGNALS.has(g.signal) || signalChanged || timeOk) {
      lastAnnouncedSignalRef.current = g.signal;
      lastAnnouncedTimeRef.current = now;
      setAnnouncement(g.sentence);
      setAnnouncementUrgent(URGENT_SIGNALS.has(g.signal) || signalChanged);
      
      // Only force interrupt for truly urgent signals (reach, closer)
      const shouldForce = URGENT_SIGNALS.has(g.signal);
      speakerRef.current.say(g.speechPhrase, { 
        urgent: URGENT_SIGNALS.has(g.signal),
        force: shouldForce
      });
    }
  }

  function throttledAnnounce(text, urgent) {
    const profile = SENSITIVITY_PROFILES[settings.sensitivity] || SENSITIVITY_PROFILES.medium;
    const now = performance.now();
    if (text === announcement && now - lastAnnouncedTimeRef.current < profile.announceGap) return;
    lastAnnouncedTimeRef.current = now;
    setAnnouncement(text);
    setAnnouncementUrgent(urgent);
  }

  function draw(predictions, targetMatch) {
    const canvas = canvasRef.current;
    const video  = videoRef.current;
    if (!canvas || !video) return;

    const W = video.videoWidth || 640, H = video.videoHeight || 480;
    const displayW = canvas.clientWidth || W;
    const displayH = canvas.clientHeight || H;
    canvas.width = displayW;
    canvas.height = displayH;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, displayW, displayH);

    const scale = Math.max(displayW / W, displayH / H);
    const offsetX = (W * scale - displayW) / 2;
    const offsetY = (H * scale - displayH) / 2;
    const mapBox = ([x, y, w, h]) => [x * scale - offsetX, y * scale - offsetY, w * scale, h * scale];

    if (settings.showAllBoxes) {
      predictions.forEach(p => {
        if (targetMatch?.source === p) return;
        const [x, y, w, h] = mapBox(p.bbox);
        ctx.strokeStyle = 'rgba(255,255,255,0.28)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([]);
        ctx.strokeRect(x, y, w, h);
      });
    }

    if (!targetMatch) return;

    const [x, y, bw, bh] = mapBox(targetMatch.bbox);
    // visualize extrapolation: fade box as ageMs grows
    const opacity = targetMatch.ageMs > 100 ? 0.7 : 1;
    ctx.globalAlpha = opacity;
    ctx.strokeStyle = '#00ff9d';
    ctx.lineWidth   = 8;
    ctx.strokeRect(x, y, bw, bh);
    ctx.globalAlpha = 1;

    const display = targetMatch.displayClass || targetMatch.class;
    const label = `${display} ${Math.round(targetMatch.score * 100)}%`;
    const labelW = Math.min(280, bw);
    ctx.fillStyle = '#00ff9d';
    ctx.fillRect(x, Math.max(0, y - 38), labelW, 38);
    ctx.fillStyle = '#05100d';
    ctx.font = '800 20px system-ui';
    ctx.fillText(label, x + 10, Math.max(26, y - 12));
  }

  return (
    <main className={`scanner signal-${signal}`}>
      <video ref={videoRef} playsInline muted aria-hidden="true" />
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={match ? `Detection overlay: ${match.name} ${match.direction}, ${match.distance}` : 'Detection overlay'}
      />

      <Announcer message={announcement} urgent={announcementUrgent} />

      {!isRunning && (
        <button className="start-button" type="button" onClick={startScanner} aria-label="Start camera scanner">
          <Camera size={30} aria-hidden="true" />
          <span>Start</span>
        </button>
      )}

      {/* Top-right control rail */}
      <div className="top-rail" role="group" aria-label="Quick controls">
        {torchAvail && (
          <button
            type="button"
            className={`rail-btn${torchOn ? ' active' : ''}`}
            onClick={toggleTorch}
            aria-label={torchOn ? 'Turn flashlight off' : 'Turn flashlight on'}
            aria-pressed={torchOn}
          >
            {torchOn ? <Flashlight size={18} aria-hidden="true" /> : <FlashlightOff size={18} aria-hidden="true" />}
          </button>
        )}
        <button
          type="button"
          className="rail-btn"
          onClick={() => setSettingsOpen(true)}
          aria-label="Open settings"
        >
          <SettingsIcon size={18} aria-hidden="true" />
        </button>
      </div>

      {status === 'loading' && (
        <div className="loading-pill" role="status">
          <Loader2 size={18} aria-hidden="true" />
          <span>Loading…</span>
        </div>
      )}

      {error && (
        <div className="error-pill" role="alert">
          {error}
          <button
            type="button"
            className="error-dismiss"
            onClick={() => setError('')}
            aria-label="Dismiss error"
          >
            ×
          </button>
        </div>
      )}

      <div className="scan-sweep" aria-hidden="true" />

      <div className="reticle" aria-hidden="true">
        <div className="reticle-inner">
          <div className="reticle-corner reticle-corner--tl" />
          <div className="reticle-corner reticle-corner--tr" />
          <div className="reticle-corner reticle-corner--bl" />
          <div className="reticle-corner reticle-corner--br" />
          <div className="reticle-dot" />
          <div className="reticle-scan" />
        </div>
      </div>

      <div className="target-bar">
        <button
          className={`mic-btn${isListening ? ' mic-active' : ''}${!canVoice ? ' mic-disabled' : ''}`}
          type="button"
          onClick={startVoice}
          aria-label={isListening ? 'Listening' : canVoice ? 'Tap to speak' : 'Voice not supported, type instead'}
          disabled={!canVoice}
        >
          <Mic size={22} aria-hidden="true" />
        </button>

        <div className="target-display" aria-label="Target object">
          <input
            className="target-input"
            type="text"
            value={draftTarget}
            onChange={e => setDraftTarget(e.target.value)}
            onKeyDown={handleTypedKeyDown}
            placeholder="say or type what to find…"
            autoComplete="off"
            autoCapitalize="off"
            enterKeyHint="go"
            aria-label="Type a target to find"
          />
          <button className="go-btn" type="button" onClick={submitTypedTarget} aria-label="Submit target">
            Go
          </button>
        </div>

        <button
          type="button"
          className="scan-btn"
          onClick={handleScanClick}
          aria-label={isRunning ? 'Stop scanning' : 'Start scanning'}
        >
          {isRunning ? <Square size={18} aria-hidden="true" /> : <ScanLine size={20} aria-hidden="true" />}
        </button>
      </div>

      <div className="signal-strip" aria-live="polite">
        <div className="signal-strip-dot" aria-hidden="true" />
        <div className="signal-strip-text">
          <strong>{status}</strong>
          <span>
            {match
              ? `${match.direction} · ${match.distance}`
              : hapticsAvail ? 'scan slowly' : 'visual guidance mode'}
          </span>
        </div>
      </div>

      <SettingsSheet
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={settings}
        onChange={setSettings}
        hapticsAvailable={hapticsAvail}
        speechAvailable={speechAvail}
      />
    </main>
  );
}

function findTarget(predictions, target, priorBox = null, frame = null) {
  const norm = normalizeTargetText(target);
  const aliases = TARGET_ALIASES[norm] ? [TARGET_ALIASES[norm]] : [norm];

  return predictions
    .filter(p => aliases.includes(p.class.toLowerCase()))
    .sort((a, b) => {
      if (!priorBox || !frame) return b.score - a.score;

      const continuity = box => {
        const [x, y, w, h] = box;
        const [px, py, pw, ph] = priorBox;
        const centerDistance = Math.hypot(
          ((x + w / 2) - (px + pw / 2)) / frame.width,
          ((y + h / 2) - (py + ph / 2)) / frame.height,
        );
        const x1 = Math.max(x, px);
        const y1 = Math.max(y, py);
        const x2 = Math.min(x + w, px + pw);
        const y2 = Math.min(y + h, py + ph);
        const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const union = w * h + pw * ph - inter;
        const iou = union <= 0 ? 0 : inter / union;
        return iou * 0.65 + Math.max(0, 1 - centerDistance * 4) * 0.35;
      };

      return (b.score + continuity(b.bbox) * 1.8) - (a.score + continuity(a.bbox) * 1.8);
    })[0];
}
