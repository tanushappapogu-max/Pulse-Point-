import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Camera, Loader2, ScanLine, Square, Mic, Settings as SettingsIcon, Flashlight, FlashlightOff } from 'lucide-react';

import { COCO_LABELS, resolveCocoTarget, findClosestCocoLabel, getAliases } from './detection/coco.js';
import { loadYoloModel, runYolo } from './detection/yolo.js';
import { BoxTracker } from './detection/tracker.js';
import { callGeminiBox, callGeminiAutopilot } from './detection/ai.js';

import { computeGuidance } from './guidance/compute.js';
import { Haptics } from './guidance/haptics.js';
import { Speaker } from './guidance/speech.js';

import { getWideCameraStream, setWidestZoom, hasTorchSupport, setTorch, captureJpeg, stopStream } from './lib/camera.js';
import { startListening, isVoiceSupported, isVagueIntent, extractTarget } from './lib/voice.js';
import { loadSettings, saveSettings } from './lib/settings.js';

import Announcer from './ui/Announcer.jsx';
import SettingsSheet from './ui/SettingsSheet.jsx';

// ---- detection cadence -----------------------------------------------------
const HEAVY_COOLDOWN_MS = 2500;       // AI fallback throttle
const NO_FIND_SECONDS = 3;
const ADAPTIVE_FPS_MIN = 4;
const ADAPTIVE_FPS_MAX = 15;
const ADAPTIVE_FPS_INITIAL = 10;
const ADAPTIVE_SLACK_MS = 25;

// sensitivity → haptic gap & re-announce gap
const SENSITIVITY_PROFILES = {
  gentle: { hapticGap: 720, announceGap: 1700 },
  medium: { hapticGap: 520, announceGap: 1200 },
  sharp:  { hapticGap: 360, announceGap: 800 },
};

// guidance signals that should bypass spam-prevention because they change
// the user's required action.
const URGENT_SIGNALS = new Set(['reach', 'closer', 'lost']);

// ---- App -------------------------------------------------------------------
export default function App() {
  // refs that need to be closure-stable
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
  const aiBoxRef          = useRef(null);
  const aiInFlightRef     = useRef(false);
  const localTargetRef    = useRef(null);
  const targetRef         = useRef('');
  const isRunningRef      = useRef(false);
  const lastAnnouncedSignalRef = useRef('');
  const lastAnnouncedTimeRef   = useRef(0);

  // adaptive FPS state
  const lightFpsRef       = useRef(ADAPTIVE_FPS_INITIAL);
  const inferenceWindowRef = useRef([]); // ms timings, last 5

  // tracker (created once)
  const trackerRef = useRef(null);
  if (!trackerRef.current) trackerRef.current = new BoxTracker();

  // haptic + speech engines (created once, settings applied below)
  const hapticsRef = useRef(null);
  if (!hapticsRef.current) hapticsRef.current = new Haptics();
  const speakerRef = useRef(null);
  if (!speakerRef.current) speakerRef.current = new Speaker();

  // ---- React state -------------------------------------------------------
  const [target,        setTargetState]   = useState('');
  const [draftTarget,   setDraftTarget]   = useState('');
  const [status,        setStatus]        = useState('ready');
  const [signal,        setSignal]        = useState('looking');
  const [match,         setMatch]         = useState(null);
  const [error,         setError]         = useState('');
  const [isRunning,     setIsRunning]     = useState(false);
  const [hapticsAvail,  setHapticsAvail]  = useState(true);
  const [isListening,   setIsListening]   = useState(false);
  const [aiLabel,       setAiLabel]       = useState('');
  const [mode,          setMode]          = useState('normal'); // 'normal' | 'autopilot'
  const [autoCands,     setAutoCands]     = useState([]);
  const [autoIdx,       setAutoIdx]       = useState(0);
  const [settings,      setSettings]      = useState(() => loadSettings());
  const [settingsOpen,  setSettingsOpen]  = useState(false);
  const [torchOn,       setTorchOn]       = useState(false);
  const [torchAvail,    setTorchAvail]    = useState(false);
  const [announcement,  setAnnouncement]  = useState('');
  const [announcementUrgent, setAnnouncementUrgent] = useState(false);

  const canVoice = useMemo(() => isVoiceSupported(), []);
  const speechAvail = useMemo(() => speakerRef.current.isAvailable(), []);

  // ---- effects -----------------------------------------------------------
  useEffect(() => {
    setHapticsAvail(hapticsRef.current.isAvailable());
    return () => stopScanner();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // apply settings to engines whenever they change
  useEffect(() => {
    hapticsRef.current.setEnabled(settings.haptics);
    speakerRef.current.setEnabled(settings.speech);
    speakerRef.current.setRate(settings.speechRate);
    saveSettings(settings);
  }, [settings]);

  // ---- target handling ---------------------------------------------------
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

  // ---- voice -------------------------------------------------------------
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

  async function handleVoice(text) {
    setIsListening(false);
    if (isVagueIntent(text)) {
      runAutopilot(text);
      return;
    }
    const extracted = extractTarget(text);
    if (extracted) {
      setTarget(extracted);
      setMode('normal');
      if (!isRunningRef.current) startScanner();
      else setStatus('looking');
    }
  }

  async function runAutopilot(intent) {
    setMode('autopilot');
    setStatus('thinking…');
    setAiLabel('scanning');
    pauseDetectRef.current = true;
    if (!streamRef.current) await startScanner();
    try {
      const frame = captureJpeg(videoRef.current);
      const result = await callGeminiAutopilot(frame, intent);
      if (result?.__error) {
        setMode('normal');
        setError(`AI error: ${result.__error}`);
        setStatus('ready');
      } else if (result?.candidates?.length > 0) {
        setAutoCands(result.candidates);
        setAutoIdx(0);
        setTarget(result.candidates[0]);
        setStatus('looking');
        hapticsRef.current.fire('confirm', true);
        if (!isRunningRef.current) startScanner();
      } else {
        setMode('normal');
        setError('Nothing matched that intent. Try being more specific.');
        setStatus('ready');
      }
    } finally {
      pauseDetectRef.current = false;
      setAiLabel('');
      if (streamRef.current && !loopRef.current) detect();
    }
  }

  function nextAutoCand() {
    const next = autoIdx + 1;
    if (next >= autoCands.length) { setAutoCands([]); setMode('normal'); return; }
    setAutoIdx(next);
    setTarget(autoCands[next]);
    hapticsRef.current.fire('confirm', true);
  }

  // ---- camera lifecycle --------------------------------------------------
  async function startScanner() {
    if (isRunning || startInFlightRef.current) return;
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

  // ---- detect loop -------------------------------------------------------
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

      // ---- heavy YOLO inference (paced) -------------------------------
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

      // ---- find target in predictions ---------------------------------
      const directLabel = resolveCocoTarget(tgt);
      const localInfo = localTargetRef.current;
      const mappedLabel = localInfo?.label || directLabel;
      const cocoMatchRaw = mappedLabel ? findTarget(predictions, mappedLabel, frame, true) : null;
      const cocoMatch = cocoMatchRaw ? {
        ...cocoMatchRaw,
        displayClass: mappedLabel !== tgt ? tgt : cocoMatchRaw.class,
        source: cocoMatchRaw,
      } : null;

      // ---- AI fallback (paced) ----------------------------------------
      const localCanFind = Boolean(mappedLabel);
      const noFindFramesLimit = Math.round(lightFpsRef.current * NO_FIND_SECONDS);
      const needsAi = tgt && !cocoMatch && (!localCanFind || noFindFramesRef.current > noFindFramesLimit);
      const shouldRunHeavy = needsAi && (now - lastHeavyRunRef.current >= HEAVY_COOLDOWN_MS);

      if (shouldRunHeavy && !aiInFlightRef.current) {
        lastHeavyRunRef.current = now;
        aiInFlightRef.current = true;
        setAiLabel('scanning');
        const img = captureJpeg(video);
        callGeminiBox(img, tgt)
          .then(r => {
            if (r?.found && r.x != null) {
              aiBoxRef.current = r;
              setAiLabel('found');
              setTimeout(() => setAiLabel(''), 900);
            } else if (r?.__error) {
              setAiLabel('');
              if (!error) setError(`AI: ${r.__error}`);
            } else {
              aiBoxRef.current = null;
              setAiLabel('');
            }
          })
          .catch(() => setAiLabel(''))
          .finally(() => { aiInFlightRef.current = false; });
      }

      // ---- combine YOLO + AI -----------------------------------------
      let freshMatch = cocoMatch;
      if (!cocoMatch && aiBoxRef.current?.found) {
        const cb = aiBoxRef.current;
        freshMatch = {
          class: tgt,
          displayClass: tgt,
          score: cb.confidence || 0.8,
          bbox: [cb.x * frame.width, cb.y * frame.height, cb.w * frame.width, cb.h * frame.height],
          fromAi: true,
        };
      }

      // ---- feed tracker on fresh detection only ----------------------
      if (freshMatch && ranLight) {
        trackerRef.current.update(
          freshMatch.bbox,
          freshMatch.score,
          freshMatch.displayClass || freshMatch.class,
          now,
          freshMatch.fromAi || false,
        );
      }

      // ---- predict box for THIS render frame -------------------------
      const predicted = trackerRef.current.predict(now);
      const displayMatch = predicted ? {
        class: predicted.label,
        displayClass: predicted.label,
        bbox: predicted.bbox,
        score: predicted.confidence,
        fromAi: predicted.fromAi,
        ageMs: predicted.ageMs,
      } : null;

      draw(predictions, displayMatch);

      // ---- guidance + signaling --------------------------------------
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
          fromAi: displayMatch.fromAi,
        });
        setStatus(g.status);
        setSignal(g.signal);

        if (!foundOnceRef.current) {
          foundOnceRef.current = true;
          hapticsRef.current.fire('found', true);
          setAnnouncement(g.sentence);
          setAnnouncementUrgent(true);
          speakerRef.current.say(g.sentence, { urgent: true, force: true });
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
    if (!signalChanged && !timeOk) return;
    if (URGENT_SIGNALS.has(g.signal) || signalChanged || timeOk) {
      lastAnnouncedSignalRef.current = g.signal;
      lastAnnouncedTimeRef.current = now;
      setAnnouncement(g.sentence);
      setAnnouncementUrgent(URGENT_SIGNALS.has(g.signal) || signalChanged);
      speakerRef.current.say(g.sentence, { urgent: URGENT_SIGNALS.has(g.signal) });
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

  // ---- canvas draw -------------------------------------------------------
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
    ctx.setLineDash(targetMatch.fromAi ? [12, 6] : []);
    ctx.strokeStyle = '#00ff9d';
    ctx.lineWidth   = targetMatch.fromAi ? 5 : 8;
    ctx.strokeRect(x, y, bw, bh);
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    const display = targetMatch.displayClass || targetMatch.class;
    const label = `${display} ${Math.round(targetMatch.score * 100)}%${targetMatch.fromAi ? ' ✦' : ''}`;
    const labelW = Math.min(280, bw);
    ctx.fillStyle = '#00ff9d';
    ctx.fillRect(x, Math.max(0, y - 38), labelW, 38);
    ctx.fillStyle = '#05100d';
    ctx.font = '800 20px system-ui';
    ctx.fillText(label, x + 10, Math.max(26, y - 12));
  }

  // ---- render ------------------------------------------------------------
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

      {aiLabel && (
        <div className={`ai-pill${aiLabel === 'found' ? ' ai-found' : ''}`} role="status">
          {aiLabel === 'scanning'
            ? <><Loader2 size={14} aria-hidden="true" /><span>AI scanning…</span></>
            : <span>✦ AI found it</span>}
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

      <div className="reticle" aria-hidden="true"><span /></div>

      {mode === 'autopilot' && autoCands.length > 1 && autoIdx < autoCands.length - 1 && (
        <button className="next-cand-btn" onClick={nextAutoCand} aria-label={`Try next candidate, ${autoCands[autoIdx + 1] || ''}`}>
          Next option →
        </button>
      )}

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
          {mode === 'autopilot' && autoCands.length > 0 && (
            <span className="auto-label">AUTO</span>
          )}
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
        <strong>{status}</strong>
        <span>
          {match
            ? `${match.direction} · ${match.distance}${match.fromAi ? ' · ✦AI' : ''}`
            : aiLabel === 'scanning' ? 'AI scanning…'
            : hapticsAvail ? 'scan slowly' : 'visual guidance mode'}
        </span>
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

// ---- target-finding helpers (kept here so they share the module's COCO knowledge) ---
function findTarget(predictions, target, frame, allowCandidate = true) {
  const aliases = getAliases(target);
  const exact = predictions
    .filter(p => aliases.includes(p.class.toLowerCase()))
    .sort((a, b) => b.score - a.score)[0];
  if (exact) return exact;
  return allowCandidate ? findCenterCandidate(predictions, frame) : null;
}

function findCenterCandidate(predictions, frame) {
  const cx = frame.width / 2, cy = frame.height / 2;
  return predictions
    .map(p => {
      const [x, y, w, h] = p.bbox;
      const pcx = x + w / 2, pcy = y + h / 2;
      const nd = Math.hypot((pcx - cx) / frame.width, (pcy - cy) / frame.height);
      const area = (w * h) / (frame.width * frame.height);
      return {
        ...p,
        score: p.score * 0.72,
        centerScore: p.score + area * 1.6 - nd * 1.4,
        isCandidate: true,
      };
    })
    .filter(p => p.centerScore > 0.22 && p.score > 0.22)
    .sort((a, b) => b.centerScore - a.centerScore)[0];
}
