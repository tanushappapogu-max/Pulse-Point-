import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import '@tensorflow/tfjs';
import { Camera, Loader2, ScanLine, Square, Mic, AlertTriangle } from 'lucide-react';
import './styles.css';

// ── All 80 COCO-SSD classes — used to decide when Claude fallback is needed ────
const COCO_KNOWN = new Set([
  'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
  'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
  'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
  'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
  'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
  'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
  'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake',
  'chair','couch','potted plant','bed','dining table','toilet','tv','laptop',
  'mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink',
  'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
  'toothbrush',
]);

const TARGET_ALIASES = {
  phone: ['cell phone'], iphone: ['cell phone'], android: ['cell phone'],
  'my phone': ['cell phone'], 'cell phone': ['cell phone'],
  'computer mouse': ['mouse'], trackpad: ['mouse'], 'my mouse': ['mouse'],
  tv: ['tv'], television: ['tv'], monitor: ['tv'],
  sofa: ['couch'], couch: ['couch'],
  laptop: ['laptop'], computer: ['laptop'],
  remote: ['remote'], 'tv remote': ['remote'],
};

const HAPTIC = {
  looking: [24, 260],
  found:   [170, 80, 170, 80, 260],
  left:    [80, 45, 80],
  right:   [80, 45, 80, 45, 80],
  closer:  [45, 38, 45, 38, 45],
  locked:  [260],
  reach:   [360, 80, 360],
  lost:    [35, 120, 35],
  sos:     [500, 100, 500, 100, 500],
  confirm: [200, 80, 200],
};

// ── Claude API ─────────────────────────────────────────────────────────────────
// Calls Claude directly from the browser. API key must be in .env as VITE_ANTHROPIC_API_KEY.

async function callClaude(imageBase64, prompt) {
  const key = import.meta.env.VITE_ANTHROPIC_API_KEY;
  if (!key) return null;

  const content = [];
  if (imageBase64) {
    content.push({
      type: 'image',
      source: {
        type: 'base64',
        media_type: 'image/jpeg',
        data: imageBase64.replace(/^data:image\/[a-z]+;base64,/, ''),
      },
    });
  }
  content.push({ type: 'text', text: prompt });

  try {
    const res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'x-api-key': key,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
        'anthropic-dangerous-direct-browser-access': 'true',
      },
      body: JSON.stringify({
        model: 'claude-opus-4-7',
        max_tokens: 512,
        system: 'You are a spatial object detection assistant. Respond ONLY with valid JSON. Never add explanation outside the JSON.',
        messages: [{ role: 'user', content }],
      }),
    });
    const data = await res.json();
    const text = data.content?.find(b => b.type === 'text')?.text || '';
    const s = text.indexOf('{'), e = text.lastIndexOf('}') + 1;
    if (s >= 0 && e > s) return JSON.parse(text.slice(s, e));
  } catch { /* silent — COCO handles it */ }
  return null;
}

function captureJpeg(video, quality = 0.75) {
  const c = document.createElement('canvas');
  c.width  = video.videoWidth  || 640;
  c.height = video.videoHeight || 480;
  c.getContext('2d').drawImage(video, 0, 0);
  return c.toDataURL('image/jpeg', quality);
}

// ── App ────────────────────────────────────────────────────────────────────────

function App() {
  const videoRef          = useRef(null);
  const canvasRef         = useRef(null);
  const modelRef          = useRef(null);
  const streamRef         = useRef(null);
  const loopRef           = useRef(null);
  const lastHapticRef     = useRef(0);
  const lastSignalRef     = useRef('');
  const prevAreaRef       = useRef(0);
  const foundOnceRef      = useRef(false);
  const noFindFramesRef   = useRef(0);
  const claudeCoolRef     = useRef(0);   // frames since last Claude call
  const claudeBoxRef      = useRef(null); // last Claude-returned bbox {x,y,w,h,confidence}
  const targetRef         = useRef('');  // always-current target for the detect loop

  const [target,        setTargetState]   = useState('');
  const [status,        setStatus]        = useState('ready');
  const [signal,        setSignal]        = useState('looking');
  const [match,         setMatch]         = useState(null);
  const [error,         setError]         = useState('');
  const [isRunning,     setIsRunning]     = useState(false);
  const [hapticsOk,     setHapticsOk]     = useState(true);
  const [isListening,   setIsListening]   = useState(false);
  const [claudeLabel,   setClaudeLabel]   = useState(''); // 'scanning' | 'found' | ''
  const [mode,          setMode]          = useState('normal'); // 'normal' | 'autopilot' | 'sos'
  const [autoCands,     setAutoCands]     = useState([]);
  const [autoIdx,       setAutoIdx]       = useState(0);

  useEffect(() => {
    setHapticsOk('vibrate' in navigator);
    return () => stopScanner();
  }, []);

  // Keep ref in sync so the detect animation loop always reads the latest target
  function setTarget(t) {
    targetRef.current = t;
    setTargetState(t);
    foundOnceRef.current  = false;
    noFindFramesRef.current = 0;
    claudeBoxRef.current  = null;
    claudeCoolRef.current = 0;
  }

  // ── Voice input ──────────────────────────────────────────────────────────────

  function startListening() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { setError('Use Chrome for voice'); return; }
    setError('');
    const r = new SR();
    r.continuous = false; r.interimResults = false; r.lang = 'en-US';
    setIsListening(true);
    r.onresult = e => { setIsListening(false); handleVoice(e.results[0][0].transcript.trim()); };
    r.onerror  = ()  => { setIsListening(false); setError('Mic error — tap to retry'); };
    r.onend    = ()  => setIsListening(false);
    navigator.vibrate?.([80]);
    r.start();
  }

  async function handleVoice(text) {
    const lower = text.toLowerCase();

    // Emergency
    if (/\b(sos|emergency|help|exit|danger|fire|lost)\b/i.test(lower)) {
      triggerSOS(); return;
    }

    // Vague / intent-based → autopilot
    const isVague =
      /\b(something|anything)\b/i.test(lower) ||
      /\b(eat|food|hungry|snack|drink|thirsty)\b/i.test(lower) ||
      /\b(write|writing|draw)\b/i.test(lower) ||
      /\b(call|text)\b/i.test(lower);

    if (isVague) {
      runAutopilot(text); return;
    }

    // Direct target — strip navigation phrases
    const extracted = lower
      .replace(/where\s+is\s+(my\s+|the\s+)?/g, '')
      .replace(/find\s+(my\s+|the\s+|a\s+)?/g, '')
      .replace(/show\s+me\s+(my\s+|the\s+)?/g, '')
      .replace(/i\s+need\s+(a\s+|my\s+)?/g, '')
      .replace(/look\s+for\s+(my\s+|the\s+|a\s+)?/g, '')
      .trim();

    if (extracted) {
      setTarget(extracted);
      setMode('normal');
      if (!isRunning) startScanner(); else setStatus('looking');
    }
  }

  // ── Autopilot — intent-based object ranking ──────────────────────────────────

  async function runAutopilot(intent) {
    setMode('autopilot');
    setStatus('thinking…');
    setClaudeLabel('scanning');
    if (!streamRef.current) await startScanner();

    const frame = captureJpeg(videoRef.current);
    const result = await callClaude(frame,
      `User said: "${intent}"
Look at this image. Identify 2–4 real visible objects that best match the user's intent, ranked by how obvious/accessible they are.
CRITICAL: Only list objects that are clearly visible as distinct items — NOT walls, floors, tables, or surfaces themselves. Only things ON surfaces or in the scene as separate objects.
Return JSON: {"candidates":["item1","item2"],"positions":["short position like 'center table'","short position"]}
If nothing matches, return {"candidates":[],"positions":[]}`
    );

    setClaudeLabel('');
    if (result?.candidates?.length > 0) {
      setAutoCands(result.candidates);
      setAutoIdx(0);
      setTarget(result.candidates[0]);
      setStatus('looking');
      vibe('confirm', true);
      if (!isRunning) startScanner();
    } else {
      setMode('normal');
      setError('Nothing found for that. Try being more specific.');
      setStatus('ready');
    }
  }

  function nextAutoCand() {
    const next = autoIdx + 1;
    if (next >= autoCands.length) { setAutoCands([]); setMode('normal'); return; }
    setAutoIdx(next);
    setTarget(autoCands[next]);
    vibe('confirm', true);
  }

  // ── Emergency SOS ────────────────────────────────────────────────────────────

  async function triggerSOS() {
    setMode('sos');
    setStatus('🆘 SOS — locating exit');
    vibe('sos', true);
    setClaudeLabel('scanning');
    if (!streamRef.current) await startScanner();

    const frame = captureJpeg(videoRef.current);
    const result = await callClaude(frame,
      `EMERGENCY: Find the fastest path to safety — a door, hallway, open space, or exit.
Ignore everything except exits and large open paths.
Return JSON: {
  "direction": "left or right or forward or backward",
  "description": "one short spoken instruction max 10 words",
  "obstacle": "obstacle name or null"
}
If no exit visible: {"direction":"forward","description":"Follow the wall to your right","obstacle":null}`
    );

    setClaudeLabel('');
    if (result) {
      const sig = result.direction === 'left' ? 'left' : result.direction === 'right' ? 'right' : 'locked';
      setSignal(sig);
      setStatus(result.description || 'Move to nearest open space');
      setMatch({
        name: '🆘 EXIT',
        direction: result.direction,
        distance: result.obstacle ? `⚠ avoid ${result.obstacle}` : 'keep moving',
      });
    }
  }

  function cancelSOS() {
    setMode('normal'); setStatus('ready'); setSignal('looking'); setMatch(null);
  }

  // ── Camera + detection loop ──────────────────────────────────────────────────

  async function startScanner() {
    setError('');
    setStatus('camera');
    setMatch(null);
    setSignal('looking');
    prevAreaRef.current     = 0;
    foundOnceRef.current    = false;
    noFindFramesRef.current = 0;
    claudeBoxRef.current    = null;
    claudeCoolRef.current   = 0;

    try {
      const stream = await getWideCameraStream();
      streamRef.current        = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      await setWidestZoom(stream);

      setIsRunning(true);
      setStatus('loading');
      vibe('looking', true);

      if (!modelRef.current) {
        modelRef.current = await cocoSsd.load({ base: 'mobilenet_v2' });
      }
      setStatus('looking');
      detect();
    } catch (err) {
      setStatus('blocked');
      setError(err?.message || 'Camera blocked — allow camera access');
    }
  }

  function stopScanner() {
    if (loopRef.current) cancelAnimationFrame(loopRef.current);
    loopRef.current = null;
    navigator.vibrate?.(0);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    setIsRunning(false);
    setStatus('ready');
    claudeBoxRef.current = null;
  }

  async function detect() {
    const video = videoRef.current;
    const model = modelRef.current;

    if (!video || !model || video.readyState < 2) {
      loopRef.current = requestAnimationFrame(detect); return;
    }

    const tgt   = targetRef.current;
    const frame = { width: video.videoWidth || 640, height: video.videoHeight || 480 };

    const predictions = await model.detect(video, 60, 0.18);
    const cocoMatch   = tgt ? findTarget(predictions, tgt, frame) : null;

    // ── Claude fallback logic ─────────────────────────────────────────────────
    // Trigger when COCO can't find it AND:
    //   • the target isn't in COCO's 80 classes at all (e.g. pencil, pen, keys), OR
    //   • COCO has been running 90+ frames (~3s) without finding it
    claudeCoolRef.current++;
    const cocoCanFind = isCocoKnown(tgt);
    const needsClaude = tgt && !cocoMatch && (!cocoCanFind || noFindFramesRef.current > 90);

    if (needsClaude && claudeCoolRef.current >= 55) {
      claudeCoolRef.current = 0;
      setClaudeLabel('scanning');
      const img = captureJpeg(video);
      callClaude(img,
        `Where is the "${tgt}" in this image?
CRITICAL rules:
- Only detect "${tgt}" if it is a DISTINCT FOREGROUND OBJECT — something separate from walls, floors, tables, and surfaces.
- A pencil lying ON a table = object. The table itself = NOT the object. The wall behind = NOT the object.
- If you see it, return its bounding box as fractions of image size (0.0 to 1.0).
Return JSON: {"found":true,"x":0.25,"y":0.30,"w":0.12,"h":0.06,"confidence":0.85}
If NOT visible or only background: {"found":false}
JSON only. No other text.`
      ).then(r => {
        if (r?.found && r.x != null) {
          claudeBoxRef.current = r;
          setClaudeLabel('found');
          setTimeout(() => setClaudeLabel(''), 900);
        } else {
          claudeBoxRef.current = null;
          setClaudeLabel('');
        }
      }).catch(() => setClaudeLabel(''));
    }

    // Build effective match — COCO result OR Claude-derived bounding box
    let effectiveMatch = cocoMatch;
    if (!cocoMatch && claudeBoxRef.current?.found) {
      const cb = claudeBoxRef.current;
      effectiveMatch = {
        class: tgt,
        score: cb.confidence || 0.8,
        bbox: [cb.x * frame.width, cb.y * frame.height, cb.w * frame.width, cb.h * frame.height],
        fromClaude: true,
      };
    }

    draw(predictions, effectiveMatch);

    if (!effectiveMatch) {
      noFindFramesRef.current++;
      setMatch(null);
      if (tgt) { setStatus('looking'); setSignal('looking'); vibe('looking'); }
      prevAreaRef.current = 0;
    } else {
      noFindFramesRef.current = 0;
      const g = getGuidance(effectiveMatch, frame, prevAreaRef.current);
      prevAreaRef.current = g.area;
      setMatch({
        name: effectiveMatch.isCandidate ? tgt : effectiveMatch.class,
        score: effectiveMatch.score,
        direction: g.direction,
        distance: g.distance,
        fromClaude: effectiveMatch.fromClaude,
      });
      setStatus(g.status);
      setSignal(g.signal);

      if (!foundOnceRef.current) { foundOnceRef.current = true; vibe('found', true); }
      else vibe(g.signal);
    }

    loopRef.current = requestAnimationFrame(detect);
  }

  function isCocoKnown(tgt) {
    if (!tgt) return false;
    const norm = tgt.trim().toLowerCase();
    if (COCO_KNOWN.has(norm)) return true;
    const aliases = TARGET_ALIASES[norm];
    return aliases ? aliases.some(a => COCO_KNOWN.has(a)) : false;
  }

  function draw(predictions, targetMatch) {
    const canvas = canvasRef.current;
    const video  = videoRef.current;
    if (!canvas || !video) return;

    const W = video.videoWidth || 640, H = video.videoHeight || 480;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    // All non-target predictions dimmed
    predictions.forEach(p => {
      if (p === targetMatch) return;
      const [x, y, w, h] = p.bbox;
      ctx.strokeStyle = 'rgba(255,255,255,0.28)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([]);
      ctx.strokeRect(x, y, w, h);
    });

    if (!targetMatch) return;

    const [x, y, bw, bh] = targetMatch.bbox;
    ctx.setLineDash(targetMatch.fromClaude ? [12, 6] : []);
    ctx.strokeStyle = '#00ff9d';
    ctx.lineWidth   = targetMatch.fromClaude ? 5 : 8;
    ctx.strokeRect(x, y, bw, bh);
    ctx.setLineDash([]);

    const label = `${targetMatch.class} ${Math.round(targetMatch.score * 100)}%${targetMatch.fromClaude ? ' ✦' : ''}`;
    const labelW = Math.min(280, bw);
    ctx.fillStyle = '#00ff9d';
    ctx.fillRect(x, Math.max(0, y - 38), labelW, 38);
    ctx.fillStyle = '#05100d';
    ctx.font = '800 20px system-ui';
    ctx.fillText(label, x + 10, Math.max(26, y - 12));
  }

  function vibe(sig, immediate = false) {
    if (!navigator.vibrate) return;
    const now = Date.now();
    const gap = sig === 'looking' ? 1150 : 520;
    if (!immediate && now - lastHapticRef.current < gap) return;
    if (!immediate && sig === lastSignalRef.current && sig === 'found') return;
    lastHapticRef.current = now; lastSignalRef.current = sig;
    navigator.vibrate(HAPTIC[sig] || HAPTIC.looking);
  }

  // ── Render ───────────────────────────────────────────────────────────────────

  return (
    <main className={`scanner signal-${signal}${mode === 'sos' ? ' sos-mode' : ''}`}>
      <video ref={videoRef} playsInline muted />
      <canvas ref={canvasRef} />

      {!isRunning && mode !== 'sos' && (
        <button className="start-button" type="button" onClick={startScanner}>
          <Camera size={30} />
          Start
        </button>
      )}

      {status === 'loading' && (
        <div className="loading-pill"><Loader2 size={18} />Loading…</div>
      )}

      {claudeLabel && (
        <div className={`claude-pill${claudeLabel === 'found' ? ' claude-found' : ''}`}>
          {claudeLabel === 'scanning'
            ? <><Loader2 size={14} />AI scanning…</>
            : '✦ AI found it'}
        </div>
      )}

      {error && <div className="error-pill">{error}</div>}

      <div className="reticle" aria-hidden="true"><span /></div>

      {/* SOS button — top right, always visible when running */}
      {isRunning && mode !== 'sos' && (
        <button className="sos-btn" onClick={triggerSOS} aria-label="Emergency SOS">
          <AlertTriangle size={16} />SOS
        </button>
      )}
      {mode === 'sos' && (
        <button className="cancel-sos-btn" onClick={cancelSOS}>✕ Cancel SOS</button>
      )}

      {/* Autopilot: next option pill */}
      {mode === 'autopilot' && autoCands.length > 1 && autoIdx < autoCands.length - 1 && (
        <button className="next-cand-btn" onClick={nextAutoCand}>
          Next option →
        </button>
      )}

      {/* Bottom bar */}
      <div className="target-bar">
        <button
          className={`mic-btn${isListening ? ' mic-active' : ''}`}
          type="button"
          onClick={startListening}
          aria-label={isListening ? 'Listening…' : 'Tap to speak'}
        >
          <Mic size={22} />
        </button>

        <div className="target-display" aria-label="Target object">
          {mode === 'autopilot' && autoCands.length > 0
            ? <><span className="auto-label">AUTO</span> {autoCands[autoIdx]}</>
            : target || <span className="target-ph">say what to find…</span>
          }
        </div>

        <button
          type="button"
          className="scan-btn"
          onClick={isRunning ? stopScanner : startScanner}
          aria-label={isRunning ? 'Stop' : 'Scan'}
        >
          {isRunning ? <Square size={18} /> : <ScanLine size={20} />}
        </button>
      </div>

      {/* Status strip — top left */}
      <div className="signal-strip" aria-live="polite">
        <strong>{mode === 'sos' ? '🆘 SOS' : status}</strong>
        <span>
          {match
            ? `${match.direction} · ${match.distance}${match.fromClaude ? ' · ✦AI' : ''}`
            : claudeLabel === 'scanning' ? 'AI scanning…'
            : hapticsOk ? 'scan slowly' : 'haptics unavailable'}
        </span>
      </div>
    </main>
  );
}

// ── Camera helpers ─────────────────────────────────────────────────────────────

async function getWideCameraStream() {
  const base = {
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 }, aspectRatio: { ideal: 16 / 9 } },
    audio: false,
  };
  const first = await navigator.mediaDevices.getUserMedia(base);
  const devices = await navigator.mediaDevices.enumerateDevices();
  const wide = devices
    .filter(d => d.kind === 'videoinput')
    .find(d => /ultra|wide|back|rear|environment/i.test(d.label));
  if (!wide) return first;
  first.getTracks().forEach(t => t.stop());
  return navigator.mediaDevices.getUserMedia({
    video: { deviceId: { exact: wide.deviceId }, width: { ideal: 1920 }, height: { ideal: 1080 } },
    audio: false,
  });
}

async function setWidestZoom(stream) {
  const track = stream.getVideoTracks()[0];
  const caps  = track?.getCapabilities?.();
  if (!caps?.zoom) return;
  try { await track.applyConstraints({ advanced: [{ zoom: caps.zoom.min }] }); } catch { /* ignore */ }
}

// ── Detection helpers ──────────────────────────────────────────────────────────

function findTarget(predictions, target, frame) {
  const aliases = getAliases(target);
  const exact   = predictions
    .filter(p => aliases.includes(p.class.toLowerCase()))
    .sort((a, b) => b.score - a.score)[0];
  if (exact) return exact;
  return findCenterCandidate(predictions, frame);
}

function getAliases(target) {
  const norm = target.trim().toLowerCase();
  return TARGET_ALIASES[norm] || [norm];
}

function getGuidance(prediction, frame, prevArea) {
  const [x,, w, h] = prediction.bbox;
  const cx   = (x + w / 2) / frame.width;
  const area = (w * h) / (frame.width * frame.height);
  const dist = estimateDistance(area);
  const centered    = cx > 0.42 && cx < 0.58;
  const close       = area > 0.2;
  const gettingNear = prevArea > 0 && area > prevArea * 1.08;

  if (centered && close) return { signal: 'reach',  status: 'reach',  direction: 'center',     distance: dist, area };
  if (centered)          return { signal: gettingNear ? 'closer' : 'locked', status: gettingNear ? 'closer' : 'locked', direction: 'center', distance: dist, area };
  if (cx < 0.42)         return { signal: 'left',   status: 'left',   direction: 'turn left',  distance: dist, area };
  return                        { signal: 'right',  status: 'right',  direction: 'turn right', distance: dist, area };
}

function findCenterCandidate(predictions, frame) {
  const cx = frame.width / 2, cy = frame.height / 2;
  return predictions
    .map(p => {
      const [x, y, w, h] = p.bbox;
      const pcx = x + w / 2, pcy = y + h / 2;
      const nd  = Math.hypot((pcx - cx) / frame.width, (pcy - cy) / frame.height);
      const area = (w * h) / (frame.width * frame.height);
      return { ...p, score: p.score * 0.72, centerScore: p.score + area * 1.6 - nd * 1.4, isCandidate: true };
    })
    .filter(p => p.centerScore > 0.12 && p.score > 0.14)
    .sort((a, b) => b.centerScore - a.centerScore)[0];
}

function estimateDistance(area) {
  if (area > 0.24) return 'very close';
  if (area > 0.14) return 'close';
  if (area > 0.07) return 'medium';
  return 'far';
}

createRoot(document.getElementById('root')).render(<App />);
