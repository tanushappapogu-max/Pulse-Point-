import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import * as tf from '@tensorflow/tfjs';
import { pipeline } from '@xenova/transformers';
import { Camera, Loader2, ScanLine, Square, Mic, AlertTriangle } from 'lucide-react';
import './styles.css';

const YOLO_INPUT = 640;
const LIGHT_FPS = 10;
const HEAVY_FPS = 2; // TODO: set this to your desired fallback fps (X)
const NO_FIND_SECONDS = 3;
const NO_FIND_FRAMES = Math.round(LIGHT_FPS * NO_FIND_SECONDS);
const GEMINI_MODEL = 'gemini-2.5-flash-lite';
const EMBED_MODEL = 'Xenova/all-MiniLM-L6-v2';
const VECTOR_SIM_THRESHOLD = 0.78;

// ── COCO 80 class labels in official order ───────────────────────────────────
const COCO_LABELS = [
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
];
const COCO_KNOWN = new Set(COCO_LABELS);

const TARGET_ALIASES = {
  phone: ['cell phone'], iphone: ['cell phone'], android: ['cell phone'],
  'my phone': ['cell phone'], 'cell phone': ['cell phone'],
  'computer mouse': ['mouse'], trackpad: ['mouse'], 'my mouse': ['mouse'],
  tv: ['tv'], television: ['tv'], monitor: ['tv'],
  sofa: ['couch'], couch: ['couch'],
  laptop: ['laptop'], computer: ['laptop'],
  remote: ['remote'], 'tv remote': ['remote'],
  ship: ['boat'],
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

let embedderPromise;
const cocoEmbeddingCache = new Map();
const targetEmbeddingCache = new Map();

async function getEmbedder() {
  if (!embedderPromise) {
    embedderPromise = pipeline('feature-extraction', EMBED_MODEL, { quantized: true });
  }
  return embedderPromise;
}

async function embedText(text, cache) {
  const norm = text.trim().toLowerCase();
  if (!norm) return null;
  if (cache.has(norm)) return cache.get(norm);
  const extractor = await getEmbedder();
  const output = await extractor(norm, { pooling: 'mean', normalize: true });
  const vec = Array.from(output.data || []);
  if (vec.length) cache.set(norm, vec);
  return vec.length ? vec : null;
}

function cosineSimilarity(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

async function findClosestCocoLabel(text) {
  try {
    const targetVec = await embedText(text, targetEmbeddingCache);
    if (!targetVec) return null;

    let best = { label: null, score: 0 };
    for (const label of COCO_LABELS) {
      const labelVec = await embedText(label, cocoEmbeddingCache);
      if (!labelVec) continue;
      const sim = cosineSimilarity(targetVec, labelVec);
      if (sim > best.score) best = { label, score: sim };
    }

    return best.score >= VECTOR_SIM_THRESHOLD ? best : null;
  } catch {
    return null;
  }
}

async function loadYoloModel() {
  return tf.loadGraphModel('/yolo11n_web_model/model.json');
}

async function runYolo(video, model) {
  const tfImg = tf.browser.fromPixels(video);
  const resized = tf.image.resizeBilinear(tfImg, [YOLO_INPUT, YOLO_INPUT]);
  const normalized = resized.div(255);
  const batched = normalized.expandDims(0);

  const output = await model.executeAsync(batched);
  const outputTensor = Array.isArray(output)
    ? output[0]
    : (output?.output0 || output);
  const data = await outputTensor.data();
  const shape = outputTensor.shape;

  tfImg.dispose();
  resized.dispose();
  normalized.dispose();
  batched.dispose();
  if (Array.isArray(output)) output.forEach(t => t.dispose());
  else if (output && typeof output === 'object' && output !== outputTensor) {
    Object.values(output).forEach(t => t.dispose?.());
  } else {
    outputTensor.dispose();
  }

  const frame = { width: video.videoWidth || 640, height: video.videoHeight || 480 };
  return decodeYoloOutput(data, shape, frame);
}

function decodeYoloOutput(output, shape, frame) {
  if (!shape || shape.length < 3) return [];
  const [_, dim1, dim2] = shape;
  const features = 5 + COCO_LABELS.length;
  const transposed = dim1 === features;
  const numBoxes = transposed ? dim2 : dim1;
  if (transposed && dim2 <= 0) return [];
  if (!transposed && dim2 !== features) return [];

  const getVal = (i, f) => (transposed ? output[f * numBoxes + i] : output[i * features + f]);

  const raw = [];
  let maxCoord = 0;
  for (let i = 0; i < numBoxes; i++) {
    const cx = getVal(i, 0);
    const cy = getVal(i, 1);
    const w = getVal(i, 2);
    const h = getVal(i, 3);
    const obj = getVal(i, 4);
    maxCoord = Math.max(maxCoord, cx, cy, w, h);

    let bestScore = 0;
    let bestClass = 0;
    for (let c = 0; c < COCO_LABELS.length; c++) {
      const cls = getVal(i, 5 + c);
      const score = obj * cls;
      if (score > bestScore) {
        bestScore = score;
        bestClass = c;
      }
    }

    if (bestScore >= 0.25) {
      raw.push({ cx, cy, w, h, score: bestScore, class: COCO_LABELS[bestClass] });
    }
  }

  const usePixelCoords = maxCoord > 2;
  const sx = usePixelCoords ? frame.width / YOLO_INPUT : frame.width;
  const sy = usePixelCoords ? frame.height / YOLO_INPUT : frame.height;

  const predictions = raw.map(r => {
    const x = (r.cx - r.w / 2) * sx;
    const y = (r.cy - r.h / 2) * sy;
    const bw = r.w * sx;
    const bh = r.h * sy;
    return { class: r.class, score: r.score, bbox: [x, y, bw, bh] };
  });

  return nonMaxSuppression(predictions, 0.45, 60);
}

function nonMaxSuppression(predictions, iouThreshold, maxDetections) {
  const sorted = [...predictions].sort((a, b) => b.score - a.score);
  const selected = [];

  for (const pred of sorted) {
    if (selected.length >= maxDetections) break;
    let keep = true;
    for (const picked of selected) {
      if (iou(pred.bbox, picked.bbox) > iouThreshold) { keep = false; break; }
    }
    if (keep) selected.push(pred);
  }

  return selected;
}

function iou(a, b) {
  const [ax, ay, aw, ah] = a;
  const [bx, by, bw, bh] = b;
  const x1 = Math.max(ax, bx);
  const y1 = Math.max(ay, by);
  const x2 = Math.min(ax + aw, bx + bw);
  const y2 = Math.min(ay + ah, by + bh);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = aw * ah + bw * bh - inter;
  return union <= 0 ? 0 : inter / union;
}

// ── Gemini 2.5 Flash Lite API ────────────────────────────────────────────────
// API key must be in .env as VITE_GEMINI_API_KEY

async function callGeminiJson(imageBase64, prompt) {
  const key = import.meta.env.VITE_GEMINI_API_KEY;
  if (!key || !imageBase64) return null;

  const base64Data = imageBase64.replace(/^data:image\/[a-z]+;base64,/, '');

  try {
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${key}`,
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          contents: [{
            parts: [
              {
                inline_data: {
                  mime_type: 'image/jpeg',
                  data: base64Data,
                },
              },
              { text: prompt },
            ],
          }],
          generationConfig: {
            temperature: 0,
            maxOutputTokens: 128,
          },
        }),
      }
    );

    const data = await res.json();
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text || '';

    const s = text.indexOf('{'), e = text.lastIndexOf('}') + 1;
    if (s < 0 || e <= s) return null;
    return JSON.parse(text.slice(s, e));
  } catch {
    return null;
  }
}

async function callGeminiBox(imageBase64, target) {
  const prompt = `
Find the object "${target}" in this camera frame.

Return ONLY valid JSON:
{
  "found": true,
  "label": "${target}",
  "box_2d": [ymin, xmin, ymax, xmax],
  "confidence": 0.0
}

Requirements:
- box_2d must be [ymin, xmin, ymax, xmax]
- coordinates must be integers from 0 to 1000 (normalized to image dimensions)
- return only one box, for the clearest and most reachable instance
- if not visible, return:
{"found":false,"label":"${target}","box_2d":null,"confidence":0.0}
- no markdown, no explanation, JSON only
`;

  const parsed = await callGeminiJson(imageBase64, prompt);
  if (!parsed?.found || !parsed.box_2d) return null;

  const [yMin, xMin, yMax, xMax] = parsed.box_2d;
  return {
    found: true,
    x: xMin / 1000,
    y: yMin / 1000,
    w: (xMax - xMin) / 1000,
    h: (yMax - yMin) / 1000,
    confidence: parsed.confidence || 0.8,
  };
}
// ── App ────────────────────────────────────────────────────────────────────────

function App() {
  const videoRef          = useRef(null);
  const canvasRef         = useRef(null);
  const modelRef          = useRef(null);
  const streamRef         = useRef(null);
  const loopRef           = useRef(null);
  const lastLightRunRef   = useRef(0);
  const lastHeavyRunRef   = useRef(0);
  const lastPredsRef      = useRef([]);
  const lastHapticRef     = useRef(0);
  const lastSignalRef     = useRef('');
  const prevAreaRef       = useRef(0);
  const foundOnceRef      = useRef(false);
  const noFindFramesRef   = useRef(0);
  const aiBoxRef          = useRef(null); // last AI-returned bbox {x,y,w,h,confidence}
  const aiInFlightRef     = useRef(false);
  const localTargetRef    = useRef(null); // { label, score, source }
  const localPendingRef   = useRef(false);
  const targetRef         = useRef('');  // always-current target for the detect loop

  const [target,        setTargetState]   = useState('');
  const [status,        setStatus]        = useState('ready');
  const [signal,        setSignal]        = useState('looking');
  const [match,         setMatch]         = useState(null);
  const [error,         setError]         = useState('');
  const [isRunning,     setIsRunning]     = useState(false);
  const [hapticsOk,     setHapticsOk]     = useState(true);
  const [isListening,   setIsListening]   = useState(false);
  const [aiLabel,       setAiLabel]       = useState(''); // 'scanning' | 'found' | ''
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
    lastLightRunRef.current = 0;
    lastHeavyRunRef.current = 0;
    lastPredsRef.current = [];
    aiBoxRef.current  = null;
    aiInFlightRef.current = false;
    localTargetRef.current = null;
    localPendingRef.current = false;
    setAiLabel('');
    void resolveLocalTarget(t);
  }

  async function resolveLocalTarget(tgt) {
    if (!tgt) return;
    const direct = resolveCocoTarget(tgt);
    if (direct) {
      localTargetRef.current = { label: direct, score: 1, source: 'coco' };
      return;
    }
    if (localPendingRef.current) return;
    localPendingRef.current = true;
    try {
      const closest = await findClosestCocoLabel(tgt);
      if (closest?.label) {
        localTargetRef.current = { label: closest.label, score: closest.score, source: 'vector' };
      }
    } finally {
      localPendingRef.current = false;
    }
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
    setAiLabel('scanning');
    if (!streamRef.current) await startScanner();

    const frame = captureJpeg(videoRef.current);
    const result = await callGeminiJson(frame,
      `User said: "${intent}"
Look at this image. Identify 2–4 real visible objects that best match the user's intent, ranked by how obvious/accessible they are.
CRITICAL: Only list objects that are clearly visible as distinct items — NOT walls, floors, tables, or surfaces themselves. Only things ON surfaces or in the scene as separate objects.
Return JSON: {"candidates":["item1","item2"],"positions":["short position like 'center table'","short position"]}
If nothing matches, return {"candidates":[],"positions":[]}`
    );

    setAiLabel('');
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
    setAiLabel('scanning');
    if (!streamRef.current) await startScanner();

    const frame = captureJpeg(videoRef.current);
    const result = await callGeminiJson(frame,
      `EMERGENCY: Find the fastest path to safety — a door, hallway, open space, or exit.
Ignore everything except exits and large open paths.
Return JSON: {
  "direction": "left or right or forward or backward",
  "description": "one short spoken instruction max 10 words",
  "obstacle": "obstacle name or null"
}
If no exit visible: {"direction":"forward","description":"Follow the wall to your right","obstacle":null}`
    );

    setAiLabel('');
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
    lastLightRunRef.current = 0;
    lastHeavyRunRef.current = 0;
    lastPredsRef.current    = [];
    aiBoxRef.current        = null;
    aiInFlightRef.current   = false;

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
        modelRef.current = await loadYoloModel();
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
    aiBoxRef.current = null;
  }

  async function detect() {
    const video = videoRef.current;
    const model = modelRef.current;

    if (!video || !model || video.readyState < 2) {
      loopRef.current = requestAnimationFrame(detect); return;
    }

    const tgt = targetRef.current;
    if (tgt && !localTargetRef.current && !localPendingRef.current) {
      void resolveLocalTarget(tgt);
    }

    const frame = { width: video.videoWidth || 640, height: video.videoHeight || 480 };
    const now = performance.now();

    const lightInterval = 1000 / LIGHT_FPS;
    const heavyInterval = 1000 / HEAVY_FPS;
    const ranLight = now - lastLightRunRef.current >= lightInterval;

    let predictions = lastPredsRef.current;
    if (ranLight) {
      lastLightRunRef.current = now;
      predictions = await runYolo(video, model);
      lastPredsRef.current = predictions;
    }

    const directLabel = resolveCocoTarget(tgt);
    const localInfo = localTargetRef.current;
    const mappedLabel = localInfo?.label || directLabel;
    const cocoMatchRaw = mappedLabel ? findTarget(predictions, mappedLabel, frame, true) : null;
    const cocoMatch = cocoMatchRaw ? {
      ...cocoMatchRaw,
      displayClass: mappedLabel !== tgt ? tgt : cocoMatchRaw.class,
      fromVector: localInfo?.source === 'vector',
      source: cocoMatchRaw,
    } : null;

    const localCanFind = Boolean(mappedLabel);
    const waitingForVector = !localCanFind && localPendingRef.current;
    const needsAi = tgt && !cocoMatch && ((!localCanFind && !waitingForVector) || noFindFramesRef.current > NO_FIND_FRAMES);
    const shouldRunHeavy = needsAi && (now - lastHeavyRunRef.current >= heavyInterval);

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
          } else {
            aiBoxRef.current = null;
            setAiLabel('');
          }
        })
        .catch(() => setAiLabel(''))
        .finally(() => { aiInFlightRef.current = false; });
    }

    let effectiveMatch = cocoMatch;
    if (!cocoMatch && aiBoxRef.current?.found) {
      const cb = aiBoxRef.current;
      effectiveMatch = {
        class: tgt,
        displayClass: tgt,
        score: cb.confidence || 0.8,
        bbox: [cb.x * frame.width, cb.y * frame.height, cb.w * frame.width, cb.h * frame.height],
        fromAi: true,
      };
    }

    draw(predictions, effectiveMatch);

    if (!effectiveMatch) {
      if (ranLight) noFindFramesRef.current++;
      setMatch(null);
      if (tgt) { setStatus('looking'); setSignal('looking'); vibe('looking'); }
      prevAreaRef.current = 0;
    } else {
      if (ranLight) noFindFramesRef.current = 0;
      const g = getGuidance(effectiveMatch, frame, prevAreaRef.current);
      prevAreaRef.current = g.area;
      setMatch({
        name: effectiveMatch.isCandidate ? tgt : (effectiveMatch.displayClass || effectiveMatch.class),
        score: effectiveMatch.score,
        direction: g.direction,
        distance: g.distance,
        fromAi: effectiveMatch.fromAi,
      });
      setStatus(g.status);
      setSignal(g.signal);

      if (!foundOnceRef.current) { foundOnceRef.current = true; vibe('found', true); }
      else vibe(g.signal);
    }

    loopRef.current = requestAnimationFrame(detect);
  }

  function resolveCocoTarget(tgt) {
    if (!tgt) return null;
    const norm = tgt.trim().toLowerCase();
    if (COCO_KNOWN.has(norm)) return norm;
    const aliases = TARGET_ALIASES[norm];
    if (!aliases) return null;
    return aliases.find(a => COCO_KNOWN.has(a)) || null;
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
      if (targetMatch?.source === p) return;
      const [x, y, w, h] = p.bbox;
      ctx.strokeStyle = 'rgba(255,255,255,0.28)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([]);
      ctx.strokeRect(x, y, w, h);
    });

    if (!targetMatch) return;

    const [x, y, bw, bh] = targetMatch.bbox;
    ctx.setLineDash(targetMatch.fromAi ? [12, 6] : []);
    ctx.strokeStyle = '#00ff9d';
    ctx.lineWidth   = targetMatch.fromAi ? 5 : 8;
    ctx.strokeRect(x, y, bw, bh);
    ctx.setLineDash([]);

    const display = targetMatch.displayClass || targetMatch.class;
    const label = `${display} ${Math.round(targetMatch.score * 100)}%${targetMatch.fromAi ? ' ✦' : ''}`;
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

      {aiLabel && (
        <div className={`claude-pill${aiLabel === 'found' ? ' claude-found' : ''}`}>
          {aiLabel === 'scanning'
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
            ? `${match.direction} · ${match.distance}${match.fromAi ? ' · ✦AI' : ''}`
            : aiLabel === 'scanning' ? 'AI scanning…'
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

function captureJpeg(video) {
  if (!video) return null;
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.82);
}

// ── Detection helpers ──────────────────────────────────────────────────────────

function findTarget(predictions, target, frame, allowCandidate = true) {
  const aliases = getAliases(target);
  const exact   = predictions
    .filter(p => aliases.includes(p.class.toLowerCase()))
    .sort((a, b) => b.score - a.score)[0];
  if (exact) return exact;
  return allowCandidate ? findCenterCandidate(predictions, frame) : null;
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
