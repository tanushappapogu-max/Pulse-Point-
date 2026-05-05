import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import * as tf from '@tensorflow/tfjs';
import { Camera, Loader2, ScanLine, Square, Mic } from 'lucide-react';
import './styles.css';

const YOLO_INPUT = 640;
const LIGHT_FPS = 15;
const HEAVY_COOLDOWN_MS = 2500; // ai cooldown, keep it chill
const BOX_SMOOTH_ALPHA = 0.55; // 0.55 of new box, 0.45 of prev — kills jitter without lag
const BOX_SMOOTH_SNAP_IOU = 0.3; // if new box barely overlaps the smoothed one, snap instead of blend
const NO_FIND_SECONDS = 3;
const NO_FIND_FRAMES = Math.round(LIGHT_FPS * NO_FIND_SECONDS);
const GEMINI_MODEL = 'google/gemini-2.0-flash-lite-preview-02-05:free';
const FUZZY_SIM_THRESHOLD = 0.42;

// COCO labels, big list incoming
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

// L4 fix: removed dead 'sos' pattern since SOS was ripped out
// these are the only haptic signals we actually use
const HAPTIC = {
  looking: [24, 260],
  found:   [170, 80, 170, 80, 260],
  left:    [80, 45, 80],
  right:   [80, 45, 80, 45, 80],
  closer:  [45, 38, 45, 38, 45],
  locked:  [260],
  reach:   [360, 80, 360],
  lost:    [35, 120, 35],
  confirm: [200, 80, 200],
};

function tokenize(text) {
  return normalizeTargetText(text)
    .split(' ')
    .filter(Boolean);
}

function bigrams(text) {
  const compact = normalizeTargetText(text).replace(/\s+/g, '');
  const grams = [];
  for (let i = 0; i < compact.length - 1; i++) grams.push(compact.slice(i, i + 2));
  return grams;
}

function jaccard(a, b) {
  if (!a.length || !b.length) return 0;
  const setA = new Set(a);
  const setB = new Set(b);
  let intersection = 0;
  setA.forEach(item => { if (setB.has(item)) intersection++; });
  return intersection / (setA.size + setB.size - intersection || 1);
}

function stringSimilarity(a, b) {
  const normA = normalizeTargetText(a);
  const normB = normalizeTargetText(b);
  if (!normA || !normB) return 0;
  if (normA === normB) return 1;

  const tokensA = tokenize(normA);
  const tokensB = tokenize(normB);
  const tokenScore = jaccard(tokensA, tokensB);
  const gramScore = jaccard(bigrams(normA), bigrams(normB));
  const prefixScore = normA.startsWith(normB) || normB.startsWith(normA) ? 0.85 : 0;
  return Math.min(1, Math.max(tokenScore * 0.9 + gramScore * 0.4, prefixScore));
}

function normalizeTargetText(text) {
  return text
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function findClosestCocoLabel(text) {
  const norm = normalizeTargetText(text);
  if (!norm) return null;

  let best = { label: null, score: 0 };
  for (const label of COCO_LABELS) {
    const score = stringSimilarity(norm, label);
    if (score > best.score) best = { label, score };
  }

  return best.score >= FUZZY_SIM_THRESHOLD ? best : null;
}

async function loadYoloModel() {
  return tf.loadGraphModel('/yolo11n_web_model/model.json');
}

async function runYolo(video, model) {
  const tensors = [];
  let output = null;
  let outputTensor = null;
  try {
    const tfImg = tf.browser.fromPixels(video);
    tensors.push(tfImg);
    const resized = tf.image.resizeBilinear(tfImg, [YOLO_INPUT, YOLO_INPUT]);
    tensors.push(resized);
    const normalized = resized.div(255);
    tensors.push(normalized);
    const batched = normalized.expandDims(0);
    tensors.push(batched);

    output = await model.executeAsync(batched);
    outputTensor = Array.isArray(output)
      ? output[0]
      : (output?.output0 || output);
    const data = await outputTensor.data();
    const shape = outputTensor.shape;

    const frame = { width: video.videoWidth || 640, height: video.videoHeight || 480 };
    return decodeYoloOutput(data, shape, frame);
  } finally {
    tensors.forEach(t => t.dispose());
    if (Array.isArray(output)) output.forEach(t => t.dispose());
    else if (output && typeof output === 'object' && output !== outputTensor) {
      Object.values(output).forEach(t => t.dispose?.());
    } else {
      outputTensor?.dispose();
    }
  }
}

function decodeYoloOutput(output, shape, frame) {
  if (!shape || shape.length < 3) return [];
  const [_, dim1, dim2] = shape;
  const featuresNoObj = 4 + COCO_LABELS.length;
  const featuresWithObj = 5 + COCO_LABELS.length;
  const transposed = dim1 === featuresNoObj || dim1 === featuresWithObj;
  const features = transposed ? dim1 : dim2;
  const numBoxes = transposed ? dim2 : dim1;
  if (transposed && dim2 <= 0) return [];
  if (![featuresNoObj, featuresWithObj].includes(features)) return [];

  const hasObj = features === featuresWithObj;
  const classOffset = hasObj ? 5 : 4;

  const getVal = (i, f) => (transposed ? output[f * numBoxes + i] : output[i * features + f]);

  const raw = [];
  let maxCoord = 0;
  for (let i = 0; i < numBoxes; i++) {
    const cx = getVal(i, 0);
    const cy = getVal(i, 1);
    const w = getVal(i, 2);
    const h = getVal(i, 3);
    const obj = hasObj ? getVal(i, 4) : 1;
    maxCoord = Math.max(maxCoord, cx, cy, w, h);

    let bestScore = 0;
    let bestClass = 0;
    for (let c = 0; c < COCO_LABELS.length; c++) {
      const cls = getVal(i, classOffset + c);
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

// openrouter call, still using VITE_GEMINI_API_KEY

async function callGeminiJson(imageBase64, prompt) {
  const key = import.meta.env.VITE_GEMINI_API_KEY;
  if (!key || !imageBase64) return null;

  try {
    const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        authorization: `Bearer ${key}`,
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        model: GEMINI_MODEL,
        temperature: 0,
        max_tokens: 128,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: prompt },
              { type: 'image_url', image_url: { url: imageBase64 } },
            ],
          },
        ],
      }),
    });

    const data = await res.json();

    // B3 fix: openrouter sends back data.error on rate limit, bad key, etc.
    // used to just silently return null and the user would see 'AI scanning...' forever
    if (data.error) {
      console.warn('OpenRouter error:', data.error?.message || data.error);
      return { __error: data.error?.message || 'AI unavailable' };
    }

    let text = data.choices?.[0]?.message?.content || '';
    if (Array.isArray(text)) {
      text = text.map(part => part?.text || '').join('');
    }

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
// ok here is the app core
function App() {
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
  const lastHapticRef     = useRef(0);
  const lastSignalRef     = useRef('');
  const prevAreaRef       = useRef(0);
  const foundOnceRef      = useRef(false);
  const noFindFramesRef   = useRef(0);
  const aiBoxRef          = useRef(null); // last ai bbox, rough coords + confidence
  const aiInFlightRef     = useRef(false);
  const localTargetRef    = useRef(null); // {label, score, source} map-ish
  const targetRef         = useRef('');  // always-current target for the detect loop
  const smoothedBoxRef    = useRef(null); // EMA-smoothed bbox so guidance arrows don't twitch

  // B2 fix: isRunning as a ref too so closures don't go stale
  // React state is great for rendering but closures capture old values — ref always current
  const isRunningRef      = useRef(false);

  // M1: detect voice support once at init, don't check it every render
  const canVoice = Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);

  const [target,        setTargetState]   = useState('');
  const [draftTarget,   setDraftTarget]   = useState('');
  const [status,        setStatus]        = useState('ready');
  const [signal,        setSignal]        = useState('looking');
  const [match,         setMatch]         = useState(null);
  const [error,         setError]         = useState('');
  const [isRunning,     setIsRunning]     = useState(false);
  const [hapticsOk,     setHapticsOk]     = useState(true);
  const [isListening,   setIsListening]   = useState(false);
  const [aiLabel,       setAiLabel]       = useState(''); // 'scanning' | 'found' | ''
  const [mode,          setMode]          = useState('normal'); // 'normal' | 'autopilot'
  const [autoCands,     setAutoCands]     = useState([]);
  const [autoIdx,       setAutoIdx]       = useState(0);

  useEffect(() => {
    setHapticsOk('vibrate' in navigator);
    return () => stopScanner();
  }, []);

  // keep targetRef synced with the loop, no drama
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
    smoothedBoxRef.current = null;
    setAiLabel('');
    resolveLocalTarget(t);
  }

  function submitTypedTarget() {
    const text = draftTarget.trim();
    if (!text) return;
    setError('');
    setTarget(text);
    setMode('normal');
    // B2 fix: use the ref, not the state — closures capture old state values
    if (!isRunningRef.current) startScanner(); else setStatus('looking');
  }

  function handleScanClick() {
    // B2 fix: same deal, ref not state
    if (isRunningRef.current) { stopScanner(); return; }
    if (draftTarget.trim()) { submitTypedTarget(); return; }
    startScanner();
  }

  function handleTypedKeyDown(e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      submitTypedTarget();
    }
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

  // voice input stuff, kinda picky

  function startListening() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      const isiOS = /iPad|iPhone|iPod/i.test(navigator.userAgent || '');
      setError(isiOS
        ? 'Voice input is not supported on iPhone Safari. Type below or use dictation.'
        : 'Voice input is not supported in this browser. Type below.'
      );
      return;
    }
    setError('');
    const r = new SR();
    r.continuous = false; r.interimResults = false; r.lang = 'en-US';
    setIsListening(true);
    r.onresult = e => { setIsListening(false); handleVoice(e.results[0][0].transcript.trim()); };
    r.onerror  = e  => {
      setIsListening(false);
      const err = e?.error;
      if (err === 'not-allowed' || err === 'service-not-allowed') {
        setError('Mic blocked — allow microphone access in browser settings.');
      } else if (err === 'no-speech') {
        setError('No speech heard — tap to retry.');
      } else if (err === 'network') {
        setError('Voice service unavailable — check connection or type below.');
      } else {
        setError('Voice input failed — type below.');
      }
    };
    r.onend    = ()  => setIsListening(false);
    navigator.vibrate?.([80]);
    r.start();
  }

  async function handleVoice(text) {
    const lower = text.toLowerCase();

    // vague intent -> autopilot
    const isVague =
      /\b(something|anything)\b/i.test(lower) ||
      /\b(eat|food|hungry|snack|drink|thirsty)\b/i.test(lower) ||
      /\b(write|writing|draw)\b/i.test(lower) ||
      /\b(call|text)\b/i.test(lower);

    if (isVague) {
      runAutopilot(text); return;
    }

    // direct target, strip the fluff
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
      // B2 fix: ref is always current, state might be stale in this callback
      if (!isRunningRef.current) startScanner(); else setStatus('looking');
    }
  }

  // autopilot mode, let the model pick
  async function runAutopilot(intent) {
    setMode('autopilot');
    setStatus('thinking…');
    setAiLabel('scanning');
    pauseDetectRef.current = true;
    if (!streamRef.current) await startScanner();
    try {
      const frame = captureJpeg(videoRef.current);
      const result = await callGeminiJson(frame,
        `User said: "${intent}"
Look at this image. Identify 2–4 real visible objects that best match the user's intent, ranked by how obvious/accessible they are.
CRITICAL: Only list objects that are clearly visible as distinct items — NOT walls, floors, tables, or surfaces themselves. Only things ON surfaces or in the scene as separate objects.
Return JSON: {"candidates":["item1","item2"],"positions":["short position like 'center table'","short position"]}
If nothing matches, return {"candidates":[],"positions":[]}`
      );

      // B3 fix: check if the result itself is an error signal from openrouter
      if (result?.__error) {
        setMode('normal');
        setError(`AI error: ${result.__error}`);
        setStatus('ready');
      } else if (result?.candidates?.length > 0) {
        setAutoCands(result.candidates);
        setAutoIdx(0);
        setTarget(result.candidates[0]);
        setStatus('looking');
        vibe('confirm', true);
        // B2 fix: ref not state
        if (!isRunningRef.current) startScanner();
      } else {
        setMode('normal');
        setError('Nothing found for that. Try being more specific.');
        setStatus('ready');
      }
    } finally {
      pauseDetectRef.current = false;
      setAiLabel('');
      // B1 fix: restart the detect loop now that autopilot pause is lifted
      // without this the loop just... dies quietly
      if (streamRef.current && !loopRef.current) detect();
    }
  }

  function nextAutoCand() {
    const next = autoIdx + 1;
    if (next >= autoCands.length) { setAutoCands([]); setMode('normal'); return; }
    setAutoIdx(next);
    setTarget(autoCands[next]);
    vibe('confirm', true);
  }

  // camera + detect loop
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
    smoothedBoxRef.current  = null;

    try {
      const stream = await getWideCameraStream();
      streamRef.current        = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      await setWidestZoom(stream);

      isRunningRef.current = true;
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
    } finally {
      startInFlightRef.current = false;
    }
  }

  function stopScanner() {
    if (loopRef.current) cancelAnimationFrame(loopRef.current);
    loopRef.current = null;
    navigator.vibrate?.(0);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    pauseDetectRef.current = false;
    detectingRef.current = false;
    isRunningRef.current = false;
    setIsRunning(false);
    setStatus('ready');
    aiBoxRef.current = null;
    smoothedBoxRef.current = null;
  }

  async function detect() {
    if (detectingRef.current) {
      if (streamRef.current) loopRef.current = requestAnimationFrame(detect);
      return;
    }

    detectingRef.current = true;
    try {
      const video = videoRef.current;
      const model = modelRef.current;

      if (!video || !model || video.readyState < 2) return;
      if (pauseDetectRef.current) return;

      const tgt = targetRef.current;
      if (tgt && !localTargetRef.current) {
        void resolveLocalTarget(tgt);
      }

      const frame = { width: video.videoWidth || 640, height: video.videoHeight || 480 };
      const now = performance.now();

      const lightInterval = 1000 / LIGHT_FPS;
      const heavyInterval = HEAVY_COOLDOWN_MS;
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
        source: cocoMatchRaw,
      } : null;

      const localCanFind = Boolean(mappedLabel);
      const needsAi = tgt && !cocoMatch && (!localCanFind || noFindFramesRef.current > NO_FIND_FRAMES);
      const shouldRunHeavy = needsAi && (now - lastHeavyRunRef.current >= heavyInterval);

      if (shouldRunHeavy && !aiInFlightRef.current) {
        lastHeavyRunRef.current = now;
        aiInFlightRef.current = true;
        setAiLabel('scanning');
        const img = captureJpeg(video);
        callGeminiBox(img, tgt)
          .then(r => {
            // B3 fix: callGeminiBox returns null on failure, but callGeminiJson
            // now returns {__error} on openrouter API errors
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

      // EMA smoothing on the tracked box — kills frame-to-frame jitter so guidance is steady.
      // Snap (no blend) when the new detection barely overlaps the smoothed one — that means
      // the target moved a lot, the user re-aimed, or detection re-acquired after a miss.
      if (effectiveMatch) {
        const raw = effectiveMatch.bbox;
        const prev = smoothedBoxRef.current;
        if (!prev) {
          smoothedBoxRef.current = [...raw];
        } else {
          const overlap = iou(prev, raw);
          const a = overlap < BOX_SMOOTH_SNAP_IOU ? 1 : BOX_SMOOTH_ALPHA;
          smoothedBoxRef.current = [
            a * raw[0] + (1 - a) * prev[0],
            a * raw[1] + (1 - a) * prev[1],
            a * raw[2] + (1 - a) * prev[2],
            a * raw[3] + (1 - a) * prev[3],
          ];
        }
        effectiveMatch = { ...effectiveMatch, bbox: [...smoothedBoxRef.current] };
      } else {
        smoothedBoxRef.current = null;
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
    } finally {
      detectingRef.current = false;
      // B1 fix: don't reschedule while autopilot has paused us
      // before this we were burning 60fps of RAF calls that all just early-returned immediately
      if (streamRef.current && !pauseDetectRef.current) {
        loopRef.current = requestAnimationFrame(detect);
      }
    }
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

    // non-target preds, a little faded
    predictions.forEach(p => {
      if (targetMatch?.source === p) return;
      const [x, y, w, h] = mapBox(p.bbox);
      ctx.strokeStyle = 'rgba(255,255,255,0.28)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([]);
      ctx.strokeRect(x, y, w, h);
    });

    if (!targetMatch) return;

    const [x, y, bw, bh] = mapBox(targetMatch.bbox);
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

  return (
    <main className={`scanner signal-${signal}`}>
      <video ref={videoRef} playsInline muted />
      <canvas ref={canvasRef} />

      {!isRunning && (
        <button className="start-button" type="button" onClick={startScanner}>
          <Camera size={30} />
          Start
        </button>
      )}

      {status === 'loading' && (
        <div className="loading-pill"><Loader2 size={18} />Loading…</div>
      )}

      {aiLabel && (
        <div className={`ai-pill${aiLabel === 'found' ? ' ai-found' : ''}`}>
          {aiLabel === 'scanning'
            ? <><Loader2 size={14} />AI scanning…</>
            : '✦ AI found it'}
        </div>
      )}

      {error && <div className="error-pill">{error}</div>}

      <div className="reticle" aria-hidden="true"><span /></div>

      {/* autopilot next option, in case we have more */}
      {mode === 'autopilot' && autoCands.length > 1 && autoIdx < autoCands.length - 1 && (
        <button className="next-cand-btn" onClick={nextAutoCand}>
          Next option →
        </button>
      )}

      {/* bottom bar, your main controls */}
      <div className="target-bar">
        {/* M1 fix: on iOS (Safari) SpeechRecognition flat out doesn't exist
            show the button still but disabled + dim it so user knows why */}
        <button
          className={`mic-btn${isListening ? ' mic-active' : ''}${!canVoice ? ' mic-disabled' : ''}`}
          type="button"
          onClick={startListening}
          aria-label={isListening ? 'Listening…' : canVoice ? 'Tap to speak' : 'Voice not supported — type instead'}
          disabled={!canVoice}
        >
          <Mic size={22} />
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
            aria-label="Type a target"
          />
          <button className="go-btn" type="button" onClick={submitTypedTarget} aria-label="Submit target">
            Go
          </button>
        </div>

        <button
          type="button"
          className="scan-btn"
          onClick={handleScanClick}
          aria-label={isRunning ? 'Stop' : 'Go and scan'}
        >
          {isRunning ? <Square size={18} /> : <ScanLine size={20} />}
        </button>
      </div>

      {/* status strip, keep it short */}
      <div className="signal-strip" aria-live="polite">
        <strong>{status}</strong>
        <span>
          {match
            ? `${match.direction} · ${match.distance}${match.fromAi ? ' · ✦AI' : ''}`
            : aiLabel === 'scanning' ? 'AI scanning…'
            // M1 fix: 'haptics unavailable' sounds broken. 'visual guidance' sounds intentional
            : hapticsOk ? 'scan slowly' : 'visual guidance mode'}
        </span>
      </div>
    </main>
  );
}

// camera helpers, widest lens if possible
async function getWideCameraStream() {
  const base = {
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 }, aspectRatio: { ideal: 16 / 9 } },
    audio: false,
  };
  const first = await navigator.mediaDevices.getUserMedia(base);
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter(d => d.kind === 'videoinput');
  const backCams = cameras.filter(d => /back|rear|environment/i.test(d.label));
  const wide = backCams.find(d => /ultra|wide/i.test(d.label))
    || backCams[0]
    || cameras.find(d => /ultra|wide/i.test(d.label));
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
  try { await track.applyConstraints({ advanced: [{ zoom: caps.zoom.min }] }); } catch { /* ok, ignore zoom */ }
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

// detect helpers, this is the guidance math
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
  const gettingNear = prevArea > 0 && area > prevArea * 1.05;

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
    .filter(p => p.centerScore > 0.22 && p.score > 0.22)
    .sort((a, b) => b.centerScore - a.centerScore)[0];
}

function estimateDistance(area) {
  if (area > 0.24) return 'very close';
  if (area > 0.14) return 'close';
  if (area > 0.07) return 'medium';
  return 'far';
}

createRoot(document.getElementById('root')).render(<App />);
