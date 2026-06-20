/**
 * Visual grounding — hybrid mode.
 *
 * If VITE_SERVER_URL is set (Modal GPU server with LocateAnything-3B):
 *   → POST /detect with the video frame + target label
 *   → falls back to Florence-2 if the server is unreachable
 *
 * If VITE_SERVER_URL is not set (local / Vercel without env var):
 *   → Florence-2-base-ft runs fully in-browser via Transformers.js (ONNX WASM)
 *     Model downloads once and is cached in IndexedDB (~130 MB).
 *
 * Exported API (unchanged so App.jsx needs no edits):
 *   detectWithServer(video, target) → result object | null
 *   isServerAvailable()            → bool
 */

import {
  Florence2ForConditionalGeneration,
  AutoProcessor,
  RawImage,
} from '@huggingface/transformers';

const SERVER_URL = import.meta.env.VITE_SERVER_URL || '';
const MODEL_ID   = 'onnx-community/Florence-2-base-ft';
const TASK       = '<OPEN_VOCABULARY_DETECTION>';

// ── Florence-2 browser state ─────────────────────────────────────────────────
let _processor = null;
let _model     = null;
let _loading   = false;
let _ready     = false;
let _failed    = false;

async function _loadFlorence() {
  if (_loading || _ready || _failed) return;
  _loading = true;
  try {
    _processor = await AutoProcessor.from_pretrained(MODEL_ID);
    _model = await Florence2ForConditionalGeneration.from_pretrained(MODEL_ID, {
      dtype: { encoder_model: 'fp16', decoder_model_merged: 'q4' },
    });
    _ready = true;
    console.log('[Ground] Florence-2 ready');
  } catch (e) {
    _failed = true;
    console.warn('[Ground] Florence-2 load failed:', e);
  } finally {
    _loading = false;
  }
}

// Pre-load Florence-2 if no server configured — warms cache on first visit
if (!SERVER_URL) _loadFlorence();

// ── Server reachability ───────────────────────────────────────────────────────
let _serverOk    = null;   // null = unknown, true/false = tested
let _serverCheck = 0;

async function _checkServer() {
  const now = Date.now();
  if (now - _serverCheck < 30_000) return _serverOk;
  _serverCheck = now;
  try {
    const r = await fetch(`${SERVER_URL}/health`, { signal: AbortSignal.timeout(4000) });
    _serverOk = r.ok;
  } catch {
    _serverOk = false;
  }
  if (!_serverOk && !_ready && !_loading && !_failed) _loadFlorence();
  return _serverOk;
}

// Kick off an early health check when server URL is configured
if (SERVER_URL) _checkServer();

// ── Public API ────────────────────────────────────────────────────────────────
export function isServerAvailable() {
  return (SERVER_URL && _serverOk === true) || _ready;
}

export async function detectWithServer(video, target = '') {
  if (!video || video.readyState < 2 || !target) return null;

  const W = video.videoWidth  || 640;
  const H = video.videoHeight || 480;

  const canvas = document.createElement('canvas');
  canvas.width  = W;
  canvas.height = H;
  canvas.getContext('2d').drawImage(video, 0, 0, W, H);

  // ── Try GPU server ──────────────────────────────────────────────────────────
  if (SERVER_URL) {
    const ok = await _checkServer();
    if (ok) {
      try {
        const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.85));
        const form = new FormData();
        form.append('file', blob, 'frame.jpg');
        form.append('target', target);

        const t0 = performance.now();
        const r  = await fetch(`${SERVER_URL}/detect`, {
          method: 'POST',
          body: form,
          signal: AbortSignal.timeout(8000),
        });
        if (r.ok) {
          const data = await r.json();
          if (data.detected) {
            const { x, y, width, height } = data.boundingBox;
            return {
              class:        data.name || target,
              score:        data.confidence ?? 0.90,
              bbox:         [x * W, y * H, width * W, height * H],
              fromServer:   true,
              model:        'LocateAnything',
              alternatives: [],
              latency_ms:   Math.round(performance.now() - t0),
            };
          }
          return null;
        }
      } catch (e) {
        _serverOk = false;
        console.warn('[Ground] Server error, falling back to Florence-2:', e.message);
        if (!_ready && !_loading && !_failed) _loadFlorence();
      }
    }
  }

  // ── Florence-2 browser fallback ─────────────────────────────────────────────
  if (!_ready) {
    if (!_loading && !_failed) _loadFlorence();
    return null;
  }

  const t0 = performance.now();
  try {
    const image  = await RawImage.fromCanvas(canvas);
    const inputs = await _processor(image, TASK + target);

    const ids = await _model.generate({ ...inputs, max_new_tokens: 256 });
    const raw = _processor.batch_decode(ids, { skip_special_tokens: false })[0];
    const out = _processor.post_process_generation(raw, TASK, [W, H]);
    const det = out[TASK];

    if (!det?.bboxes?.length) return null;

    const [x1, y1, x2, y2] = det.bboxes[0];

    return {
      class:        det.labels?.[0] || target,
      score:        0.88,
      bbox:         [x1, y1, x2 - x1, y2 - y1],
      fromServer:   false,
      model:        'Florence-2',
      alternatives: det.bboxes.slice(1, 5).map((b, i) => ({
        name:       det.labels?.[i + 1] || target,
        confidence: 0.75,
      })),
      latency_ms:   Math.round(performance.now() - t0),
    };
  } catch (e) {
    console.warn('[Ground] Inference error:', e);
    return null;
  }
}
