/**
 * Browser-side visual grounding using Florence-2-base-ft.
 *
 * Exports the same API as before so App.jsx needs no changes:
 *   detectWithServer(video, target) → {class, score, bbox, latency_ms, …} | null
 *   isServerAvailable()            → bool
 *
 * Florence-2 runs fully in-browser via Transformers.js (ONNX WASM backend).
 * The model downloads once and is cached in IndexedDB.
 */

import {
  Florence2ForConditionalGeneration,
  AutoProcessor,
  RawImage,
} from '@huggingface/transformers';

const MODEL_ID = 'onnx-community/Florence-2-base-ft';
const TASK     = '<OPEN_VOCABULARY_DETECTION>';

let _processor = null;
let _model     = null;
let _loading   = false;
let _ready     = false;
let _failed    = false;

async function _load() {
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

// Begin loading immediately in background — cached after first visit
_load();

export function isServerAvailable() {
  return _ready;
}

export async function detectWithServer(video, target = '') {
  if (!video || video.readyState < 2 || !target) return null;
  if (!_ready) {
    if (!_loading && !_failed) _load();
    return null;
  }

  const W = video.videoWidth  || 640;
  const H = video.videoHeight || 480;

  const canvas = document.createElement('canvas');
  canvas.width  = W;
  canvas.height = H;
  canvas.getContext('2d').drawImage(video, 0, 0, W, H);

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
