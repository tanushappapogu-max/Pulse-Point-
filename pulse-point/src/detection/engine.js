import * as ort from 'onnxruntime-web';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
ort.env.wasm.numThreads = 1;

const INPUT_W = 640;
const INPUT_H = 640;
const CONF_THRESH = 0.25;
const IOU_THRESH  = 0.45;
const NUM_CLASSES = 80;
const NUM_ANCHORS = 8400;

const CLASSES = [
  'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
  'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
  'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
  'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
  'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
  'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
  'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
  'donut','cake','chair','couch','potted plant','bed','dining table','toilet',
  'tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
  'toaster','sink','refrigerator','book','clock','vase','scissors',
  'teddy bear','hair drier','toothbrush',
];

let _session = null;

export async function loadModel() {
  if (_session) return _session;
  _session = await ort.InferenceSession.create('/net.onnx', {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  return _session;
}

export async function runInference(video) {
  const session = _session || await loadModel();

  const vw = video.videoWidth  || 640;
  const vh = video.videoHeight || 480;

  const scale = Math.min(INPUT_W / vw, INPUT_H / vh);
  const sw = Math.round(vw * scale);
  const sh = Math.round(vh * scale);
  const pad_x = Math.round((INPUT_W - sw) / 2);
  const pad_y = Math.round((INPUT_H - sh) / 2);

  const canvas = document.createElement('canvas');
  canvas.width  = INPUT_W;
  canvas.height = INPUT_H;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, INPUT_W, INPUT_H);
  ctx.drawImage(video, pad_x, pad_y, sw, sh);

  const px = ctx.getImageData(0, 0, INPUT_W, INPUT_H).data;
  const n  = INPUT_W * INPUT_H;
  const buf = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    buf[i]         = px[i * 4]     / 255;
    buf[n + i]     = px[i * 4 + 1] / 255;
    buf[2 * n + i] = px[i * 4 + 2] / 255;
  }

  const input = new ort.Tensor('float32', buf, [1, 3, INPUT_H, INPUT_W]);
  const out   = await session.run({ images: input });
  const raw   = out[Object.keys(out)[0]].data;

  const hits = [];
  for (let i = 0; i < NUM_ANCHORS; i++) {
    let best = 0, cls = 0;
    for (let c = 0; c < NUM_CLASSES; c++) {
      const s = raw[(4 + c) * NUM_ANCHORS + i];
      if (s > best) { best = s; cls = c; }
    }
    if (best < CONF_THRESH) continue;

    const cx = raw[0 * NUM_ANCHORS + i];
    const cy = raw[1 * NUM_ANCHORS + i];
    const bw = raw[2 * NUM_ANCHORS + i];
    const bh = raw[3 * NUM_ANCHORS + i];

    const x = ((cx - bw / 2) - pad_x) / scale;
    const y = ((cy - bh / 2) - pad_y) / scale;
    const w = bw / scale;
    const h = bh / scale;

    hits.push({ class: CLASSES[cls], score: best, bbox: [x, y, w, h] });
  }

  return _nms(hits);
}

function _nms(dets) {
  const sorted = [...dets].sort((a, b) => b.score - a.score);
  const kept = [];
  for (const d of sorted) {
    if (!kept.some(k => _iou(d.bbox, k.bbox) > IOU_THRESH)) kept.push(d);
  }
  return kept;
}

function _iou([ax, ay, aw, ah], [bx, by, bw, bh]) {
  const ix1 = Math.max(ax, bx), iy1 = Math.max(ay, by);
  const ix2 = Math.min(ax + aw, bx + bw), iy2 = Math.min(ay + ah, by + bh);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  return inter / (aw * ah + bw * bh - inter || 1);
}
