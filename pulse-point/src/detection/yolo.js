// YOLO inference pipeline: load → run → decode → NMS.
// Backend: TF.js WebGL (default). Phone perf is the limiter.

import * as tf from '@tensorflow/tfjs';
import { COCO_LABELS } from './coco.js';

const YOLO_INPUT = 640;
const CONF_THRESHOLD = 0.25;
const NMS_IOU_THRESHOLD = 0.45;
const MAX_DETECTIONS = 60;

export async function loadYoloModel() {
  return tf.loadGraphModel('/yolo11n_web_model/model.json');
}

export async function runYolo(video, model) {
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
    outputTensor = Array.isArray(output) ? output[0] : (output?.output0 || output);
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
  const [, dim1, dim2] = shape;
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
      if (score > bestScore) { bestScore = score; bestClass = c; }
    }

    if (bestScore >= CONF_THRESHOLD) {
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

  return nonMaxSuppression(predictions, NMS_IOU_THRESHOLD, MAX_DETECTIONS);
}

function nonMaxSuppression(predictions, iouThreshold, maxDetections) {
  const sorted = [...predictions].sort((a, b) => b.score - a.score);
  const selected = [];
  for (const pred of sorted) {
    if (selected.length >= maxDetections) break;
    let keep = true;
    for (const picked of selected) {
      if (pred.class === picked.class && iou(pred.bbox, picked.bbox) > iouThreshold) {
        keep = false;
        break;
      }
    }
    if (keep) selected.push(pred);
  }
  return selected;
}

export function iou(a, b) {
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
