// Class-aware distance estimation.
//
// Pinhole camera model:
//   distance = (real_object_width * focal_length_px) / observed_bbox_width_px
//
// We don't know exact focal length per device, so we estimate it from a typical
// phone main rear camera horizontal FOV (~64°). For a 1920px frame:
//   focal_px = (1920 / 2) / tan(32°) ≈ 1536px
//
// Accuracy is roughly ±25–35% — much better than raw bbox-area buckets, and the
// answer is in METERS instead of "close / very close." That number drives the
// haptic + spoken guidance for blind users, where "1.2 m, 30 degrees right" beats
// "close · right" every time.

const ASSUMED_HFOV_DEG = 64;

// Typical real-world widths in cm. These are deliberately rough — real objects
// vary, but we just need order-of-magnitude correctness for guidance.
export const REFERENCE_WIDTHS_CM = {
  person: 45,
  bicycle: 60, motorcycle: 100, car: 180, bus: 250, truck: 250, train: 300,
  airplane: 3000, boat: 200,
  'traffic light': 30, 'fire hydrant': 40, 'stop sign': 75,
  'parking meter': 20, bench: 150,
  bird: 15, cat: 25, dog: 30, horse: 60, sheep: 50, cow: 80,
  elephant: 200, bear: 80, zebra: 80, giraffe: 80,
  backpack: 35, umbrella: 100, handbag: 30, tie: 8, suitcase: 50,
  frisbee: 25, skis: 12, snowboard: 25, 'sports ball': 22,
  kite: 80, 'baseball bat': 7, 'baseball glove': 25,
  skateboard: 20, surfboard: 50, 'tennis racket': 28,
  bottle: 7, 'wine glass': 7, cup: 8, fork: 3, knife: 3, spoon: 4,
  bowl: 18, banana: 18, apple: 8, sandwich: 15, orange: 8,
  broccoli: 12, carrot: 4, 'hot dog': 15, pizza: 30, donut: 10, cake: 22,
  chair: 50, couch: 200, 'potted plant': 25, bed: 200,
  'dining table': 100, toilet: 50,
  tv: 90, laptop: 35, mouse: 10, remote: 5, keyboard: 36, 'cell phone': 7.5,
  microwave: 50, oven: 60, toaster: 30, sink: 50, refrigerator: 80,
  book: 14, clock: 25, vase: 15, scissors: 15,
  'teddy bear': 30, 'hair drier': 25, toothbrush: 2,
};

export function focalLengthPx(frameWidthPx, hFovDeg = ASSUMED_HFOV_DEG) {
  return (frameWidthPx / 2) / Math.tan((hFovDeg / 2) * Math.PI / 180);
}

export function estimateDistanceMeters(label, bboxWidthPx, frameWidthPx) {
  const ref = REFERENCE_WIDTHS_CM[(label || '').toLowerCase()];
  if (!ref || !bboxWidthPx || !frameWidthPx) return null;
  const focal = focalLengthPx(frameWidthPx);
  const distanceCm = (ref * focal) / bboxWidthPx;
  if (!isFinite(distanceCm) || distanceCm <= 0) return null;
  return distanceCm / 100;
}

export function describeDistanceMeters(meters) {
  if (meters == null) return null;
  if (meters < 0.4)  return 'within reach';
  if (meters < 0.9)  return "arm's length";
  if (meters < 1.6)  return 'one step away';
  if (meters < 3)    return 'a few steps away';
  if (meters < 6)    return 'across the room';
  return 'far';
}

// Fallback when we don't have a reference width for the class (e.g., AI-found
// objects with arbitrary labels).
export function describeDistanceArea(bbox, frame) {
  const [, , w, h] = bbox;
  const area = (w * h) / (frame.width * frame.height);
  if (area > 0.24) return 'very close';
  if (area > 0.14) return 'close';
  if (area > 0.07) return 'medium';
  return 'far';
}

export function formatMeters(m) {
  if (m == null) return null;
  if (m < 1) return `${Math.round(m * 100)} cm`;
  if (m < 10) return `${m.toFixed(1)} m`;
  return `${Math.round(m)} m`;
}
