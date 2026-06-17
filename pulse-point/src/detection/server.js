const SERVER_URL = import.meta.env.VITE_SERVER_URL || '';
const ENDPOINT = SERVER_URL ? `${SERVER_URL}/detect` : '/api/detect';

let _available = true;
let _lastFailTime = 0;
const BACKOFF_MS = 10_000;

/**
 * Send a video frame to PulsePointNet and return a prediction in the same
 * shape as engine.js ({ class, score, bbox: [x,y,w,h] in pixels }) or null.
 */
export async function detectWithServer(video, target = '') {
  if (!video || video.readyState < 2) return null;

  const now = Date.now();
  if (!_available && now - _lastFailTime < BACKOFF_MS) return null;

  const W = video.videoWidth || 640;
  const H = video.videoHeight || 480;

  const canvas = document.createElement('canvas');
  canvas.width  = W;
  canvas.height = H;
  canvas.getContext('2d').drawImage(video, 0, 0, W, H);

  return new Promise(resolve => {
    canvas.toBlob(async blob => {
      if (!blob) { resolve(null); return; }
      try {
        const fd = new FormData();
        fd.append('image', blob, 'frame.jpg');
        if (target) fd.append('target', target);

        const res = await fetch(ENDPOINT, {
          method: 'POST',
          body: fd,
          signal: AbortSignal.timeout(5000),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data = await res.json();
        _available = true;

        if (!data.detected) { resolve(null); return; }

        const bb = data.boundingBox;
        resolve({
          class:        data.name,
          score:        data.confidence,
          bbox:         [bb.x * W, bb.y * H, bb.width * W, bb.height * H],
          fromServer:   true,
          model:        data.model || 'PulsePointNet',
          alternatives: data.alternatives || [],
          latency_ms:   data.latency_ms,
        });
      } catch {
        _available = false;
        _lastFailTime = Date.now();
        resolve(null);
      }
    }, 'image/jpeg', 0.82);
  });
}

export function isServerAvailable() {
  return _available || Date.now() - _lastFailTime >= BACKOFF_MS;
}
