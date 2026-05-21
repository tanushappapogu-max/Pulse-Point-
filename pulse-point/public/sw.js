// Service worker for Pulse Point.
// Goal: cache the YOLO model weights on first load so subsequent sessions
// work without a network round-trip (the shards are ~10 MB total).
//
// Strategy:
//   /yolo11n_web_model/* → cache-first (immutable weights, versioned via cache name)
//   everything else      → network-first with cache fallback

const MODEL_CACHE = 'pulse-point-model-v1';
const APP_CACHE   = 'pulse-point-app-v1';

self.addEventListener('install', event => {
  // Activate immediately without waiting for old clients to close.
  event.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', event => {
  // Take control of existing clients so the cache is available right away.
  event.waitUntil(
    Promise.all([
      self.clients.claim(),
      // Remove stale model caches from previous versions.
      caches.keys().then(keys =>
        Promise.all(
          keys
            .filter(k => k.startsWith('pulse-point-model-') && k !== MODEL_CACHE)
            .map(k => caches.delete(k)),
        ),
      ),
    ]),
  );
});

self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  // Only intercept same-origin GET requests.
  if (request.method !== 'GET' || url.origin !== self.location.origin) return;

  if (url.pathname.startsWith('/yolo11n_web_model/')) {
    // Cache-first: return cached weights immediately; fetch and store on miss.
    event.respondWith(
      caches.open(MODEL_CACHE).then(async cache => {
        const cached = await cache.match(request);
        if (cached) return cached;
        const response = await fetch(request);
        if (response.ok) cache.put(request, response.clone());
        return response;
      }),
    );
    return;
  }

  // Network-first for HTML / JS / CSS — fall back to cache when offline.
  event.respondWith(
    fetch(request)
      .then(response => {
        if (response.ok) {
          caches.open(APP_CACHE).then(cache => cache.put(request, response.clone()));
        }
        return response;
      })
      .catch(() => caches.match(request)),
  );
});
