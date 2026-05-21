// Vercel serverless proxy for OpenRouter/Gemini calls.
//
// Why this exists: the previous client called OpenRouter directly with a key
// in `import.meta.env.VITE_GEMINI_API_KEY`. Vite inlines VITE_* envs into the
// production bundle, so anyone visiting the deployed site could extract the
// key from the JS and burn the user's OpenRouter credit. This proxy keeps the
// key server-side.
//
// Deploy steps for the operator:
//   1. In Vercel project settings → Environment Variables, set
//        OPENROUTER_API_KEY = <your key>
//      (NOT prefixed with VITE_)
//   2. Remove VITE_GEMINI_API_KEY from production env if it was set.
//   3. Redeploy.
//
// Local development:
//   - `vercel dev` will run this function alongside vite (recommended), OR
//   - Keep VITE_GEMINI_API_KEY in .env.local for direct calls — the client
//     falls back to that automatically when the proxy is unavailable.

export const config = {
  runtime: 'nodejs',
  // 30s is plenty for a single vision request; tighter than Vercel's 60s
  // default so a hung upstream doesn't burn the function timeout.
  maxDuration: 30,
};

const ALLOWED_MODELS = new Set([
  'google/gemini-2.0-flash-exp:free',
  'google/gemini-2.0-flash-lite-preview-02-05:free',
  'google/gemini-flash-1.5',
  'google/gemini-flash-1.5-8b',
]);

// Allowed origins: production deployment + local dev ports.
const ALLOWED_ORIGINS = new Set([
  process.env.ALLOWED_ORIGIN || 'https://pulse-point.vercel.app',
  'http://localhost:5173',
  'http://localhost:4173',
]);

// In-memory sliding window rate limiter.
// Vercel warm instances share this map; cold starts reset it — acceptable
// trade-off for a free-tier protection layer.
// Key = client IP, value = array of request timestamps within the window.
const rateLimitMap = new Map();
const RATE_LIMIT_WINDOW_MS = 60_000; // 1 minute
const RATE_LIMIT_MAX = 10;           // max requests per window per IP

function checkRateLimit(ip) {
  const now = Date.now();
  const prev = (rateLimitMap.get(ip) || []).filter(t => now - t < RATE_LIMIT_WINDOW_MS);
  if (prev.length >= RATE_LIMIT_MAX) return false;
  prev.push(now);
  rateLimitMap.set(ip, prev);
  return true;
}

// Evict stale entries every 2 min so the map doesn't grow on warm instances.
setInterval(() => {
  const cutoff = Date.now() - RATE_LIMIT_WINDOW_MS;
  for (const [ip, times] of rateLimitMap) {
    const fresh = times.filter(t => t > cutoff);
    if (fresh.length === 0) rateLimitMap.delete(ip);
    else rateLimitMap.set(ip, fresh);
  }
}, 120_000);

export default async function handler(req, res) {
  const origin = req.headers.origin || '';

  // Strict origin enforcement — only the deployed app (and local dev) may call this.
  if (origin && !ALLOWED_ORIGINS.has(origin)) {
    return res.status(403).json({ error: 'Forbidden' });
  }

  res.setHeader('Vary', 'Origin');
  if (origin && ALLOWED_ORIGINS.has(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  }

  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Per-IP rate limiting. X-Forwarded-For is set by Vercel's edge.
  const clientIp =
    (req.headers['x-forwarded-for'] || '').split(',')[0].trim() ||
    req.socket?.remoteAddress ||
    'unknown';
  if (!checkRateLimit(clientIp)) {
    res.setHeader('Retry-After', '60');
    return res.status(429).json({ error: 'Too many requests — try again in a minute' });
  }

  const key = process.env.OPENROUTER_API_KEY;
  if (!key) {
    return res.status(500).json({ error: 'AI service not configured. Set OPENROUTER_API_KEY in Vercel env.' });
  }

  // Parse body (Vercel auto-parses JSON when content-type is set, but be defensive)
  let body = req.body;
  if (typeof body === 'string') {
    try { body = JSON.parse(body); }
    catch { return res.status(400).json({ error: 'Invalid JSON body' }); }
  }
  if (!body || typeof body !== 'object' || !Array.isArray(body.messages)) {
    return res.status(400).json({ error: 'Invalid request body — expected { model, messages, ... }' });
  }

  // Allow-list models so a compromised client can't bill us for a giant model.
  if (body.model && !ALLOWED_MODELS.has(body.model)) {
    return res.status(400).json({ error: `Model not allowed: ${body.model}` });
  }
  if (!body.model) body.model = 'google/gemini-2.0-flash-exp:free';

  // Cap output to prevent abuse.
  body.max_tokens = Math.min(Math.max(body.max_tokens || 256, 32), 512);

  // Cap message count to prevent giant context bills.
  if (body.messages.length > 8) {
    return res.status(400).json({ error: 'Too many messages' });
  }

  try {
    const upstream = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        authorization: `Bearer ${key}`,
        'content-type': 'application/json',
        // OpenRouter likes a referer for free-tier rate limit fairness
        'http-referer': origin || 'https://pulse-point.vercel.app',
        'x-title': 'Pulse Point',
      },
      body: JSON.stringify(body),
    });

    const data = await upstream.json();
    return res.status(upstream.status).json(data);
  } catch (err) {
    return res.status(502).json({ error: 'AI upstream unavailable' });
  }
}
