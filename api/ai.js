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

export default async function handler(req, res) {
  // CORS for same-origin only (the deployed app)
  res.setHeader('Vary', 'Origin');

  if (req.method === 'OPTIONS') {
    res.setHeader('Allow', 'POST, OPTIONS');
    return res.status(204).end();
  }
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Method not allowed' });
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
        'http-referer': req.headers.origin || 'https://pulse-point.vercel.app',
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
