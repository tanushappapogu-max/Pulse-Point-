// AI fallback: when YOLO doesn't find the user's target (e.g., it's a class
// outside COCO, or partially occluded), we send a frame to OpenRouter/Gemini
// for an open-vocabulary detection.
//
// Calls go through /api/ai (Vercel serverless) so the API key stays on the
// server. For local dev, falls back to direct OpenRouter calls if the env var
// VITE_GEMINI_API_KEY is set — convenience only, do not deploy with that var.

const PROXY_ENDPOINT = '/api/ai';
const DIRECT_ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions';
const DEV_KEY = import.meta.env.VITE_GEMINI_API_KEY || '';
const GEMINI_MODEL = 'google/gemini-2.0-flash-exp:free';

async function callAI(messages, maxTokens = 256) {
  const body = {
    model: GEMINI_MODEL,
    temperature: 0,
    max_tokens: maxTokens,
    messages,
  };

  // Production path: proxy through our serverless function
  try {
    const res = await fetch(PROXY_ENDPOINT, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (res.ok) return await res.json();
    // 404 means the proxy isn't deployed (likely vite dev). Fall through.
    if (res.status !== 404) {
      const err = await res.json().catch(() => ({}));
      return { __error: err.error?.message || err.error || `HTTP ${res.status}` };
    }
  } catch {
    // network error — try dev fallback
  }

  // Dev fallback (only if key is in env)
  if (DEV_KEY) {
    try {
      const res = await fetch(DIRECT_ENDPOINT, {
        method: 'POST',
        headers: {
          authorization: `Bearer ${DEV_KEY}`,
          'content-type': 'application/json',
        },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (data.error) return { __error: data.error?.message || 'AI unavailable' };
      return data;
    } catch {
      return { __error: 'Network error' };
    }
  }

  return { __error: 'AI not configured' };
}

function parseFirstJson(text) {
  if (!text) return null;
  if (Array.isArray(text)) text = text.map(p => p?.text || '').join('');
  const s = text.indexOf('{'), e = text.lastIndexOf('}') + 1;
  if (s < 0 || e <= s) return null;
  try { return JSON.parse(text.slice(s, e)); } catch { return null; }
}

function contentText(data) {
  return data?.choices?.[0]?.message?.content || '';
}

/**
 * Open-vocabulary single-object localization.
 * Returns { found: true, x, y, w, h (all 0..1), confidence } | null | { __error }
 */
export async function callGeminiBox(imageBase64, target) {
  if (!imageBase64) return null;
  const prompt =
`Find the object "${target}" in this camera frame.

Return ONLY valid JSON:
{"found":true,"label":"${target}","box_2d":[ymin,xmin,ymax,xmax],"confidence":0.0}

Requirements:
- box_2d uses integers 0..1000 (normalized to image dimensions)
- return ONE box for the clearest, most reachable instance
- if not visible: {"found":false,"label":"${target}","box_2d":null,"confidence":0.0}
- no markdown, no commentary, JSON only`;

  const data = await callAI([{
    role: 'user',
    content: [
      { type: 'text', text: prompt },
      { type: 'image_url', image_url: { url: imageBase64 } },
    ],
  }], 256);

  if (data.__error) return { __error: data.__error };

  const parsed = parseFirstJson(contentText(data));
  if (!parsed?.found || !parsed.box_2d) return null;
  const [yMin, xMin, yMax, xMax] = parsed.box_2d;
  if ([yMin, xMin, yMax, xMax].some(v => typeof v !== 'number')) return null;
  return {
    found: true,
    x: xMin / 1000,
    y: yMin / 1000,
    w: (xMax - xMin) / 1000,
    h: (yMax - yMin) / 1000,
    confidence: parsed.confidence || 0.8,
  };
}

/**
 * Autopilot: given a vague intent, return ranked candidate object names that
 * match what's visible in the frame.
 */
export async function callGeminiAutopilot(imageBase64, intent) {
  if (!imageBase64) return null;
  const prompt =
`User said: "${intent}"
Look at this image. Identify 2–4 real visible objects that best match the user's intent, ranked by how obvious/accessible they are.
CRITICAL: Only list objects that are clearly visible as distinct items — NOT walls, floors, tables, or surfaces themselves. Only things ON surfaces or in the scene as separate objects.
Return JSON: {"candidates":["item1","item2"],"positions":["short position like 'center table'","short position"]}
If nothing matches, return {"candidates":[],"positions":[]}`;

  const data = await callAI([{
    role: 'user',
    content: [
      { type: 'text', text: prompt },
      { type: 'image_url', image_url: { url: imageBase64 } },
    ],
  }], 256);

  if (data.__error) return { __error: data.__error };
  return parseFirstJson(contentText(data));
}
