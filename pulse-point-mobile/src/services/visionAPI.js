const API_URL = __DEV__
  ? 'http://192.168.1.100:8000'
  : 'https://pulse-point-api.onrender.com';

let _serverUrl = API_URL;

export function setServerUrl(url) {
  _serverUrl = url.replace(/\/+$/, '');
}

export function getServerUrl() {
  return _serverUrl;
}

export async function detectObject(imageUri, targetName) {
  const formData = new FormData();
  formData.append('image', {
    uri: imageUri,
    type: 'image/jpeg',
    name: 'frame.jpg',
  });
  if (targetName) {
    formData.append('target', targetName.trim().toLowerCase());
  }

  const response = await fetch(`${_serverUrl}/detect`, {
    method: 'POST',
    body: formData,
    headers: { 'Accept': 'application/json' },
  });

  if (!response.ok) {
    throw new Error(`Vision API error: ${response.status}`);
  }

  return response.json();
}

export async function checkHealth() {
  try {
    const response = await fetch(`${_serverUrl}/health`, { timeout: 5000 });
    if (!response.ok) return false;
    const data = await response.json();
    return data.status === 'ok';
  } catch {
    return false;
  }
}

export async function listKnownObjects() {
  const response = await fetch(`${_serverUrl}/objects`);
  if (!response.ok) throw new Error('Failed to fetch objects');
  const data = await response.json();
  return data.objects;
}

// ── Tea Text CNN ─────────────────────────────────────────────────────────────

/**
 * Classify a tea text string using the server-side TextCNN.
 *
 * @param {string} text   Tea description, product label, or spoken query
 * @param {number} topK   Number of alternative tea types to return (default 3)
 * @returns {Promise<{
 *   tea_type: {label: string, confidence: number},
 *   flavors:  Array<{label: string, confidence: number}>,
 *   quality:  {label: string, confidence: number},
 *   alternatives: Array<{label: string, confidence: number}>,
 *   embedding: number[],
 *   latency_ms: number,
 * }>}
 */
export async function classifyText(text, topK = 3) {
  const response = await fetch(`${_serverUrl}/classify-text`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify({ text: text.trim(), top_k: topK }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(`TextCNN API error ${response.status}: ${err.error || response.statusText}`);
  }

  return response.json();
}

/**
 * Fetch the tea label ontology (types, flavors, quality tiers) from the server.
 * Cache-friendly: the schema rarely changes.
 */
export async function fetchTeaSchema() {
  const response = await fetch(`${_serverUrl}/tea-schema`, {
    headers: { 'Accept': 'application/json' },
  });
  if (!response.ok) throw new Error('Failed to fetch tea schema');
  return response.json();
}

export function apiResultToDetection(apiResult, targetName) {
  if (!apiResult.detected) return null;

  return {
    id: `${(targetName || apiResult.name).toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`,
    name: apiResult.name,
    source: 'cnn',
    confidence: apiResult.confidence,
    position: {
      x: apiResult.boundingBox.x + apiResult.boundingBox.width / 2,
      y: apiResult.boundingBox.y + apiResult.boundingBox.height / 2,
      zMeters: estimateDistance(apiResult.boundingBox),
    },
    boundingBox: apiResult.boundingBox,
    alternatives: apiResult.alternatives || [],
    latencyMs: apiResult.latency_ms,
  };
}

function estimateDistance(bbox) {
  const area = bbox.width * bbox.height;
  if (area > 0.4) return 0.3;
  if (area > 0.2) return 0.8;
  if (area > 0.1) return 1.5;
  if (area > 0.05) return 2.5;
  return 3.5;
}
