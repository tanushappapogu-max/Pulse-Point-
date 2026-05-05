// COCO labels + alias resolution + fuzzy matching.
// Pure functions, no React, no DOM — easy to test in isolation.

export const COCO_LABELS = [
  'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
  'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
  'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
  'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
  'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
  'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
  'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake',
  'chair','couch','potted plant','bed','dining table','toilet','tv','laptop',
  'mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink',
  'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
  'toothbrush',
];

const COCO_KNOWN = new Set(COCO_LABELS);

// Map common spoken/typed phrasing to COCO classes.
export const TARGET_ALIASES = {
  phone: 'cell phone', iphone: 'cell phone', android: 'cell phone',
  'my phone': 'cell phone', 'cell phone': 'cell phone', mobile: 'cell phone',
  'computer mouse': 'mouse', trackpad: 'mouse', 'my mouse': 'mouse',
  tv: 'tv', television: 'tv', monitor: 'tv', screen: 'tv',
  sofa: 'couch', couch: 'couch',
  laptop: 'laptop', computer: 'laptop', macbook: 'laptop', notebook: 'laptop',
  remote: 'remote', 'tv remote': 'remote', 'remote control': 'remote',
  ship: 'boat',
  cup: 'cup', mug: 'cup', glass: 'cup',
  bottle: 'bottle', 'water bottle': 'bottle',
  keys: 'remote', // bad fallback — AI handles it better
  pen: 'cell phone', pencil: 'cell phone',
};

const FUZZY_THRESHOLD = 0.42;

export function normalizeTargetText(text) {
  return (text || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function tokenize(text) {
  return normalizeTargetText(text).split(' ').filter(Boolean);
}

function bigrams(text) {
  const compact = normalizeTargetText(text).replace(/\s+/g, '');
  const grams = [];
  for (let i = 0; i < compact.length - 1; i++) grams.push(compact.slice(i, i + 2));
  return grams;
}

function jaccard(a, b) {
  if (!a.length || !b.length) return 0;
  const setA = new Set(a);
  const setB = new Set(b);
  let intersection = 0;
  setA.forEach(item => { if (setB.has(item)) intersection++; });
  return intersection / (setA.size + setB.size - intersection || 1);
}

export function stringSimilarity(a, b) {
  const normA = normalizeTargetText(a);
  const normB = normalizeTargetText(b);
  if (!normA || !normB) return 0;
  if (normA === normB) return 1;
  const tokensA = tokenize(normA);
  const tokensB = tokenize(normB);
  const tokenScore = jaccard(tokensA, tokensB);
  const gramScore = jaccard(bigrams(normA), bigrams(normB));
  const prefixScore = normA.startsWith(normB) || normB.startsWith(normA) ? 0.85 : 0;
  return Math.min(1, Math.max(tokenScore * 0.9 + gramScore * 0.4, prefixScore));
}

export function resolveCocoTarget(tgt) {
  if (!tgt) return null;
  const norm = normalizeTargetText(tgt);
  if (!norm) return null;
  if (COCO_KNOWN.has(norm)) return norm;
  return TARGET_ALIASES[norm] || null;
}

export function findClosestCocoLabel(text) {
  const norm = normalizeTargetText(text);
  if (!norm) return null;
  let best = { label: null, score: 0 };
  for (const label of COCO_LABELS) {
    const score = stringSimilarity(norm, label);
    if (score > best.score) best = { label, score };
  }
  return best.score >= FUZZY_THRESHOLD ? best : null;
}

export function getAliases(target) {
  const norm = normalizeTargetText(target);
  const alias = TARGET_ALIASES[norm];
  return alias ? [alias] : [norm];
}
