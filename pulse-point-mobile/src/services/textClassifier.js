/**
 * textClassifier.js — Offline tea text classifier for Pulse Point
 *
 * Lightweight keyword-based fallback when the Python TextCNN server is
 * unavailable. Uses the same label ontology as the server-side model so
 * callers can treat both paths identically.
 *
 * Usage:
 *   import { classifyTextOffline } from './textClassifier';
 *   const result = classifyTextOffline('gyokuro shade grown umami');
 *   // { tea_type: {label:'green', confidence:0.87}, flavors:[...], ... }
 *
 * When the server IS reachable, prefer classifyText() from visionAPI.js.
 */

// ── Label ontology (mirrors tea_dataset.py) ──────────────────────────────────

export const TEA_TYPES = [
  'green', 'black', 'white', 'oolong',
  'pu_erh', 'herbal', 'yellow', 'dark',
];

export const FLAVOR_LABELS = [
  'floral', 'earthy', 'grassy', 'smoky', 'sweet',
  'bitter', 'fruity', 'nutty', 'vegetal', 'marine',
];

export const QUALITY_TIERS = ['ceremonial', 'premium', 'standard', 'culinary'];


// ── Keyword dictionaries ──────────────────────────────────────────────────────

/** Maps tea type index → set of trigger words/phrases */
const TEA_TYPE_KEYWORDS = [
  // 0: green
  new Set([
    'green', 'sencha', 'matcha', 'gyokuro', 'kabusecha', 'bancha',
    'hojicha', 'genmaicha', 'longjing', 'dragonwell', 'bi luo chun',
    'anji bai cha', 'lu shan', 'shincha', 'tencha', 'kukicha',
    'gunpowder', 'pan fired', 'steamed', 'needles',
  ]),
  // 1: black
  new Set([
    'black', 'darjeeling', 'assam', 'ceylon', 'lapsang', 'souchong',
    'keemun', 'english breakfast', 'irish breakfast', 'earl grey',
    'yunnan', 'dianhong', 'nilgiri', 'golden monkey', 'orthodox',
    'breakfast', 'chai', 'malty', 'brisk', 'bergamot',
  ]),
  // 2: white
  new Set([
    'white', 'silver needle', 'bai hao', 'yinzhen', 'white peony',
    'bai mu dan', 'shou mei', 'gong mei', 'moonlight', 'bai yue liang',
    'downy', 'minimally processed', 'silver tips',
  ]),
  // 3: oolong
  new Set([
    'oolong', 'tieguanyin', 'iron goddess', 'dan cong', 'dong ding',
    'ali shan', 'da hong pao', 'oriental beauty', 'wuyi', 'rock oolong',
    'phoenix', 'milk oolong', 'baozhong', 'pouchong', 'concubine',
    'high mountain oolong', 'wu yi',
  ]),
  // 4: pu_erh
  new Set([
    'pu erh', 'pu-erh', 'puer', 'shou', 'sheng', 'tuo cha', 'raw pu',
    'ripe pu', 'yunnan pu', 'gushu', 'ancient tree', 'cake compressed',
    'cooked dark',
  ]),
  // 5: herbal
  new Set([
    'herbal', 'tisane', 'chamomile', 'peppermint', 'hibiscus', 'rooibos',
    'ginger root', 'lavender', 'echinacea', 'lemon balm', 'rosehip',
    'turmeric', 'valerian', 'spearmint', 'licorice', 'elderflower',
    'nettle', 'red bush', 'caffeine free',
  ]),
  // 6: yellow
  new Set([
    'yellow', 'jun shan', 'meng ding', 'huang ya', 'huang', 'smothered',
    'mellow green', 'yellow bud',
  ]),
  // 7: dark
  new Set([
    'liu bao', 'fu brick', 'anhua', 'tibetan', 'heicha', 'dark tea',
    'post-fermented', 'post fermented', 'golden flower', 'guangxi',
    'hunan', 'betel',
  ]),
];

const FLAVOR_KEYWORDS = [
  // 0: floral
  new Set(['floral', 'flower', 'flowery', 'orchid', 'jasmine', 'bloom', 'blossom', 'rose', 'lavender', 'elderflower']),
  // 1: earthy
  new Set(['earthy', 'earth', 'soil', 'musty', 'loamy', 'humus', 'woody', 'wood', 'forest', 'mushroom', 'camphor']),
  // 2: grassy
  new Set(['grassy', 'grass', 'fresh', 'green', 'vegetal', 'hay', 'spinach', 'lawn']),
  // 3: smoky
  new Set(['smoky', 'smoked', 'smoke', 'fire', 'campfire', 'charred', 'roasted', 'pine']),
  // 4: sweet
  new Set(['sweet', 'honey', 'honeyed', 'sugar', 'mellow', 'syrupy', 'caramel', 'vanilla', 'candy']),
  // 5: bitter
  new Set(['bitter', 'astringent', 'brisk', 'tannic', 'sharp', 'tannin', 'dry']),
  // 6: fruity
  new Set(['fruity', 'fruit', 'berry', 'citrus', 'apricot', 'peach', 'muscatel', 'tropical', 'rosehip', 'hibiscus']),
  // 7: nutty
  new Set(['nutty', 'nut', 'walnut', 'almond', 'hazelnut', 'toasted', 'roasted', 'malty']),
  // 8: vegetal
  new Set(['vegetal', 'vegetable', 'seaweed', 'marine', 'spinach', 'cooked green', 'umami']),
  // 9: marine
  new Set(['marine', 'ocean', 'sea', 'kelp', 'umami', 'briny', 'oceanic', 'seaweed']),
];

const QUALITY_KEYWORDS = [
  // 0: ceremonial
  new Set(['ceremonial', 'gushu', 'ancient tree', 'single origin', 'premier', 'grand cru', 'imperial']),
  // 1: premium
  new Set(['premium', 'first flush', 'spring harvest', 'high mountain', 'whole leaf', 'tippy', 'silver tips']),
  // 2: standard
  new Set(['standard', 'everyday', 'loose leaf', 'orthodox', 'classic', 'regular']),
  // 3: culinary
  new Set(['culinary', 'teabag', 'tea bag', 'blend', 'fannings', 'dust', 'powder', 'budget']),
];


// ── Scoring helpers ───────────────────────────────────────────────────────────

function tokenize(text) {
  return text.toLowerCase().replace(/[^a-z0-9\s-]/g, ' ').split(/\s+/).filter(Boolean);
}

/** Score text against a keyword Set using token and bigram matching. */
function scoreAgainst(tokens, keywordSet) {
  let hits = 0;
  const bigrams = tokens.slice(0, -1).map((t, i) => `${t} ${tokens[i + 1]}`);
  const allNgrams = [...tokens, ...bigrams];
  for (const ng of allNgrams) {
    if (keywordSet.has(ng)) hits++;
  }
  return hits;
}

function softmax(scores) {
  const exp = scores.map(s => Math.exp(s));
  const sum = exp.reduce((a, b) => a + b, 0) || 1;
  return exp.map(e => e / sum);
}


// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Classify tea text offline (no server required).
 *
 * @param {string} text
 * @param {number} topK
 * @returns {{
 *   tea_type: {label: string, confidence: number},
 *   flavors:  Array<{label: string, confidence: number}>,
 *   quality:  {label: string, confidence: number},
 *   alternatives: Array<{label: string, confidence: number}>,
 *   source: 'offline-keywords'
 * }}
 */
export function classifyTextOffline(text, topK = 3) {
  const tokens = tokenize(text);

  // -- Tea type -----------------------------------------------------------
  const typeRawScores = TEA_TYPE_KEYWORDS.map(kw => scoreAgainst(tokens, kw));
  // Add a small uniform prior so all classes have non-zero probability
  const typeScoresWithPrior = typeRawScores.map(s => s + 0.1);
  const typeProbs = softmax(typeScoresWithPrior);

  const sortedTypeIdxs = typeProbs
    .map((p, i) => ({ i, p }))
    .sort((a, b) => b.p - a.p);

  const bestTypeIdx = sortedTypeIdxs[0].i;
  const alternatives = sortedTypeIdxs.slice(0, topK).map(({ i, p }) => ({
    label: TEA_TYPES[i],
    confidence: parseFloat(p.toFixed(4)),
  }));

  // -- Flavors (multi-label) -----------------------------------------------
  const FLAVOR_THRESHOLD = 0.5; // minimum hit count to activate
  const flavorHits = FLAVOR_KEYWORDS.map(kw => scoreAgainst(tokens, kw));
  const maxFlavorHit = Math.max(...flavorHits, 1);
  const flavors = flavorHits
    .map((hits, i) => ({ label: FLAVOR_LABELS[i], confidence: parseFloat((hits / maxFlavorHit).toFixed(4)) }))
    .filter(f => f.confidence >= FLAVOR_THRESHOLD)
    .sort((a, b) => b.confidence - a.confidence);

  const detectedFlavors = flavors.length > 0
    ? flavors
    : [{ label: FLAVOR_LABELS[flavorHits.indexOf(Math.max(...flavorHits))], confidence: 0.3 }];

  // -- Quality tier -------------------------------------------------------
  const qualityRaw = QUALITY_KEYWORDS.map(kw => scoreAgainst(tokens, kw));
  const qualityWithPrior = qualityRaw.map(s => s + 0.05);
  const qualityProbs = softmax(qualityWithPrior);
  const bestQualityIdx = qualityProbs.indexOf(Math.max(...qualityProbs));

  return {
    tea_type: {
      label: TEA_TYPES[bestTypeIdx],
      confidence: parseFloat(typeProbs[bestTypeIdx].toFixed(4)),
    },
    flavors: detectedFlavors,
    quality: {
      label: QUALITY_TIERS[bestQualityIdx],
      confidence: parseFloat(qualityProbs[bestQualityIdx].toFixed(4)),
    },
    alternatives,
    source: 'offline-keywords',
  };
}


/**
 * Convenience wrapper: try the server TextCNN, fall back to offline classifier.
 *
 * @param {string} text
 * @param {Function} serverClassifyFn  The classifyText() fn from visionAPI.js
 * @param {number}  topK
 */
export async function classifyTextWithFallback(text, serverClassifyFn, topK = 3) {
  try {
    const result = await serverClassifyFn(text, topK);
    return { ...result, source: 'server-cnn' };
  } catch {
    return classifyTextOffline(text, topK);
  }
}
