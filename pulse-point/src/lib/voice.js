// Voice input wrapper around webkitSpeechRecognition. Single-shot capture.

export const isVoiceSupported = () =>
  typeof window !== 'undefined' &&
  Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);

const VAGUE_PATTERNS = [
  /\b(something|anything)\b/i,
  /\b(eat|food|hungry|snack|drink|thirsty)\b/i,
  /\b(write|writing|draw)\b/i,
  /\b(call|text)\b/i,
];

export function isVagueIntent(text) {
  return VAGUE_PATTERNS.some(re => re.test(text));
}

export function extractTarget(text) {
  return (text || '').toLowerCase()
    .replace(/where\s+is\s+(my\s+|the\s+)?/g, '')
    .replace(/find\s+(my\s+|the\s+|a\s+)?/g, '')
    .replace(/show\s+me\s+(my\s+|the\s+)?/g, '')
    .replace(/i\s+need\s+(a\s+|my\s+)?/g, '')
    .replace(/look\s+for\s+(my\s+|the\s+|a\s+)?/g, '')
    .trim();
}

export function isiOS() {
  return typeof navigator !== 'undefined' &&
    /iPad|iPhone|iPod/i.test(navigator.userAgent || '');
}

/**
 * @param {{onResult:(text:string)=>void, onError:(err:string)=>void, onEnd?:()=>void}} cb
 * @returns {SpeechRecognition|null}
 */
export function startListening(cb) {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    cb.onError('not-supported');
    return null;
  }
  const r = new SR();
  r.continuous = false;
  r.interimResults = false;
  r.lang = 'en-US';
  r.onresult = e => cb.onResult(e.results[0][0].transcript.trim());
  r.onerror = e => cb.onError(e?.error || 'unknown');
  r.onend = () => cb.onEnd?.();
  r.start();
  return r;
}
