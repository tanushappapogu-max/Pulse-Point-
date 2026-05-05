// Tiny localStorage-backed settings store. Keep the schema flat and explicit;
// versioned key so we can migrate safely later.

const KEY = 'pulsepoint:settings:v1';

export const SENSITIVITY_LEVELS = ['gentle', 'medium', 'sharp'];

export const DEFAULT_SETTINGS = Object.freeze({
  haptics: true,
  speech: false,            // off by default — user opts in (autoplay-policy friendly)
  speechRate: 1.15,
  sensitivity: 'medium',    // affects haptic gap + closer threshold
  showAllBoxes: true,       // draw non-target predictions faintly
});

export function loadSettings() {
  try {
    const raw = typeof localStorage !== 'undefined' ? localStorage.getItem(KEY) : null;
    if (!raw) return { ...DEFAULT_SETTINGS };
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_SETTINGS, ...parsed };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(s) {
  try {
    if (typeof localStorage === 'undefined') return;
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    // private browsing / quota — silently ignore
  }
}

export function updateSetting(s, key, value) {
  const next = { ...s, [key]: value };
  saveSettings(next);
  return next;
}
