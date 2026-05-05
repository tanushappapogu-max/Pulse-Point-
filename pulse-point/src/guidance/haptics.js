// Haptic vocabulary. Each pattern is an array of ms durations alternating
// vibration / pause. The vocabulary is designed to be discriminable by feel:
//   - "looking" = single soft pulse
//   - "found"   = three rising pulses (good news)
//   - "left"    = two short pulses (binary feel — left taps once on each side)
//   - "right"   = three short pulses (one extra, biased to right hand)
//   - "up"      = short-then-long (rising)
//   - "down"    = long-then-short (falling)
//   - "closer"  = quick triplet (urgency)
//   - "locked"  = single long (commitment)
//   - "reach"   = two long pulses (decisive arrival)
//   - "lost"    = sharp double-tap then silence

export const HAPTIC_PATTERNS = {
  looking: [24, 260],
  found:   [170, 80, 170, 80, 260],
  left:    [80, 45, 80],
  right:   [80, 45, 80, 45, 80],
  up:      [40, 50, 110],
  down:    [110, 50, 40],
  closer:  [45, 38, 45, 38, 45],
  locked:  [260],
  reach:   [360, 80, 360],
  lost:    [35, 120, 35],
  confirm: [200, 80, 200],
};

const SIGNAL_GAP_MS = {
  looking: 1150,    // continuous-search signal — slow cadence
  default: 520,
};

export class Haptics {
  constructor() {
    this.lastTime = 0;
    this.lastSignal = '';
    this.enabled = typeof navigator !== 'undefined' && 'vibrate' in navigator;
  }

  setEnabled(on) {
    this.enabled = on && typeof navigator !== 'undefined' && 'vibrate' in navigator;
    if (!on) this.cancel();
  }

  isAvailable() {
    return typeof navigator !== 'undefined' && 'vibrate' in navigator;
  }

  cancel() {
    if (typeof navigator !== 'undefined' && navigator.vibrate) {
      navigator.vibrate(0);
    }
  }

  /**
   * Trigger a named haptic signal.
   * @param {string} signal
   * @param {boolean} immediate - bypass debounce
   */
  fire(signal, immediate = false) {
    if (!this.enabled) return;
    const pattern = HAPTIC_PATTERNS[signal] || HAPTIC_PATTERNS.looking;
    const now = Date.now();
    const gap = SIGNAL_GAP_MS[signal] || SIGNAL_GAP_MS.default;
    if (!immediate && now - this.lastTime < gap) return;
    if (!immediate && signal === this.lastSignal && signal === 'found') return;
    this.lastTime = now;
    this.lastSignal = signal;
    navigator.vibrate(pattern);
  }
}
