/**
 * directionHaptics.js — Directional Haptic Guidance Engine
 *
 * Single-motor phones can't produce true directional vibration, but temporal
 * pattern encoding creates a reliable directional feel via cognitive mapping.
 * This approach is used in professional accessibility haptic systems.
 *
 * DIRECTION ENCODING
 * ──────────────────
 * LEFT    [heavy · pause · light]   asymmetric, front-heavy = "pulled left"
 * RIGHT   [light · pause · heavy]   asymmetric, back-heavy  = "push right"
 * UP      [light → medium → heavy]  ascending energy        = "go up"
 * DOWN    [heavy → medium → light]  descending energy       = "go down"
 * LOCKED  [medium · short · medium] symmetric               = "on target"
 * REACH   rapid 80 ms bursts        urgent                  = "stop + reach"
 *
 * PROXIMITY SCALING
 * ─────────────────
 * Direction patterns repeat on an interval that shrinks as the object
 * gets closer. The closer you are, the faster it buzzes.
 *
 *   bbox area   est. distance   repeat interval
 *   < 0.010     > 4 m           2 000 ms
 *   0.010–0.040 2–4 m           1 100 ms
 *   0.040–0.100 1–2 m             600 ms
 *   0.100–0.220 0.5–1 m           280 ms
 *   > 0.220     < 0.5 m            90 ms  (rapid fire)
 *
 * DIRECTION DETECTION
 * ───────────────────
 * Derived from the CNN bounding-box position in the camera frame.
 * No ARKit, BLE, or sensor fusion needed — the bbox already gives
 * sub-5° angular accuracy at the distances where guidance matters.
 *
 *   bbox center X < 0.38  →  LEFT
 *   bbox center X > 0.62  →  RIGHT
 *   bbox center Y < 0.35  →  UP   (horizontal zone must be clear)
 *   bbox center Y > 0.65  →  DOWN
 *   else                  →  LOCKED
 */

import * as Haptics from 'expo-haptics';

// ── Public direction / proximity constants ────────────────────────────────────

export const Direction = Object.freeze({
  LEFT:   'left',
  RIGHT:  'right',
  UP:     'up',
  DOWN:   'down',
  LOCKED: 'locked',   // centered horizontally and vertically
  REACH:  'reach',    // extremely close — stop and reach
});

export const Proximity = Object.freeze({
  FAR:       'far',
  MEDIUM:    'medium',
  CLOSE:     'close',
  NEAR:      'near',
  REACH:     'reach',
});

// ── Haptic pattern library ────────────────────────────────────────────────────
// Each entry: array of { style, duration } OR null for gaps
// 'style' maps to expo-haptics ImpactFeedbackStyle

const S = Haptics.ImpactFeedbackStyle;  // shorthand

const PATTERNS = {
  [Direction.LEFT]: [
    { style: S.Heavy,  ms: 0  },
    { gap: 85 },
    { style: S.Light,  ms: 0  },
  ],
  [Direction.RIGHT]: [
    { style: S.Light,  ms: 0  },
    { gap: 85 },
    { style: S.Heavy,  ms: 0  },
  ],
  [Direction.UP]: [
    { style: S.Light,  ms: 0  },
    { gap: 60 },
    { style: S.Medium, ms: 0  },
    { gap: 60 },
    { style: S.Heavy,  ms: 0  },
  ],
  [Direction.DOWN]: [
    { style: S.Heavy,  ms: 0  },
    { gap: 60 },
    { style: S.Medium, ms: 0  },
    { gap: 60 },
    { style: S.Light,  ms: 0  },
  ],
  [Direction.LOCKED]: [
    { style: S.Medium, ms: 0  },
    { gap: 55 },
    { style: S.Medium, ms: 0  },
  ],
  [Direction.REACH]: [
    { style: S.Heavy,  ms: 0  },
    { gap: 55 },
    { style: S.Heavy,  ms: 0  },
    { gap: 55 },
    { style: S.Heavy,  ms: 0  },
  ],
};

// ── Proximity → repeat interval ───────────────────────────────────────────────

const PROXIMITY_INTERVALS = {
  [Proximity.FAR]:    2000,
  [Proximity.MEDIUM]: 1100,
  [Proximity.CLOSE]:   600,
  [Proximity.NEAR]:    280,
  [Proximity.REACH]:    90,
};

// Dead-zone widths — fraction of frame width/height
const H_DEAD = 0.24;   // ±12% from center horizontally
const V_DEAD = 0.30;   // ±15% from center vertically

// ── Detection analysis ────────────────────────────────────────────────────────

/**
 * Given a normalised bounding box {x, y, width, height}, return the
 * direction and proximity the user needs to act on.
 *
 * @param {{ x: number, y: number, width: number, height: number }} bbox
 * @returns {{ direction: string, proximity: string, distanceM: number, cx: number, cy: number }}
 */
export function analyzeDetection(bbox) {
  const cx   = bbox.x + bbox.width  / 2;
  const cy   = bbox.y + bbox.height / 2;
  const area = bbox.width * bbox.height;

  // ── Proximity ─────────────────────────────────────────────────────────
  let proximity;
  let distanceM;
  if      (area > 0.22)  { proximity = Proximity.REACH;  distanceM = 0.35; }
  else if (area > 0.10)  { proximity = Proximity.NEAR;   distanceM = 0.75; }
  else if (area > 0.04)  { proximity = Proximity.CLOSE;  distanceM = 1.50; }
  else if (area > 0.010) { proximity = Proximity.MEDIUM; distanceM = 3.00; }
  else                   { proximity = Proximity.FAR;    distanceM = 5.00; }

  // ── Direction ─────────────────────────────────────────────────────────
  let direction;
  if (proximity === Proximity.REACH) {
    direction = Direction.REACH;
  } else {
    const hOff = cx - 0.5;   // negative = left, positive = right
    const vOff = cy - 0.5;   // negative = up,   positive = down

    if (Math.abs(hOff) >= H_DEAD / 2) {
      direction = hOff < 0 ? Direction.LEFT : Direction.RIGHT;
    } else if (Math.abs(vOff) >= V_DEAD / 2) {
      direction = vOff < 0 ? Direction.UP : Direction.DOWN;
    } else {
      direction = Direction.LOCKED;
    }
  }

  return { direction, proximity, distanceM, cx, cy };
}

// ── DirectionHapticEngine ─────────────────────────────────────────────────────

/**
 * Continuous directional haptic guidance.
 *
 * Usage:
 *   const engine = new DirectionHapticEngine();
 *
 *   // Every time a new detection arrives:
 *   engine.update(detection.boundingBox);
 *
 *   // When done navigating:
 *   engine.stop();
 */
export class DirectionHapticEngine {
  constructor() {
    this._active    = false;
    this._running   = false;   // true while _loop() is executing
    this._direction = Direction.LOCKED;
    this._proximity = Proximity.FAR;
    this._stopFlag  = false;
    this._onChange  = null;    // optional callback(direction, proximity, distanceM)
  }

  /**
   * Subscribe to direction/proximity changes for UI updates.
   * @param {(direction: string, proximity: string, distanceM: number) => void} fn
   */
  onUpdate(fn) {
    this._onChange = fn;
    return this;
  }

  /**
   * Feed a new bounding box into the engine.
   * On first call, starts the haptic loop; subsequent calls update direction.
   *
   * @param {{ x: number, y: number, width: number, height: number }} bbox
   */
  update(bbox) {
    const { direction, proximity, distanceM } = analyzeDetection(bbox);

    const changed = direction !== this._direction || proximity !== this._proximity;
    this._direction = direction;
    this._proximity = proximity;

    if (changed && this._onChange) {
      this._onChange(direction, proximity, distanceM);
    }

    if (!this._active) {
      this._active   = true;
      this._stopFlag = false;
      this._loop();
    }
  }

  /** Stop guidance and silence haptics. */
  stop() {
    this._stopFlag = true;
    this._active   = false;
  }

  /** True while guidance is running. */
  get isActive() {
    return this._active;
  }

  // ── Private ─────────────────────────────────────────────────────────────

  async _loop() {
    if (this._running) return;
    this._running = true;

    while (!this._stopFlag) {
      const dir  = this._direction;
      const prox = this._proximity;

      await this._playPattern(dir);
      if (this._stopFlag) break;

      // Wait out the proximity-based interval before next burst.
      // During the wait, direction/proximity may be updated by _update().
      const interval = PROXIMITY_INTERVALS[prox];
      await _sleep(interval);
    }

    this._running = false;
  }

  async _playPattern(dir) {
    const steps = PATTERNS[dir] ?? PATTERNS[Direction.LOCKED];
    for (const step of steps) {
      if (this._stopFlag) return;
      if (step.gap) {
        await _sleep(step.gap);
      } else {
        await Haptics.impactAsync(step.style);
      }
    }
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function _sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * One-shot haptic for state transitions (found / complete / error).
 */
export async function playTransitionHaptic(type) {
  // type: 'found' | 'complete' | 'lost'
  if (type === 'found') {
    await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    await _sleep(110);
    await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  } else if (type === 'complete') {
    await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    await _sleep(90);
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
    await _sleep(90);
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
  } else if (type === 'lost') {
    await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
  }
}

/**
 * Human-readable label for a direction value.
 */
export function directionLabel(direction) {
  return {
    [Direction.LEFT]:   '← Left',
    [Direction.RIGHT]:  '→ Right',
    [Direction.UP]:     '↑ Up',
    [Direction.DOWN]:   '↓ Down',
    [Direction.LOCKED]: '● Locked',
    [Direction.REACH]:  '✋ Reach',
  }[direction] ?? '';
}

/**
 * Arrow character for compact UI display.
 */
export function directionArrow(direction) {
  return {
    [Direction.LEFT]:   '←',
    [Direction.RIGHT]:  '→',
    [Direction.UP]:     '↑',
    [Direction.DOWN]:   '↓',
    [Direction.LOCKED]: '●',
    [Direction.REACH]:  '✋',
  }[direction] ?? '?';
}
