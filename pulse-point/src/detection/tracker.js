const VELOCITY_EMA = 0.55;
const STALE_AFTER_MS = 600;
const SNAP_IOU = 0.3;
const SMOOTH_ALPHA = 0.55;
const MAX_SPEED_PX_PER_MS = 4;

export class BoxTracker {
  constructor() {
    this.reset();
  }

  reset() {
    this.box = null;
    this.timestamp = 0;
    this.velocity = [0, 0];
    this.confidence = 0;
    this.label = null;
    this.fromAi = false;
    this.fresh = false;
  }

  update(box, confidence, label, now, fromAi = false) {
    if (!box) return;

    if (this.box && now > this.timestamp) {
      const dt = now - this.timestamp;
      if (dt > 0 && dt < STALE_AFTER_MS) {
        const overlap = iouLocal(this.box, box);
        if (overlap >= SNAP_IOU) {
          const a = SMOOTH_ALPHA;
          const blended = [
            a * box[0] + (1 - a) * this.box[0],
            a * box[1] + (1 - a) * this.box[1],
            a * box[2] + (1 - a) * this.box[2],
            a * box[3] + (1 - a) * this.box[3],
          ];
          const [px, py] = centerOf(this.box);
          const [cx, cy] = centerOf(blended);
          const rawVx = clamp((cx - px) / dt, -MAX_SPEED_PX_PER_MS, MAX_SPEED_PX_PER_MS);
          const rawVy = clamp((cy - py) / dt, -MAX_SPEED_PX_PER_MS, MAX_SPEED_PX_PER_MS);
          this.velocity = [
            VELOCITY_EMA * rawVx + (1 - VELOCITY_EMA) * this.velocity[0],
            VELOCITY_EMA * rawVy + (1 - VELOCITY_EMA) * this.velocity[1],
          ];
          this.box = blended;
        } else {
          this.box = [...box];
          this.velocity = [0, 0];
        }
      } else {
        this.box = [...box];
        this.velocity = [0, 0];
      }
    } else {
      this.box = [...box];
      this.velocity = [0, 0];
    }

    this.timestamp = now;
    this.confidence = confidence;
    this.label = label;
    this.fromAi = fromAi;
    this.fresh = true;
  }

  predict(now) {
    if (!this.box) return null;
    const dt = now - this.timestamp;
    if (dt > STALE_AFTER_MS) return null;
    const [vx, vy] = this.velocity;
    const [x, y, w, h] = this.box;
    return {
      bbox: [x + vx * dt, y + vy * dt, w, h],
      confidence: this.confidence * decayFactor(dt),
      label: this.label,
      fromAi: this.fromAi,
      ageMs: dt,
    };
  }

  markStale() {
    this.fresh = false;
  }

  isFresh() {
    return this.fresh;
  }

  isAlive(now) {
    return Boolean(this.box) && (now - this.timestamp) <= STALE_AFTER_MS;
  }
}

function centerOf(box) {
  const [x, y, w, h] = box;
  return [x + w / 2, y + h / 2];
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function decayFactor(dt) {
  return Math.max(0.6, 1 - (dt / STALE_AFTER_MS) * 0.4);
}

function iouLocal(a, b) {
  const [ax, ay, aw, ah] = a;
  const [bx, by, bw, bh] = b;
  const x1 = Math.max(ax, bx);
  const y1 = Math.max(ay, by);
  const x2 = Math.min(ax + aw, bx + bw);
  const y2 = Math.min(ay + ah, by + bh);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = aw * ah + bw * bh - inter;
  return union <= 0 ? 0 : inter / union;
}
