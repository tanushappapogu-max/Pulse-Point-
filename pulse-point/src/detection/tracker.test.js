import { describe, test, expect, beforeEach } from 'vitest';
import { BoxTracker } from './tracker.js';

let tracker;

beforeEach(() => {
  tracker = new BoxTracker();
});

describe('BoxTracker — initial state', () => {
  test('box is null before any update', () => {
    expect(tracker.box).toBeNull();
  });

  test('predict returns null before any update', () => {
    expect(tracker.predict(Date.now())).toBeNull();
  });

  test('isAlive returns false before any update', () => {
    expect(tracker.isAlive(Date.now())).toBe(false);
  });

  test('isFresh returns false before any update', () => {
    expect(tracker.isFresh()).toBe(false);
  });
});

describe('BoxTracker — single update', () => {
  test('sets box and marks fresh', () => {
    const now = 1000;
    tracker.update([100, 100, 50, 50], 0.9, 'laptop', now);
    expect(tracker.box).not.toBeNull();
    expect(tracker.isFresh()).toBe(true);
    expect(tracker.label).toBe('laptop');
    expect(tracker.confidence).toBe(0.9);
  });

  test('isAlive is true right after update', () => {
    const now = Date.now();
    tracker.update([100, 100, 50, 50], 0.9, 'laptop', now);
    expect(tracker.isAlive(now)).toBe(true);
  });

  test('isAlive is false after STALE_AFTER_MS (600 ms)', () => {
    const t0 = 1000;
    tracker.update([100, 100, 50, 50], 0.9, 'laptop', t0);
    expect(tracker.isAlive(t0 + 700)).toBe(false);
  });

  test('predict returns null after stale window', () => {
    const t0 = 1000;
    tracker.update([100, 100, 50, 50], 0.9, 'laptop', t0);
    expect(tracker.predict(t0 + 700)).toBeNull();
  });

  test('predict returns a box within the stale window', () => {
    const t0 = 1000;
    tracker.update([100, 100, 50, 50], 0.9, 'laptop', t0);
    const pred = tracker.predict(t0 + 100);
    expect(pred).not.toBeNull();
    expect(pred.bbox).toHaveLength(4);
    expect(pred.label).toBe('laptop');
  });
});

describe('BoxTracker — EMA smoothing on overlapping updates', () => {
  test('second overlapping update smooths position (not a hard snap)', () => {
    const t0 = 1000;
    tracker.update([100, 100, 50, 50], 0.9, 'cup', t0);
    const rawBox = [...tracker.box];

    // Second update 100 ms later, box shifted by 10 px — high IoU so EMA applies
    tracker.update([110, 110, 50, 50], 0.9, 'cup', t0 + 100);
    const smoothed = tracker.box;

    // Smoothed value should be between the two extremes (not exactly at new position)
    expect(smoothed[0]).toBeGreaterThan(rawBox[0]);
    expect(smoothed[0]).toBeLessThan(110);
  });

  test('velocity is estimated after two close updates', () => {
    const t0 = 1000;
    tracker.update([100, 100, 50, 50], 0.9, 'cup', t0);
    tracker.update([110, 110, 50, 50], 0.9, 'cup', t0 + 100);
    // Some non-zero velocity should have been computed
    const [vx, vy] = tracker.velocity;
    expect(Math.abs(vx) + Math.abs(vy)).toBeGreaterThan(0);
  });
});

describe('BoxTracker — non-overlapping update resets velocity', () => {
  test('low-IoU jump resets velocity to zero', () => {
    const t0 = 1000;
    tracker.update([100, 100, 50, 50], 0.9, 'cup', t0);
    // Jump to a completely different location (no overlap)
    tracker.update([500, 400, 50, 50], 0.9, 'cup', t0 + 100);
    expect(tracker.velocity[0]).toBe(0);
    expect(tracker.velocity[1]).toBe(0);
  });
});

describe('BoxTracker — fresh / stale lifecycle', () => {
  test('markStale clears fresh flag', () => {
    tracker.update([100, 100, 50, 50], 0.9, 'cup', 1000);
    expect(tracker.isFresh()).toBe(true);
    tracker.markStale();
    expect(tracker.isFresh()).toBe(false);
  });

  test('subsequent update re-marks fresh', () => {
    tracker.update([100, 100, 50, 50], 0.9, 'cup', 1000);
    tracker.markStale();
    tracker.update([105, 105, 50, 50], 0.9, 'cup', 1100);
    expect(tracker.isFresh()).toBe(true);
  });
});

describe('BoxTracker — reset', () => {
  test('reset clears all state', () => {
    tracker.update([100, 100, 50, 50], 0.9, 'cup', 1000);
    tracker.reset();
    expect(tracker.box).toBeNull();
    expect(tracker.isFresh()).toBe(false);
    expect(tracker.isAlive(Date.now())).toBe(false);
  });
});
