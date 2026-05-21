import { describe, test, expect } from 'vitest';
import {
  estimateDistanceMeters,
  describeDistanceMeters,
  describeDistanceArea,
  formatMeters,
  focalLengthPx,
} from './distance.js';

describe('focalLengthPx', () => {
  test('returns a positive number for a standard frame', () => {
    const f = focalLengthPx(640, 64);
    expect(f).toBeGreaterThan(0);
    // focal ≈ (640/2) / tan(32°) ≈ 512
    expect(f).toBeCloseTo(512, 0);
  });
});

describe('estimateDistanceMeters', () => {
  test('returns null for an unknown label', () => {
    expect(estimateDistanceMeters('unicorn', 100, 640)).toBeNull();
  });

  test('returns null when bbox width is zero', () => {
    expect(estimateDistanceMeters('laptop', 0, 640)).toBeNull();
  });

  test('estimates laptop at ~2 m when bbox is 90 px in 640-px frame', () => {
    // focal≈512, refWidth=35cm → dist=(35*512)/90≈199cm≈1.99m
    const d = estimateDistanceMeters('laptop', 90, 640);
    expect(d).not.toBeNull();
    expect(d).toBeCloseTo(1.99, 1);
  });

  test('close object (large bbox) gives small distance', () => {
    const d = estimateDistanceMeters('laptop', 512, 640);
    expect(d).not.toBeNull();
    expect(d).toBeLessThan(0.5); // under 50 cm
  });

  test('far object (tiny bbox) gives large distance', () => {
    const d = estimateDistanceMeters('laptop', 5, 640);
    expect(d).not.toBeNull();
    expect(d).toBeGreaterThan(20); // more than 20 m
  });

  test('works with aliased COCO labels', () => {
    expect(estimateDistanceMeters('cell phone', 60, 640)).not.toBeNull();
  });
});

describe('describeDistanceMeters', () => {
  test('null → null', () => {
    expect(describeDistanceMeters(null)).toBeNull();
  });

  test('< 0.4 m → within reach', () => {
    expect(describeDistanceMeters(0.3)).toBe('within reach');
  });

  test('0.4–0.9 m → arm\'s length', () => {
    expect(describeDistanceMeters(0.7)).toBe("arm's length");
  });

  test('0.9–1.6 m → one step away', () => {
    expect(describeDistanceMeters(1.2)).toBe('one step away');
  });

  test('1.6–3 m → a few steps away', () => {
    expect(describeDistanceMeters(2.5)).toBe('a few steps away');
  });

  test('3–6 m → across the room', () => {
    expect(describeDistanceMeters(4.5)).toBe('across the room');
  });

  test('>= 6 m → far', () => {
    expect(describeDistanceMeters(10)).toBe('far');
  });
});

describe('describeDistanceArea', () => {
  const FRAME = { width: 640, height: 480 };

  test('area > 0.24 → very close', () => {
    // 300*300 / 307200 ≈ 0.29
    expect(describeDistanceArea([0, 0, 300, 300], FRAME)).toBe('very close');
  });

  test('area 0.14–0.24 → close', () => {
    // 200*250 / 307200 ≈ 0.163
    expect(describeDistanceArea([0, 0, 200, 250], FRAME)).toBe('close');
  });

  test('area 0.07–0.14 → medium', () => {
    // 170*140 / 307200 ≈ 0.077
    expect(describeDistanceArea([0, 0, 170, 140], FRAME)).toBe('medium');
  });

  test('area < 0.07 → far', () => {
    // 100*100 / 307200 ≈ 0.033
    expect(describeDistanceArea([0, 0, 100, 100], FRAME)).toBe('far');
  });
});

describe('formatMeters', () => {
  test('null → null', () => {
    expect(formatMeters(null)).toBeNull();
  });

  test('sub-meter → cm string', () => {
    expect(formatMeters(0.75)).toBe('75 cm');
  });

  test('fractional sub-meter rounds correctly', () => {
    expect(formatMeters(0.553)).toBe('55 cm');
  });

  test('1–10 m → one-decimal string', () => {
    expect(formatMeters(2.547)).toBe('2.5 m');
  });

  test('>= 10 m → rounded integer string', () => {
    expect(formatMeters(15.3)).toBe('15 m');
  });
});
