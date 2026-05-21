import { describe, test, expect } from 'vitest';
import { resolveCocoTarget, findClosestCocoLabel, COCO_LABELS } from './coco.js';

describe('resolveCocoTarget', () => {
  test('returns null for empty / null input', () => {
    expect(resolveCocoTarget(null)).toBeNull();
    expect(resolveCocoTarget('')).toBeNull();
  });

  test('exact COCO label passes through', () => {
    expect(resolveCocoTarget('cell phone')).toBe('cell phone');
    expect(resolveCocoTarget('laptop')).toBe('laptop');
    expect(resolveCocoTarget('person')).toBe('person');
  });

  test('alias: phone → cell phone', () => {
    expect(resolveCocoTarget('phone')).toBe('cell phone');
  });

  test('alias: iphone → cell phone', () => {
    expect(resolveCocoTarget('iphone')).toBe('cell phone');
  });

  test('alias: television → tv', () => {
    expect(resolveCocoTarget('television')).toBe('tv');
  });

  test('alias: sofa → couch', () => {
    expect(resolveCocoTarget('sofa')).toBe('couch');
  });

  test('alias: mug → cup', () => {
    expect(resolveCocoTarget('mug')).toBe('cup');
  });

  test('unknown label returns null', () => {
    expect(resolveCocoTarget('hovercraft')).toBeNull();
    expect(resolveCocoTarget('banana peel')).toBeNull();
  });

  test('case-insensitive: LAPTOP → laptop', () => {
    expect(resolveCocoTarget('LAPTOP')).toBe('laptop');
  });
});

describe('findClosestCocoLabel', () => {
  test('empty string → null', () => {
    expect(findClosestCocoLabel('')).toBeNull();
  });

  test('exact match → score 1', () => {
    const result = findClosestCocoLabel('laptop');
    expect(result).not.toBeNull();
    expect(result.label).toBe('laptop');
    expect(result.score).toBeCloseTo(1, 2);
  });

  test('one-character typo still resolves', () => {
    const result = findClosestCocoLabel('labtop');
    expect(result).not.toBeNull();
    expect(result.label).toBe('laptop');
  });

  test('completely unrelated word returns null (below threshold)', () => {
    const result = findClosestCocoLabel('xyzzy');
    expect(result).toBeNull();
  });

  test('all returned labels are valid COCO classes', () => {
    const words = ['phone', 'bike', 'dog', 'chair'];
    for (const w of words) {
      const result = findClosestCocoLabel(w);
      if (result) {
        expect(COCO_LABELS).toContain(result.label);
      }
    }
  });
});
