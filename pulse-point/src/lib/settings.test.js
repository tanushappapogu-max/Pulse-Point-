import { describe, test, expect, beforeEach } from 'vitest';
import { loadSettings, saveSettings, updateSetting, DEFAULT_SETTINGS } from './settings.js';

beforeEach(() => {
  localStorage.clear();
});

describe('loadSettings', () => {
  test('returns a copy of DEFAULT_SETTINGS when nothing is stored', () => {
    const s = loadSettings();
    expect(s).toEqual(DEFAULT_SETTINGS);
  });

  test('does not return the same object reference as DEFAULT_SETTINGS', () => {
    const s = loadSettings();
    s.speech = !s.speech;
    expect(loadSettings().speech).toBe(DEFAULT_SETTINGS.speech);
  });

  test('merges stored values over defaults', () => {
    localStorage.setItem('pulsepoint:settings:v1', JSON.stringify({ speech: true }));
    const s = loadSettings();
    expect(s.speech).toBe(true);
    expect(s.haptics).toBe(DEFAULT_SETTINGS.haptics);
  });

  test('handles malformed JSON gracefully — returns defaults', () => {
    localStorage.setItem('pulsepoint:settings:v1', 'not-valid-json}}');
    const s = loadSettings();
    expect(s).toEqual(DEFAULT_SETTINGS);
  });

  test('unknown keys in stored JSON are ignored (defaults fill gaps)', () => {
    localStorage.setItem(
      'pulsepoint:settings:v1',
      JSON.stringify({ unknownKey: 42, haptics: false }),
    );
    const s = loadSettings();
    expect(s.haptics).toBe(false);
    expect(s.speech).toBe(DEFAULT_SETTINGS.speech);
  });
});

describe('saveSettings + loadSettings roundtrip', () => {
  test('persists boolean change', () => {
    saveSettings({ ...DEFAULT_SETTINGS, speech: true });
    expect(loadSettings().speech).toBe(true);
  });

  test('persists numeric change', () => {
    saveSettings({ ...DEFAULT_SETTINGS, speechRate: 1.5 });
    expect(loadSettings().speechRate).toBe(1.5);
  });

  test('persists string change', () => {
    saveSettings({ ...DEFAULT_SETTINGS, sensitivity: 'sharp' });
    expect(loadSettings().sensitivity).toBe('sharp');
  });
});

describe('updateSetting', () => {
  test('returns new object with changed key', () => {
    const next = updateSetting(DEFAULT_SETTINGS, 'speech', true);
    expect(next.speech).toBe(true);
  });

  test('does not mutate the input object', () => {
    const original = { ...DEFAULT_SETTINGS };
    updateSetting(original, 'speech', true);
    expect(original.speech).toBe(DEFAULT_SETTINGS.speech);
  });

  test('leaves other keys unchanged', () => {
    const next = updateSetting(DEFAULT_SETTINGS, 'speech', true);
    expect(next.haptics).toBe(DEFAULT_SETTINGS.haptics);
    expect(next.sensitivity).toBe(DEFAULT_SETTINGS.sensitivity);
  });

  test('persists the change to localStorage', () => {
    updateSetting(DEFAULT_SETTINGS, 'haptics', false);
    expect(loadSettings().haptics).toBe(false);
  });
});
