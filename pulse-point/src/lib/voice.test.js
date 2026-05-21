import { describe, test, expect } from 'vitest';
import { isVagueIntent, extractTarget } from './voice.js';

describe('isVagueIntent', () => {
  test('returns false for specific object targets', () => {
    expect(isVagueIntent('find my phone')).toBe(false);
    expect(isVagueIntent('where is the laptop')).toBe(false);
    expect(isVagueIntent('show me the cup')).toBe(false);
  });

  test('"something" or "anything" is vague', () => {
    expect(isVagueIntent('find something')).toBe(true);
    expect(isVagueIntent('look for anything nearby')).toBe(true);
  });

  test('food-related utterances are vague', () => {
    expect(isVagueIntent('I want food')).toBe(true);
    expect(isVagueIntent('I am hungry')).toBe(true);
    expect(isVagueIntent('find a snack')).toBe(true);
    expect(isVagueIntent('I need a drink')).toBe(true);
  });

  test('writing actions are vague', () => {
    expect(isVagueIntent('I want to write')).toBe(true);
    expect(isVagueIntent('help me draw')).toBe(true);
  });

  test('communication actions are vague', () => {
    expect(isVagueIntent('I need to call someone')).toBe(true);
    expect(isVagueIntent('send a text')).toBe(true);
  });

  test('empty string returns false', () => {
    expect(isVagueIntent('')).toBe(false);
  });
});

describe('extractTarget', () => {
  test('"where is my phone" → "phone"', () => {
    expect(extractTarget('where is my phone')).toBe('phone');
  });

  test('"where is the laptop" → "laptop"', () => {
    expect(extractTarget('where is the laptop')).toBe('laptop');
  });

  test('"find my keys" → "keys"', () => {
    expect(extractTarget('find my keys')).toBe('keys');
  });

  test('"find the bottle" → "bottle"', () => {
    expect(extractTarget('find the bottle')).toBe('bottle');
  });

  test('"find a chair" → "chair"', () => {
    expect(extractTarget('find a chair')).toBe('chair');
  });

  test('"show me the remote" → "remote"', () => {
    expect(extractTarget('show me the remote')).toBe('remote');
  });

  test('"show me my phone" → "phone"', () => {
    expect(extractTarget('show me my phone')).toBe('phone');
  });

  test('"i need a cup" → "cup"', () => {
    expect(extractTarget('i need a cup')).toBe('cup');
  });

  test('"look for a bottle" → "bottle"', () => {
    expect(extractTarget('look for a bottle')).toBe('bottle');
  });

  test('"look for my backpack" → "backpack"', () => {
    expect(extractTarget('look for my backpack')).toBe('backpack');
  });

  test('handles null/empty gracefully', () => {
    expect(extractTarget(null)).toBe('');
    expect(extractTarget('')).toBe('');
  });

  test('bare noun passes through unchanged', () => {
    expect(extractTarget('umbrella')).toBe('umbrella');
  });
});
