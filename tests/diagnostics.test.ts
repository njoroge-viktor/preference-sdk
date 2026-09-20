import { describe, expect, it } from 'vitest';

import {
  annotatorSkew,
  findConflicts,
  lengthBias,
  positionBias,
  signTest,
} from '../src/diagnostics.js';
import type { PreferencePair } from '../src/types.js';

const pair = (over: Partial<PreferencePair> = {}): PreferencePair => ({
  id: Math.random().toString(36).slice(2),
  prompt: 'p',
  chosen: 'chosen text',
  rejected: 'rejected',
  annotatorId: 'ann-a',
  ...over,
});

/** n pairs where the chosen response is longer than the rejected one. */
const longerChosen = (n: number, annotatorId = 'ann-a'): PreferencePair[] =>
  Array.from({ length: n }, (_, i) =>
    pair({
      id: `l${i}`,
      prompt: `p${i}`,
      chosen: 'x'.repeat(100),
      rejected: 'y',
      annotatorId,
    }),
  );

const shorterChosen = (n: number, annotatorId = 'ann-a'): PreferencePair[] =>
  Array.from({ length: n }, (_, i) =>
    pair({
      id: `s${i}`,
      prompt: `q${i}`,
      chosen: 'x',
      rejected: 'y'.repeat(100),
      annotatorId,
    }),
  );

describe('signTest', () => {
  it('reports no effect at exactly 50/50', () => {
    const result = signTest(50, 100);
    expect(result.rate).toBe(0.5);
    expect(result.z).toBe(0);
    expect(result.pValue).toBeCloseTo(1, 6);
    expect(result.significant).toBe(false);
  });

  it('flags a strong effect as significant', () => {
    const result = signTest(90, 100);
    expect(result.z).toBeCloseTo(8, 5);
    expect(result.pValue).toBeLessThan(0.001);
    expect(result.significant).toBe(true);
  });

  it('computes z from the binomial standard error', () => {
    // z = (60 - 50) / sqrt(100/4) = 10 / 5 = 2
    expect(signTest(60, 100).z).toBeCloseTo(2, 10);
  });

  it('two-sided p-value matches the normal tail', () => {
    // z = 2 -> two-sided p = 2 * (1 - Phi(2)) ~= 0.0455
    expect(signTest(60, 100).pValue).toBeCloseTo(0.0455, 3);
  });

  it('withholds significance below n = 30 however extreme', () => {
    const result = signTest(20, 20);
    expect(result.pValue).toBeLessThan(0.001);
    expect(result.significant).toBe(false);
  });

  it('handles an empty sample', () => {
    expect(signTest(0, 0)).toMatchObject({
      n: 0,
      rate: 0.5,
      pValue: 1,
      significant: false,
    });
  });

  it('rejects impossible counts', () => {
    expect(() => signTest(5, 3)).toThrow(RangeError);
    expect(() => signTest(-1, 3)).toThrow(RangeError);
  });
});

describe('lengthBias', () => {
  it('detects a systematic preference for longer responses', () => {
    const report = lengthBias(longerChosen(40));
    expect(report.rate).toBe(1);
    expect(report.significant).toBe(true);
    expect(report.meanDelta).toBeGreaterThan(0);
  });

  it('reports no bias on a balanced dataset', () => {
    const report = lengthBias([...longerChosen(20), ...shorterChosen(20)]);
    expect(report.rate).toBe(0.5);
    expect(report.significant).toBe(false);
    expect(report.meanDelta).toBeCloseTo(0, 10);
  });

  it('excludes equal-length pairs from the test but not the means', () => {
    const equal = Array.from({ length: 10 }, (_, i) =>
      pair({ id: `e${i}`, chosen: 'aaaa', rejected: 'bbbb' }),
    );
    const report = lengthBias([...longerChosen(5), ...equal]);
    // Only the 5 decisive pairs count toward n.
    expect(report.n).toBe(5);
    expect(report.meanChosenLength).toBeGreaterThan(0);
  });

  it('handles an empty dataset', () => {
    expect(lengthBias([])).toMatchObject({ n: 0, meanDelta: 0, significant: false });
  });

  it('reports the mean lengths it measured', () => {
    const report = lengthBias([
      pair({ chosen: 'x'.repeat(10), rejected: 'y'.repeat(4) }),
    ]);
    expect(report.meanChosenLength).toBe(10);
    expect(report.meanRejectedLength).toBe(4);
    expect(report.meanDelta).toBe(6);
  });
});

describe('positionBias', () => {
  it('detects a left-side preference', () => {
    const pairs = Array.from({ length: 40 }, (_, i) =>
      pair({ id: `p${i}`, chosenPosition: 'left' }),
    );
    const result = positionBias(pairs);
    expect(result.rate).toBe(1);
    expect(result.significant).toBe(true);
  });

  it('ignores pairs with no recorded position', () => {
    const result = positionBias([pair(), pair({ chosenPosition: 'left' })]);
    expect(result.n).toBe(1);
  });

  it('reports balance when positions were randomised', () => {
    const pairs = Array.from({ length: 40 }, (_, i) =>
      pair({ id: `p${i}`, chosenPosition: i % 2 === 0 ? 'left' : 'right' }),
    );
    expect(positionBias(pairs).significant).toBe(false);
  });
});

describe('annotatorSkew', () => {
  it('ranks the most length-biased annotator first', () => {
    const skew = annotatorSkew([
      ...longerChosen(10, 'verbose-fan'),
      ...shorterChosen(10, 'balanced'),
      ...longerChosen(10, 'balanced'),
    ]);

    expect(skew[0]?.annotatorId).toBe('verbose-fan');
    expect(skew[0]?.longerPreferredRate).toBe(1);
    expect(skew[1]?.longerPreferredRate).toBe(0.5);
  });

  it('counts only decisive pairs', () => {
    const skew = annotatorSkew([
      pair({ annotatorId: 'a', chosen: 'aaaa', rejected: 'bbbb' }),
      ...longerChosen(3, 'a'),
    ]);
    expect(skew[0]?.pairs).toBe(3);
  });

  it('returns nothing for an empty dataset', () => {
    expect(annotatorSkew([])).toHaveLength(0);
  });
});

describe('findConflicts', () => {
  it('finds two annotators who disagreed on the same pair', () => {
    const conflicts = findConflicts([
      pair({ id: '1', prompt: 'p', chosen: 'A', rejected: 'B', annotatorId: 'ann-a' }),
      pair({ id: '2', prompt: 'p', chosen: 'B', rejected: 'A', annotatorId: 'ann-b' }),
    ]);

    expect(conflicts).toHaveLength(1);
    expect(conflicts[0]?.preferredA).toEqual(['ann-a']);
    expect(conflicts[0]?.preferredB).toEqual(['ann-b']);
  });

  it('reports nothing when annotators agree', () => {
    const conflicts = findConflicts([
      pair({ id: '1', prompt: 'p', chosen: 'A', rejected: 'B', annotatorId: 'ann-a' }),
      pair({ id: '2', prompt: 'p', chosen: 'A', rejected: 'B', annotatorId: 'ann-b' }),
    ]);
    expect(conflicts).toHaveLength(0);
  });

  it('does not treat the same responses under different prompts as a conflict', () => {
    const conflicts = findConflicts([
      pair({ id: '1', prompt: 'p1', chosen: 'A', rejected: 'B', annotatorId: 'ann-a' }),
      pair({ id: '2', prompt: 'p2', chosen: 'B', rejected: 'A', annotatorId: 'ann-b' }),
    ]);
    expect(conflicts).toHaveLength(0);
  });

  it('groups every annotator on each side', () => {
    const conflicts = findConflicts([
      pair({ id: '1', prompt: 'p', chosen: 'A', rejected: 'B', annotatorId: 'ann-a' }),
      pair({ id: '2', prompt: 'p', chosen: 'A', rejected: 'B', annotatorId: 'ann-c' }),
      pair({ id: '3', prompt: 'p', chosen: 'B', rejected: 'A', annotatorId: 'ann-b' }),
    ]);

    expect(conflicts[0]?.preferredA).toEqual(['ann-a', 'ann-c']);
    expect(conflicts[0]?.preferredB).toEqual(['ann-b']);
  });
});
