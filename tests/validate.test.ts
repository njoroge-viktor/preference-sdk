import { describe, expect, it } from 'vitest';

import { PreferenceValidationError } from '../src/errors.js';
import type { DraftPair, PreferenceGroup } from '../src/types.js';
import {
  groupToPairs,
  pairId,
  validateGroups,
  validatePairs,
  validatePairsStrict,
} from '../src/validate.js';

const draft = (over: Partial<DraftPair> = {}): DraftPair => ({
  prompt: 'Explain event loops.',
  chosen: 'A detailed, correct explanation.',
  rejected: 'Dunno.',
  annotatorId: 'ann-a',
  ...over,
});

describe('pairId', () => {
  it('is deterministic across calls', () => {
    expect(pairId('p', 'c', 'r')).toBe(pairId('p', 'c', 'r'));
  });

  it('changes when any component changes', () => {
    const base = pairId('p', 'c', 'r');
    expect(pairId('p2', 'c', 'r')).not.toBe(base);
    expect(pairId('p', 'c2', 'r')).not.toBe(base);
    expect(pairId('p', 'c', 'r2')).not.toBe(base);
  });

  it('resists field-boundary collisions', () => {
    // Without a delimiter, ('ab','c') and ('a','bc') would hash identically.
    expect(pairId('ab', 'c', 'x')).not.toBe(pairId('a', 'bc', 'x'));
  });
});

describe('validatePairs', () => {
  it('accepts a well-formed pair and assigns an id', () => {
    const { valid, issues } = validatePairs([draft()]);
    expect(issues).toHaveLength(0);
    expect(valid[0]?.id).toHaveLength(16);
  });

  it('preserves a caller-supplied id', () => {
    const { valid } = validatePairs([draft({ id: 'custom-id' })]);
    expect(valid[0]?.id).toBe('custom-id');
  });

  it.each(['prompt', 'chosen', 'rejected', 'annotatorId'] as const)(
    'rejects an empty %s',
    (field) => {
      const { valid, issues } = validatePairs([draft({ [field]: '   ' })]);
      expect(valid).toHaveLength(0);
      expect(issues.some((i) => i.field === field && i.code === 'EMPTY_FIELD')).toBe(
        true,
      );
    },
  );

  it('rejects identical responses as carrying no signal', () => {
    const { valid, issues } = validatePairs([
      draft({ chosen: 'same', rejected: 'same' }),
    ]);
    expect(valid).toHaveLength(0);
    expect(issues[0]?.code).toBe('IDENTICAL_RESPONSES');
  });

  it.each([-0.1, 1.1, Number.NaN, Number.POSITIVE_INFINITY])(
    'rejects out-of-range confidence %s',
    (confidence) => {
      const { issues } = validatePairs([draft({ confidence })]);
      expect(issues[0]?.code).toBe('CONFIDENCE_OUT_OF_RANGE');
    },
  );

  it.each([0, 0.5, 1])('accepts boundary confidence %s', (confidence) => {
    const { issues } = validatePairs([draft({ confidence })]);
    expect(issues).toHaveLength(0);
  });

  it('drops the second of two identical pairs', () => {
    const { valid, issues } = validatePairs([draft(), draft()]);
    expect(valid).toHaveLength(1);
    expect(issues[0]?.code).toBe('DUPLICATE_ID');
  });

  it('keeps good records alongside bad ones', () => {
    const { valid, issues } = validatePairs([
      draft({ prompt: 'a' }),
      draft({ prompt: '' }),
      draft({ prompt: 'b' }),
    ]);
    expect(valid).toHaveLength(2);
    expect(issues).toHaveLength(1);
    expect(issues[0]?.index).toBe(1);
  });

  it('reports every problem in one record at once', () => {
    const { issues } = validatePairs([draft({ prompt: '', chosen: '', confidence: 5 })]);
    expect(issues.length).toBeGreaterThanOrEqual(3);
  });
});

describe('validatePairsStrict', () => {
  it('returns pairs when the batch is clean', () => {
    expect(validatePairsStrict([draft()])).toHaveLength(1);
  });

  it('throws carrying every issue', () => {
    try {
      validatePairsStrict([draft({ prompt: '' }), draft({ chosen: '' })]);
      expect.unreachable('should have thrown');
    } catch (error) {
      expect(error).toBeInstanceOf(PreferenceValidationError);
      expect((error as PreferenceValidationError).issues).toHaveLength(2);
    }
  });

  it('summarises large failure counts without dumping everything', () => {
    const bad = Array.from({ length: 12 }, () => draft({ prompt: '' }));
    expect(() => validatePairsStrict(bad)).toThrow(/and 7 more/);
  });
});

describe('validateGroups', () => {
  const group = (over: Partial<PreferenceGroup> = {}): PreferenceGroup => ({
    id: 'g1',
    prompt: 'Rank these.',
    annotatorId: 'ann-a',
    ranked: ['best', 'middle', 'worst'],
    ...over,
  });

  it('accepts a well-formed group', () => {
    expect(validateGroups([group()]).issues).toHaveLength(0);
  });

  it('rejects a group with fewer than two responses', () => {
    const { issues } = validateGroups([group({ ranked: ['only'] })]);
    expect(issues[0]?.code).toBe('GROUP_TOO_SMALL');
  });

  it('rejects duplicate responses as an ambiguous ranking', () => {
    const { issues } = validateGroups([group({ ranked: ['a', 'a'] })]);
    expect(issues[0]?.code).toBe('DUPLICATE_RESPONSE_IN_GROUP');
  });
});

describe('groupToPairs', () => {
  it('expands a ranking into every implied pair', () => {
    const pairs = groupToPairs({
      id: 'g1',
      prompt: 'p',
      annotatorId: 'ann-a',
      ranked: ['a', 'b', 'c'],
    });

    // n(n-1)/2 = 3 for a group of three.
    expect(pairs).toHaveLength(3);
    expect(pairs.map((p) => [p.chosen, p.rejected])).toEqual([
      ['a', 'b'],
      ['a', 'c'],
      ['b', 'c'],
    ]);
  });

  it('never inverts the ranking', () => {
    const ranked = ['1st', '2nd', '3rd', '4th'];
    const pairs = groupToPairs({ id: 'g', prompt: 'p', annotatorId: 'a', ranked });

    for (const pair of pairs) {
      expect(ranked.indexOf(pair.chosen)).toBeLessThan(ranked.indexOf(pair.rejected));
    }
  });

  it('records provenance when the group carried metadata', () => {
    const pairs = groupToPairs({
      id: 'g7',
      prompt: 'p',
      annotatorId: 'a',
      ranked: ['x', 'y'],
      metadata: { batch: 3 },
    });
    expect(pairs[0]?.metadata).toEqual({ batch: 3, sourceGroupId: 'g7' });
  });

  it('omits metadata entirely when the group had none', () => {
    const pairs = groupToPairs({
      id: 'g',
      prompt: 'p',
      annotatorId: 'a',
      ranked: ['x', 'y'],
    });
    expect(pairs[0]).not.toHaveProperty('metadata');
  });
});
