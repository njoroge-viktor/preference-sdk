/**
 * Guards the public API surface. A barrel file that silently stops exporting
 * something breaks every consumer, and nothing else in the suite imports
 * through it.
 */

import { describe, expect, it } from 'vitest';

import * as sdk from '../src/index.js';

const EXPECTED_EXPORTS = [
  'DEFAULT_RATIOS',
  'PreferenceValidationError',
  'annotatorSkew',
  'deduplicate',
  'encodePair',
  'findConflicts',
  'groupToPairs',
  'lengthBias',
  'pairId',
  'positionBias',
  'signTest',
  'splitDataset',
  'splitFor',
  'streamJsonl',
  'summarize',
  'toJsonl',
  'validateGroups',
  'validatePairs',
  'validatePairsStrict',
] as const;

describe('public API', () => {
  it.each(EXPECTED_EXPORTS)('exports %s', (name) => {
    expect(sdk[name]).toBeDefined();
  });

  it('exports nothing unintended', () => {
    expect(Object.keys(sdk).sort()).toEqual([...EXPECTED_EXPORTS].sort());
  });

  it('supports the documented end-to-end flow', () => {
    const pairs = sdk.validatePairsStrict([
      {
        prompt: 'p1',
        chosen: 'a long chosen answer',
        rejected: 'no',
        annotatorId: 'ann-a',
      },
      {
        prompt: 'p2',
        chosen: 'short',
        rejected: 'a much longer rejected one',
        annotatorId: 'ann-a',
      },
    ]);

    const deduped = sdk.deduplicate(pairs);
    const split = sdk.splitDataset(deduped);
    const report = sdk.lengthBias(deduped);
    const jsonl = sdk.toJsonl(deduped, 'trl-dpo');

    expect(deduped).toHaveLength(2);
    expect(split.train.length + split.validation.length + split.test.length).toBe(2);
    expect(report.n).toBe(2);
    expect(jsonl.trimEnd().split('\n')).toHaveLength(2);
  });
});
