import { describe, expect, it } from 'vitest';

import {
  DEFAULT_RATIOS,
  deduplicate,
  splitDataset,
  splitFor,
  summarize,
} from '../src/dataset.js';
import type { PreferencePair } from '../src/types.js';

const pair = (over: Partial<PreferencePair> = {}): PreferencePair => ({
  id: 'id',
  prompt: 'p',
  chosen: 'c',
  rejected: 'r',
  annotatorId: 'ann-a',
  ...over,
});

/** `count` prompts, each with `perPrompt` pairs from different annotators. */
const corpus = (count: number, perPrompt = 1): PreferencePair[] =>
  Array.from({ length: count }, (_, i) =>
    Array.from({ length: perPrompt }, (_, j) =>
      pair({ id: `${i}-${j}`, prompt: `prompt-${i}`, annotatorId: `ann-${j}` }),
    ),
  ).flat();

describe('splitDataset', () => {
  it('keeps every pair', () => {
    const pairs = corpus(300);
    const { train, validation, test } = splitDataset(pairs);
    expect(train.length + validation.length + test.length).toBe(pairs.length);
  });

  it('never lets one prompt straddle two splits', () => {
    // The whole point: a prompt in both train and test inflates eval scores.
    const { train, validation, test } = splitDataset(corpus(400, 3));

    const where = new Map<string, string>();
    for (const [name, bucket] of [
      ['train', train],
      ['validation', validation],
      ['test', test],
    ] as const) {
      for (const p of bucket) {
        const existing = where.get(p.prompt);
        expect(existing ?? name).toBe(name);
        where.set(p.prompt, name);
      }
    }
  });

  it('is deterministic across runs', () => {
    const pairs = corpus(200);
    const a = splitDataset(pairs);
    const b = splitDataset(pairs);
    expect(a.train.map((p) => p.id)).toEqual(b.train.map((p) => p.id));
    expect(a.test.map((p) => p.id)).toEqual(b.test.map((p) => p.id));
  });

  it('does not depend on input order', () => {
    const pairs = corpus(200);
    const forwards = splitDataset(pairs);
    const backwards = splitDataset([...pairs].reverse());

    expect(new Set(forwards.test.map((p) => p.id))).toEqual(
      new Set(backwards.test.map((p) => p.id)),
    );
  });

  it('does not reshuffle existing prompts when new ones are appended', () => {
    // Appending data must not invalidate an earlier train/test boundary.
    const original = corpus(200);
    const before = splitDataset(original);
    const after = splitDataset([
      ...original,
      ...corpus(50).map((p) =>
        pair({
          ...p,
          id: `new-${p.id}`,
          prompt: `new-${p.prompt}`,
        }),
      ),
    ]);

    const originalIds = new Set(original.map((p) => p.id));
    expect(
      after.test
        .filter((p) => originalIds.has(p.id))
        .map((p) => p.id)
        .sort(),
    ).toEqual(before.test.map((p) => p.id).sort());
  });

  it('lands roughly on the requested proportions', () => {
    const { train, validation, test } = splitDataset(corpus(3000));
    expect(train.length / 3000).toBeCloseTo(0.8, 1);
    expect(validation.length / 3000).toBeCloseTo(0.1, 1);
    expect(test.length / 3000).toBeCloseTo(0.1, 1);
  });

  it('honours custom ratios', () => {
    const { train, test } = splitDataset(corpus(2000), {
      train: 0.5,
      validation: 0,
      test: 0.5,
    });
    expect(train.length / 2000).toBeCloseTo(0.5, 1);
    expect(test.length / 2000).toBeCloseTo(0.5, 1);
  });

  it('produces a different split for a different seed', () => {
    const pairs = corpus(400);
    const a = splitDataset(pairs, DEFAULT_RATIOS, 'seed-a');
    const b = splitDataset(pairs, DEFAULT_RATIOS, 'seed-b');
    expect(a.test.map((p) => p.id)).not.toEqual(b.test.map((p) => p.id));
  });

  it.each([
    [{ train: 0.8, validation: 0.1, test: 0.2 }, 'sum to 1'],
    [{ train: -0.1, validation: 0.5, test: 0.6 }, 'non-negative'],
  ])('rejects invalid ratios %#', (ratios, message) => {
    expect(() => splitDataset(corpus(1), ratios)).toThrow(new RegExp(message));
  });

  it('handles an empty dataset', () => {
    expect(splitDataset([])).toEqual({ train: [], validation: [], test: [] });
  });
});

describe('splitFor', () => {
  it('agrees with splitDataset', () => {
    const pairs = corpus(200);
    const { test } = splitDataset(pairs);
    for (const p of test) {
      expect(splitFor(p.prompt)).toBe('test');
    }
  });
});

describe('deduplicate', () => {
  it('keeps the first occurrence of each id', () => {
    const deduped = deduplicate([
      pair({ id: 'a', prompt: 'first' }),
      pair({ id: 'a', prompt: 'second' }),
      pair({ id: 'b' }),
    ]);
    expect(deduped).toHaveLength(2);
    expect(deduped[0]?.prompt).toBe('first');
  });

  it('leaves a clean dataset untouched', () => {
    const pairs = corpus(10).map((p, i) => pair({ ...p, id: `u${i}` }));
    expect(deduplicate(pairs)).toHaveLength(10);
  });
});

describe('summarize', () => {
  it('counts distinct prompts and annotators', () => {
    const summary = summarize(corpus(10, 3));
    expect(summary.pairs).toBe(30);
    expect(summary.prompts).toBe(10);
    expect(summary.annotators).toBe(3);
    expect(summary.pairsPerPrompt).toBe(3);
  });

  it('averages confidence over only the pairs that reported it', () => {
    const summary = summarize([
      pair({ id: '1', confidence: 1 }),
      pair({ id: '2', confidence: 0 }),
      pair({ id: '3' }),
    ]);
    expect(summary.meanConfidence).toBe(0.5);
  });

  it('reports null confidence when nothing recorded it', () => {
    expect(summarize([pair()]).meanConfidence).toBeNull();
  });

  it('handles an empty dataset', () => {
    expect(summarize([])).toMatchObject({ pairs: 0, prompts: 0, pairsPerPrompt: 0 });
  });
});
