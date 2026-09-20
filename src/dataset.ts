/**
 * Splitting and deduplication.
 *
 * The one rule that matters: **never let the same prompt appear in more than
 * one split.** Reward models memorise prompts, so a prompt straddling train
 * and test inflates eval scores and hides the problem until production.
 */

import { createHash } from 'node:crypto';

import type { PreferencePair, Split, SplitRatios } from './types.js';

export const DEFAULT_RATIOS: SplitRatios = { train: 0.8, validation: 0.1, test: 0.1 };

export interface SplitDataset {
  readonly train: readonly PreferencePair[];
  readonly validation: readonly PreferencePair[];
  readonly test: readonly PreferencePair[];
}

/** Map a string onto [0, 1) deterministically, for stable hash-based bucketing. */
function hashToUnitInterval(value: string, seed: string): number {
  const digest = createHash('sha256')
    .update(seed)
    .update('\u0000')
    .update(value)
    .digest();
  // Top 6 bytes give 48 bits of resolution — far more than enough for bucketing,
  // and safely inside Number.MAX_SAFE_INTEGER.
  const int48 = digest.readUIntBE(0, 6);
  return int48 / 2 ** 48;
}

/**
 * Split pairs into train/validation/test, grouped by prompt.
 *
 * Assignment is a deterministic function of the prompt and the seed, not of
 * input order or a random number generator. Two consequences that matter in
 * practice: re-running on a regenerated export yields the same split, and
 * appending new prompts never reshuffles existing ones.
 *
 * @throws RangeError if the ratios are negative or do not sum to 1.
 */
export function splitDataset(
  pairs: readonly PreferencePair[],
  ratios: SplitRatios = DEFAULT_RATIOS,
  seed = 'preference-sdk',
): SplitDataset {
  const { train, validation, test } = ratios;

  if (train < 0 || validation < 0 || test < 0) {
    throw new RangeError('split ratios must be non-negative');
  }
  const total = train + validation + test;
  if (Math.abs(total - 1) > 1e-9) {
    throw new RangeError(`split ratios must sum to 1, got ${total}`);
  }

  const buckets: {
    train: PreferencePair[];
    validation: PreferencePair[];
    test: PreferencePair[];
  } = { train: [], validation: [], test: [] };

  for (const pair of pairs) {
    // Bucket on the prompt, so every pair sharing a prompt lands together.
    const position = hashToUnitInterval(pair.prompt, seed);
    const split: Split =
      position < train ? 'train' : position < train + validation ? 'validation' : 'test';
    buckets[split].push(pair);
  }

  return buckets;
}

/** Which split a prompt would land in, without materialising the dataset. */
export function splitFor(
  prompt: string,
  ratios: SplitRatios = DEFAULT_RATIOS,
  seed = 'preference-sdk',
): Split {
  const position = hashToUnitInterval(prompt, seed);
  if (position < ratios.train) return 'train';
  if (position < ratios.train + ratios.validation) return 'validation';
  return 'test';
}

/**
 * Drop pairs that repeat an earlier pair's id, keeping the first occurrence.
 *
 * Duplicates arise routinely when exports overlap; left in, they silently
 * upweight whatever they duplicate.
 */
export function deduplicate(pairs: readonly PreferencePair[]): readonly PreferencePair[] {
  const seen = new Set<string>();
  const out: PreferencePair[] = [];

  for (const pair of pairs) {
    if (seen.has(pair.id)) continue;
    seen.add(pair.id);
    out.push(pair);
  }

  return out;
}

/** Count of distinct prompts, annotators and pairs. */
export interface DatasetSummary {
  readonly pairs: number;
  readonly prompts: number;
  readonly annotators: number;
  readonly pairsPerPrompt: number;
  readonly meanConfidence: number | null;
}

export function summarize(pairs: readonly PreferencePair[]): DatasetSummary {
  const prompts = new Set<string>();
  const annotators = new Set<string>();
  let confidenceTotal = 0;
  let withConfidence = 0;

  for (const pair of pairs) {
    prompts.add(pair.prompt);
    annotators.add(pair.annotatorId);
    if (pair.confidence !== undefined) {
      confidenceTotal += pair.confidence;
      withConfidence += 1;
    }
  }

  return {
    pairs: pairs.length,
    prompts: prompts.size,
    annotators: annotators.size,
    pairsPerPrompt: prompts.size === 0 ? 0 : pairs.length / prompts.size,
    meanConfidence: withConfidence === 0 ? null : confidenceTotal / withConfidence,
  };
}
