/**
 * Dataset audits that catch the failure modes which quietly ruin a reward model.
 *
 * A preference dataset can be perfectly well-formed and still teach the wrong
 * thing. The classic case is **length bias**: if annotators systematically
 * prefer longer answers, the reward model learns "longer is better" and the
 * policy degenerates into padding. That is a data problem, and it is only
 * visible if you measure it before training.
 */

import type { PreferencePair } from './types.js';

/** Result of a two-sided sign test against a 50/50 null. */
export interface BiasTest {
  /** Pairs where the two sides differed on the measured attribute. */
  readonly n: number;
  /** How many of those went the measured way. */
  readonly successes: number;
  /** `successes / n`. Unbiased data sits at 0.5. */
  readonly rate: number;
  /** Normal-approximation z statistic against p = 0.5. */
  readonly z: number;
  /** Two-sided p-value. */
  readonly pValue: number;
  /** True when p < 0.05 and n is large enough for the approximation to hold. */
  readonly significant: boolean;
}

export interface LengthBiasReport extends BiasTest {
  readonly meanChosenLength: number;
  readonly meanRejectedLength: number;
  /** Positive means chosen responses are longer on average. */
  readonly meanDelta: number;
}

export interface AnnotatorSkew {
  readonly annotatorId: string;
  readonly pairs: number;
  /** How often this annotator picked the longer response. */
  readonly longerPreferredRate: number;
}

/** Two annotators who disagreed on the same ordered pair of responses. */
export interface Conflict {
  readonly prompt: string;
  readonly responseA: string;
  readonly responseB: string;
  /** Annotators who preferred A. */
  readonly preferredA: readonly string[];
  /** Annotators who preferred B. */
  readonly preferredB: readonly string[];
}

/** Abramowitz & Stegun 7.1.26: max absolute error 1.5e-7. */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);

  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t *
      Math.exp(-ax * ax);

  return sign * y;
}

/** Standard normal CDF. */
function normalCdf(z: number): number {
  return 0.5 * (1 + erf(z / Math.SQRT2));
}

/**
 * Two-sided sign test against a fair-coin null.
 *
 * Uses the normal approximation, which is accurate from roughly n >= 30. Below
 * that the statistic is still returned but `significant` stays false, because
 * claiming significance from a handful of pairs is how false alarms start.
 */
export function signTest(successes: number, n: number): BiasTest {
  if (n < 0 || successes < 0 || successes > n) {
    throw new RangeError(`invalid counts: ${successes} successes of ${n}`);
  }
  if (n === 0) {
    return { n: 0, successes: 0, rate: 0.5, z: 0, pValue: 1, significant: false };
  }

  const rate = successes / n;
  const z = (successes - n / 2) / Math.sqrt(n / 4);
  const pValue = 2 * (1 - normalCdf(Math.abs(z)));

  return {
    n,
    successes,
    rate,
    z,
    pValue,
    significant: n >= 30 && pValue < 0.05,
  };
}

/**
 * Measure whether annotators systematically preferred longer responses.
 *
 * Pairs of equal length are excluded from the test, as in any sign test: they
 * carry no directional information.
 */
export function lengthBias(pairs: readonly PreferencePair[]): LengthBiasReport {
  if (pairs.length === 0) {
    return {
      ...signTest(0, 0),
      meanChosenLength: 0,
      meanRejectedLength: 0,
      meanDelta: 0,
    };
  }

  let chosenTotal = 0;
  let rejectedTotal = 0;
  let longerChosen = 0;
  let decisive = 0;

  for (const pair of pairs) {
    const chosenLength = pair.chosen.length;
    const rejectedLength = pair.rejected.length;

    chosenTotal += chosenLength;
    rejectedTotal += rejectedLength;

    if (chosenLength === rejectedLength) continue;
    decisive += 1;
    if (chosenLength > rejectedLength) longerChosen += 1;
  }

  const meanChosenLength = chosenTotal / pairs.length;
  const meanRejectedLength = rejectedTotal / pairs.length;

  return {
    ...signTest(longerChosen, decisive),
    meanChosenLength,
    meanRejectedLength,
    meanDelta: meanChosenLength - meanRejectedLength,
  };
}

/**
 * Measure whether the side a response was shown on affected the verdict.
 *
 * Only pairs carrying `chosenPosition` contribute. A significant result means
 * the annotation UI is leaking signal and positions need randomising.
 */
export function positionBias(pairs: readonly PreferencePair[]): BiasTest {
  let left = 0;
  let total = 0;

  for (const pair of pairs) {
    if (pair.chosenPosition === undefined) continue;
    total += 1;
    if (pair.chosenPosition === 'left') left += 1;
  }

  return signTest(left, total);
}

/** Per-annotator length preference, to find who is driving a dataset-level bias. */
export function annotatorSkew(
  pairs: readonly PreferencePair[],
): readonly AnnotatorSkew[] {
  const counts = new Map<string, { longer: number; decisive: number }>();

  for (const pair of pairs) {
    const chosenLength = pair.chosen.length;
    const rejectedLength = pair.rejected.length;
    if (chosenLength === rejectedLength) continue;

    const entry = counts.get(pair.annotatorId) ?? { longer: 0, decisive: 0 };
    entry.decisive += 1;
    if (chosenLength > rejectedLength) entry.longer += 1;
    counts.set(pair.annotatorId, entry);
  }

  return [...counts.entries()]
    .map(([annotatorId, { longer, decisive }]) => ({
      annotatorId,
      pairs: decisive,
      longerPreferredRate: decisive === 0 ? 0.5 : longer / decisive,
    }))
    .sort((a, b) => b.longerPreferredRate - a.longerPreferredRate);
}

/**
 * Find prompts where annotators contradicted each other on the same two
 * responses.
 *
 * Contradictions are not automatically errors — genuinely close calls produce
 * them — but a high rate means the guidelines are underspecified, and the
 * affected pairs will fight each other during training.
 */
export function findConflicts(pairs: readonly PreferencePair[]): readonly Conflict[] {
  // Key each unordered {response1, response2} pair under its prompt, then look
  // for both directions having been chosen.
  const buckets = new Map<
    string,
    { prompt: string; a: string; b: string; preferA: Set<string>; preferB: Set<string> }
  >();

  for (const pair of pairs) {
    const [a, b] =
      pair.chosen < pair.rejected
        ? [pair.chosen, pair.rejected]
        : [pair.rejected, pair.chosen];
    const key = `${pair.prompt}\u0000${a}\u0000${b}`;

    const bucket = buckets.get(key) ?? {
      prompt: pair.prompt,
      a,
      b,
      preferA: new Set<string>(),
      preferB: new Set<string>(),
    };

    if (pair.chosen === a) bucket.preferA.add(pair.annotatorId);
    else bucket.preferB.add(pair.annotatorId);

    buckets.set(key, bucket);
  }

  const conflicts: Conflict[] = [];
  for (const bucket of buckets.values()) {
    if (bucket.preferA.size > 0 && bucket.preferB.size > 0) {
      conflicts.push({
        prompt: bucket.prompt,
        responseA: bucket.a,
        responseB: bucket.b,
        preferredA: [...bucket.preferA].sort(),
        preferredB: [...bucket.preferB].sort(),
      });
    }
  }

  return conflicts;
}
