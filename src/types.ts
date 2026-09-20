/**
 * Core domain types for RLHF preference data.
 *
 * A preference pair is the atom of DPO, GRPO and reward-model training: one
 * prompt, one response a human preferred, one they did not.
 */

/** Which side of a comparison an annotator preferred. */
export type Verdict = 'chosen' | 'rejected' | 'tie';

/** Where a response was shown in the annotation UI, for position-bias audits. */
export type Position = 'left' | 'right';

/**
 * One preference judgement.
 *
 * `chosen` and `rejected` hold the response text itself rather than an id,
 * because every downstream trainer (TRL, OpenRLHF, Axolotl) consumes the text.
 */
export interface PreferencePair {
  /** Stable identifier. Generated from content when not supplied. */
  readonly id: string;
  /** The prompt both responses answered. */
  readonly prompt: string;
  /** The preferred response. */
  readonly chosen: string;
  /** The response that was not preferred. */
  readonly rejected: string;
  /** Who made the judgement. Needed for agreement and skew analysis. */
  readonly annotatorId: string;
  /** Which model produced the chosen response, when known. */
  readonly chosenModel?: string;
  /** Which model produced the rejected response, when known. */
  readonly rejectedModel?: string;
  /** Where the chosen response appeared in the UI, for position-bias audits. */
  readonly chosenPosition?: Position;
  /** Free-text justification. Strongly recommended; reviewers rely on it. */
  readonly rationale?: string;
  /** Annotator confidence in [0, 1]. Used to weight or filter pairs. */
  readonly confidence?: number;
  /** Arbitrary passthrough metadata. */
  readonly metadata?: Readonly<Record<string, unknown>>;
}

/** A pair before validation and id assignment. */
export type DraftPair = Omit<PreferencePair, 'id'> & { readonly id?: string };

/**
 * A GRPO-style group: several responses to one prompt, ranked best to worst.
 *
 * GRPO computes advantage relative to the group mean, so it needs the whole
 * group rather than isolated pairs.
 */
export interface PreferenceGroup {
  readonly id: string;
  readonly prompt: string;
  readonly annotatorId: string;
  /** Responses ordered best first. At least two. */
  readonly ranked: readonly string[];
  readonly metadata?: Readonly<Record<string, unknown>>;
}

/** Train/validation/test assignment. */
export type Split = 'train' | 'validation' | 'test';

/** Proportions for {@link splitDataset}. Must sum to 1. */
export interface SplitRatios {
  readonly train: number;
  readonly validation: number;
  readonly test: number;
}

/** One validation failure, located precisely enough to fix. */
export interface ValidationIssue {
  /** Index of the offending record in the input array. */
  readonly index: number;
  /** Which field failed, or `'*'` for a whole-record problem. */
  readonly field: string;
  /** Machine-readable reason, stable across releases. */
  readonly code: ValidationCode;
  /** Human-readable explanation. */
  readonly message: string;
}

export type ValidationCode =
  | 'EMPTY_FIELD'
  | 'IDENTICAL_RESPONSES'
  | 'CONFIDENCE_OUT_OF_RANGE'
  | 'DUPLICATE_ID'
  | 'GROUP_TOO_SMALL'
  | 'DUPLICATE_RESPONSE_IN_GROUP';

/** Outcome of validating a batch. */
export interface ValidationResult<T> {
  readonly valid: readonly T[];
  readonly issues: readonly ValidationIssue[];
}
