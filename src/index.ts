/**
 * preference-sdk — build, validate and audit RLHF preference datasets.
 *
 * @packageDocumentation
 */

export { PreferenceValidationError } from './errors.js';

export type {
  DraftPair,
  Position,
  PreferenceGroup,
  PreferencePair,
  Split,
  SplitRatios,
  ValidationCode,
  ValidationIssue,
  ValidationResult,
  Verdict,
} from './types.js';

export {
  groupToPairs,
  pairId,
  validateGroups,
  validatePairs,
  validatePairsStrict,
} from './validate.js';

export type {
  AnnotatorSkew,
  BiasTest,
  Conflict,
  LengthBiasReport,
} from './diagnostics.js';
export {
  annotatorSkew,
  findConflicts,
  lengthBias,
  positionBias,
  signTest,
} from './diagnostics.js';

export type { DatasetSummary, SplitDataset } from './dataset.js';
export {
  DEFAULT_RATIOS,
  deduplicate,
  splitDataset,
  splitFor,
  summarize,
} from './dataset.js';

export type { ExportFormat } from './export.js';
export { encodePair, streamJsonl, toJsonl } from './export.js';
