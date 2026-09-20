/**
 * Validation and stable id assignment.
 *
 * Validation is non-throwing by default: a 40,000-pair export usually has a
 * handful of bad records, and dropping the batch because of them wastes the
 * rest. Collect the issues, keep the good records, and let the caller decide.
 */

import { createHash } from 'node:crypto';

import { PreferenceValidationError } from './errors.js';
import type {
  DraftPair,
  PreferenceGroup,
  PreferencePair,
  ValidationIssue,
  ValidationResult,
} from './types.js';

/**
 * Deterministic id derived from the content that defines a pair.
 *
 * Content-addressed rather than random so the same pair gets the same id on
 * every machine and every run: re-ingesting an export is idempotent, and
 * splits stay stable across regenerations.
 */
export function pairId(prompt: string, chosen: string, rejected: string): string {
  return createHash('sha256')
    .update(prompt)
    .update('\u0000')
    .update(chosen)
    .update('\u0000')
    .update(rejected)
    .digest('hex')
    .slice(0, 16);
}

function issue(
  index: number,
  field: string,
  code: ValidationIssue['code'],
  message: string,
): ValidationIssue {
  return { index, field, code, message };
}

function checkNonEmpty(
  value: string,
  index: number,
  field: string,
  issues: ValidationIssue[],
): boolean {
  if (value.trim().length === 0) {
    issues.push(issue(index, field, 'EMPTY_FIELD', `${field} must not be empty`));
    return false;
  }
  return true;
}

/**
 * Validate a batch of draft pairs, assigning ids where absent.
 *
 * @returns Valid pairs plus every issue found. Never throws.
 */
export function validatePairs(
  drafts: readonly DraftPair[],
): ValidationResult<PreferencePair> {
  const issues: ValidationIssue[] = [];
  const valid: PreferencePair[] = [];
  const seenIds = new Set<string>();

  drafts.forEach((draft, index) => {
    let ok = true;

    ok = checkNonEmpty(draft.prompt, index, 'prompt', issues) && ok;
    ok = checkNonEmpty(draft.chosen, index, 'chosen', issues) && ok;
    ok = checkNonEmpty(draft.rejected, index, 'rejected', issues) && ok;
    ok = checkNonEmpty(draft.annotatorId, index, 'annotatorId', issues) && ok;

    // A pair where both sides are identical carries no training signal and
    // will produce a zero gradient under DPO.
    if (draft.chosen === draft.rejected) {
      issues.push(
        issue(
          index,
          'chosen',
          'IDENTICAL_RESPONSES',
          'chosen and rejected are identical, so the pair carries no signal',
        ),
      );
      ok = false;
    }

    if (draft.confidence !== undefined) {
      if (
        !Number.isFinite(draft.confidence) ||
        draft.confidence < 0 ||
        draft.confidence > 1
      ) {
        issues.push(
          issue(
            index,
            'confidence',
            'CONFIDENCE_OUT_OF_RANGE',
            `confidence must be within [0, 1], got ${String(draft.confidence)}`,
          ),
        );
        ok = false;
      }
    }

    if (!ok) return;

    const id = draft.id ?? pairId(draft.prompt, draft.chosen, draft.rejected);

    if (seenIds.has(id)) {
      issues.push(
        issue(index, 'id', 'DUPLICATE_ID', `duplicate pair id ${id} within the batch`),
      );
      return;
    }
    seenIds.add(id);

    valid.push({ ...draft, id });
  });

  return { valid, issues };
}

/** As {@link validatePairs}, but throws {@link PreferenceValidationError} on any issue. */
export function validatePairsStrict(
  drafts: readonly DraftPair[],
): readonly PreferencePair[] {
  const { valid, issues } = validatePairs(drafts);
  if (issues.length > 0) throw new PreferenceValidationError(issues);
  return valid;
}

/** Validate GRPO-style ranked groups. */
export function validateGroups(
  groups: readonly PreferenceGroup[],
): ValidationResult<PreferenceGroup> {
  const issues: ValidationIssue[] = [];
  const valid: PreferenceGroup[] = [];

  groups.forEach((group, index) => {
    let ok = true;

    ok = checkNonEmpty(group.prompt, index, 'prompt', issues) && ok;
    ok = checkNonEmpty(group.annotatorId, index, 'annotatorId', issues) && ok;

    if (group.ranked.length < 2) {
      issues.push(
        issue(
          index,
          'ranked',
          'GROUP_TOO_SMALL',
          `a group needs at least 2 responses, got ${group.ranked.length}`,
        ),
      );
      ok = false;
    }

    // Duplicates make the ranking ambiguous: which copy outranks which?
    const unique = new Set(group.ranked);
    if (unique.size !== group.ranked.length) {
      issues.push(
        issue(
          index,
          'ranked',
          'DUPLICATE_RESPONSE_IN_GROUP',
          'the same response appears twice, making the ranking ambiguous',
        ),
      );
      ok = false;
    }

    if (ok) valid.push(group);
  });

  return { valid, issues };
}

/**
 * Expand a ranked group into every implied pairwise preference.
 *
 * A group ranked `[a, b, c]` implies a>b, a>c and b>c — the standard way to
 * feed GRPO-collected rankings into a pairwise DPO trainer.
 */
export function groupToPairs(group: PreferenceGroup): readonly PreferencePair[] {
  const pairs: PreferencePair[] = [];

  for (let better = 0; better < group.ranked.length; better += 1) {
    for (let worse = better + 1; worse < group.ranked.length; worse += 1) {
      const chosen = group.ranked[better];
      const rejected = group.ranked[worse];
      // Guarded by noUncheckedIndexedAccess; indices are in range by construction.
      if (chosen === undefined || rejected === undefined) continue;

      pairs.push({
        id: pairId(group.prompt, chosen, rejected),
        prompt: group.prompt,
        chosen,
        rejected,
        annotatorId: group.annotatorId,
        ...(group.metadata === undefined
          ? {}
          : { metadata: { ...group.metadata, sourceGroupId: group.id } }),
      });
    }
  }

  return pairs;
}
