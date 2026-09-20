import type { ValidationIssue } from './types.js';

/**
 * Thrown by the strict entry points when a batch contains invalid records.
 *
 * Carries every issue rather than only the first, so a malformed export can be
 * fixed in one pass instead of one error at a time.
 */
export class PreferenceValidationError extends Error {
  readonly issues: readonly ValidationIssue[];

  constructor(issues: readonly ValidationIssue[]) {
    const preview = issues
      .slice(0, 5)
      .map((i) => `  [${i.index}] ${i.field}: ${i.message}`)
      .join('\n');
    const more = issues.length > 5 ? `\n  ...and ${issues.length - 5} more` : '';

    super(`${issues.length} invalid record(s):\n${preview}${more}`);
    this.name = 'PreferenceValidationError';
    this.issues = issues;
  }
}
