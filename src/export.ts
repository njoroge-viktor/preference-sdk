/**
 * Serialisation to the formats real trainers consume.
 *
 * Each trainer expects slightly different field names; getting them wrong
 * fails deep inside a training run rather than at load time, so the mapping
 * lives here and is tested.
 */

import type { PreferencePair } from './types.js';

/** Field layout expected by the target trainer. */
export type ExportFormat =
  /** TRL's `DPOTrainer`: `{prompt, chosen, rejected}`. */
  | 'trl-dpo'
  /** OpenAI preference fine-tuning: messages plus `preferred_completion`. */
  | 'openai-preference'
  /** Everything, for archival and re-import. */
  | 'raw';

function toTrlDpo(pair: PreferencePair): Record<string, unknown> {
  return { prompt: pair.prompt, chosen: pair.chosen, rejected: pair.rejected };
}

function toOpenAiPreference(pair: PreferencePair): Record<string, unknown> {
  return {
    input: { messages: [{ role: 'user', content: pair.prompt }] },
    preferred_completion: [{ role: 'assistant', content: pair.chosen }],
    non_preferred_completion: [{ role: 'assistant', content: pair.rejected }],
  };
}

function toRaw(pair: PreferencePair): Record<string, unknown> {
  // Drop undefined optionals so the JSONL stays compact and stable.
  return Object.fromEntries(Object.entries(pair).filter(([, v]) => v !== undefined));
}

const ENCODERS: Record<ExportFormat, (pair: PreferencePair) => Record<string, unknown>> =
  {
    'trl-dpo': toTrlDpo,
    'openai-preference': toOpenAiPreference,
    raw: toRaw,
  };

/** Encode one pair into the target format. */
export function encodePair(
  pair: PreferencePair,
  format: ExportFormat = 'trl-dpo',
): Record<string, unknown> {
  return ENCODERS[format](pair);
}

/**
 * Serialise pairs to JSONL.
 *
 * The result has a trailing newline, which is what every line-oriented reader
 * expects and what `wc -l` counts correctly.
 */
export function toJsonl(
  pairs: readonly PreferencePair[],
  format: ExportFormat = 'trl-dpo',
): string {
  if (pairs.length === 0) return '';
  return `${pairs.map((p) => JSON.stringify(encodePair(p, format))).join('\n')}\n`;
}

/** Stream pairs as JSONL lines, for exports too large to hold in memory. */
export function* streamJsonl(
  pairs: Iterable<PreferencePair>,
  format: ExportFormat = 'trl-dpo',
): Generator<string> {
  for (const pair of pairs) {
    yield `${JSON.stringify(encodePair(pair, format))}\n`;
  }
}
