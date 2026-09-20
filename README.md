# preference-sdk

[![CI](https://github.com/njoroge-viktor/preference-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/njoroge-viktor/preference-sdk/actions/workflows/ci.yml)
[![TypeScript](https://img.shields.io/badge/TypeScript-strict-blue)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Build, validate and **audit** RLHF preference datasets before they reach a
trainer. Typed end to end, zero runtime dependencies.

```bash
npm install @njoroge-viktor/preference-sdk
```

## Why this exists

A preference dataset can be perfectly well-formed and still teach the wrong
thing. The failure modes are well known and mostly invisible without
measurement:

- **Length bias** — if annotators systematically preferred longer answers, the
  reward model learns _longer is better_ and the policy degenerates into
  padding. This is the single most common way an RLHF run quietly fails.
- **Position bias** — if the annotation UI did not randomise sides, the model
  learns the UI rather than the preference.
- **Prompt leakage across splits** — reward models memorise prompts, so a
  prompt appearing in both train and test inflates eval scores until
  production disagrees.
- **Contradictory pairs** — the same two responses preferred both ways fight
  each other during training.

This library measures all four.

## Quick start

```ts
import {
  validatePairsStrict,
  deduplicate,
  splitDataset,
  lengthBias,
  toJsonl,
} from '@njoroge-viktor/preference-sdk';

const pairs = validatePairsStrict(rawExport); // throws with every issue at once
const clean = deduplicate(pairs);

const report = lengthBias(clean);
if (report.significant) {
  throw new Error(
    `Length bias: the longer response won ${(report.rate * 100).toFixed(1)}% ` +
      `of ${report.n} pairs (p=${report.pValue.toExponential(2)})`,
  );
}

const { train, validation, test } = splitDataset(clean);
await fs.writeFile('train.jsonl', toJsonl(train, 'trl-dpo'));
```

## Auditing

`lengthBias` runs a two-sided sign test against a fair-coin null. Equal-length
pairs are excluded, as in any sign test — they carry no directional
information.

```ts
lengthBias(pairs);
// { n: 412, successes: 390, rate: 0.947, z: 18.2, pValue: 0, significant: true,
//   meanChosenLength: 1840, meanRejectedLength: 402, meanDelta: 1438 }
```

`significant` requires both `p < 0.05` **and** `n >= 30`, because the normal
approximation is not trustworthy below that and claiming significance from a
handful of pairs is how false alarms start.

Once a dataset-level bias shows up, `annotatorSkew` finds who is driving it:

```ts
annotatorSkew(pairs);
// [ { annotatorId: 'ann-c', pairs: 210, longerPreferredRate: 0.98 },   <- retrain this one
//   { annotatorId: 'ann-a', pairs: 190, longerPreferredRate: 0.52 } ]
```

`findConflicts` surfaces prompts where annotators contradicted each other.
Contradictions are not automatically errors — genuinely close calls produce
them — but a high rate means the guidelines are underspecified.

## Splitting without leakage

`splitDataset` buckets on a **hash of the prompt**, not on input order or a
random number generator:

```ts
splitDataset(pairs, { train: 0.8, validation: 0.1, test: 0.1 }, 'my-seed');
```

Three properties fall out of that, all of which matter when a dataset is
regenerated weekly:

1. **Every pair sharing a prompt lands in the same split.** No leakage.
2. **The split is reproducible** across machines and runs.
3. **Appending new prompts never reshuffles existing ones**, so an earlier
   train/test boundary stays valid.

`splitFor(prompt)` answers the same question for a single prompt without
materialising the dataset.

## Validation

Validation is non-throwing by default. A 40,000-pair export usually has a
handful of bad records, and dropping the batch because of them wastes the rest:

```ts
const { valid, issues } = validatePairs(rawExport);
// issues: [{ index: 1843, field: 'chosen', code: 'IDENTICAL_RESPONSES',
//            message: 'chosen and rejected are identical, so the pair carries no signal' }]
```

Use `validatePairsStrict` in CI, where you do want a hard failure. It throws a
`PreferenceValidationError` carrying **every** issue, so a malformed export can
be fixed in one pass.

| Code                          | Meaning                                                   |
| ----------------------------- | --------------------------------------------------------- |
| `EMPTY_FIELD`                 | A required field was blank                                |
| `IDENTICAL_RESPONSES`         | Zero gradient under DPO; the pair teaches nothing         |
| `CONFIDENCE_OUT_OF_RANGE`     | Confidence outside `[0, 1]`                               |
| `DUPLICATE_ID`                | The same pair appeared twice in the batch                 |
| `GROUP_TOO_SMALL`             | A GRPO group with fewer than two responses                |
| `DUPLICATE_RESPONSE_IN_GROUP` | The same response ranked twice, so the order is ambiguous |

Pair ids are **content-addressed** (SHA-256 over prompt, chosen and rejected,
delimited so field boundaries cannot collide). Re-ingesting an export is
therefore idempotent, and ids are stable across machines.

## GRPO groups

GRPO collects a ranked group per prompt rather than isolated pairs.
`groupToPairs` expands a ranking into every implied preference, so
group-collected data can feed a pairwise DPO trainer:

```ts
groupToPairs({
  id: 'g1',
  prompt: 'p',
  annotatorId: 'ann-a',
  ranked: ['best', 'mid', 'worst'],
});
// best > mid, best > worst, mid > worst
```

## Export formats

| Format              | Consumer                                        |
| ------------------- | ----------------------------------------------- |
| `trl-dpo`           | TRL `DPOTrainer` — `{prompt, chosen, rejected}` |
| `openai-preference` | OpenAI preference fine-tuning                   |
| `raw`               | Everything, for archival and re-import          |

`toJsonl` buffers; `streamJsonl` yields line by line for exports too large to
hold in memory. Both terminate records with a newline so `wc -l` is correct.

## Development

```bash
npm install
npm test           # 102 tests
npm run typecheck  # strict, plus noUncheckedIndexedAccess and exactOptionalPropertyTypes
npm run lint
npm run build
```

`tsconfig` runs beyond `strict`: `noUncheckedIndexedAccess` and
`exactOptionalPropertyTypes` are both on, so an absent optional and one set to
`undefined` are treated as the different things they are.

The statistics are pinned to values derived in the tests — `z = 2` must yield
`p ≈ 0.0455` against the normal tail — rather than to whatever the
implementation happened to return.

## License

MIT — see [LICENSE](LICENSE).
