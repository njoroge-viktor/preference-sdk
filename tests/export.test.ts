import { describe, expect, it } from 'vitest';

import { encodePair, streamJsonl, toJsonl } from '../src/export.js';
import type { PreferencePair } from '../src/types.js';

const pair: PreferencePair = {
  id: 'abc123',
  prompt: 'Explain closures.',
  chosen: 'A closure captures its lexical scope.',
  rejected: 'Idk.',
  annotatorId: 'ann-a',
  confidence: 0.9,
};

describe('encodePair', () => {
  it('emits exactly the fields TRL DPOTrainer reads', () => {
    expect(encodePair(pair, 'trl-dpo')).toEqual({
      prompt: pair.prompt,
      chosen: pair.chosen,
      rejected: pair.rejected,
    });
  });

  it('nests OpenAI preference records under the expected keys', () => {
    const encoded = encodePair(pair, 'openai-preference');
    expect(encoded).toEqual({
      input: { messages: [{ role: 'user', content: pair.prompt }] },
      preferred_completion: [{ role: 'assistant', content: pair.chosen }],
      non_preferred_completion: [{ role: 'assistant', content: pair.rejected }],
    });
  });

  it('round-trips every populated field in raw format', () => {
    expect(encodePair(pair, 'raw')).toEqual(pair);
  });

  it('omits absent optionals from raw output', () => {
    // Built without the key rather than setting it to undefined: under
    // exactOptionalPropertyTypes those are genuinely different things.
    const minimal: PreferencePair = {
      id: pair.id,
      prompt: pair.prompt,
      chosen: pair.chosen,
      rejected: pair.rejected,
      annotatorId: pair.annotatorId,
    };
    expect(encodePair(minimal, 'raw')).not.toHaveProperty('confidence');
  });

  it('defaults to the TRL layout', () => {
    expect(Object.keys(encodePair(pair))).toEqual(['prompt', 'chosen', 'rejected']);
  });
});

describe('toJsonl', () => {
  it('emits one parseable object per line', () => {
    const jsonl = toJsonl([pair, { ...pair, id: 'def456' }]);
    const lines = jsonl.trimEnd().split('\n');

    expect(lines).toHaveLength(2);
    expect(JSON.parse(lines[0] as string)).toHaveProperty('prompt', pair.prompt);
  });

  it('terminates with a newline so line counts are correct', () => {
    expect(toJsonl([pair])).toMatch(/\n$/);
  });

  it('returns an empty string for no pairs, not a bare newline', () => {
    expect(toJsonl([])).toBe('');
  });

  it('escapes newlines inside response text', () => {
    const multiline = { ...pair, chosen: 'line one\nline two' };
    const jsonl = toJsonl([multiline]);

    // One record must stay on one line, whatever the content.
    expect(jsonl.trimEnd().split('\n')).toHaveLength(1);
    expect(JSON.parse(jsonl.trimEnd()) as { chosen: string }).toHaveProperty(
      'chosen',
      'line one\nline two',
    );
  });
});

describe('streamJsonl', () => {
  it('yields the same bytes as the buffered writer', () => {
    const pairs = [pair, { ...pair, id: 'x' }];
    expect([...streamJsonl(pairs)].join('')).toBe(toJsonl(pairs));
  });

  it('is lazy, so it can outlive available memory', () => {
    const iterator = streamJsonl([pair, { ...pair, id: 'y' }]);
    expect(iterator.next().done).toBe(false);
  });
});
