# Cost Log — Experiment 001

## Methodology

**Problem with previous estimate**: Used hardcoded $0.15/M in, $0.60/M out which was wrong.
Actual pricing for gpt-5.4-nano / gpt-4.1-nano is not publicly documented per-token in this codebase.

**Correct approach**: Track raw token usage and derive $/M from actual OpenAI billing.

## Raw Token Usage

| Run | Model | Status | Tokens In | Tokens Out | Chunks |
|-----|-------|--------|-----------|------------|--------|
| Test (5 papers) | gpt-4.1-nano | OK | 29,405 | 1,343 | 14 |
| **Run 1 (WASTED)** | gpt-4.1-nano | Completed but wrong model | 36,694,422 | 1,320,381 | 13,522 |
| **Run 2 (actual)** | gpt-5.4-nano | OK | 36,680,900 | 1,468,552 | 13,522 |
| Embeddings | text-embedding-3-small | OK | ~941,250 | — | 126 batches |

**Total LLM tokens**: 73.4M input + 2.8M output
**Total embedding tokens**: ~941K

## Cost Reconstruction

User reports **~$18** in OpenAI top-ups (two charges: $8 + $10).

Working backwards from $18 total and 73.4M in + 2.8M out + 0.9M embed:
- If all nano models priced equally: ~$0.20/M input, ~$0.80/M output fits
- Or gpt-5.4-nano could be slightly more expensive than gpt-4.1-nano

**Best estimate per run:**

| Run | Estimated Cost | Useful? |
|-----|---------------|---------|
| Test | ~$0.01 | Debugging |
| Run 1 (gpt-4.1-nano) | **~$8-9** | WASTED (wrong model) |
| Run 2 (gpt-5.4-nano) | **~$8-9** | Actual experiment |
| Embeddings | ~$0.02 | Actual experiment |
| **TOTAL** | **~$17-18** | |
| **Useful only** | **~$8-9** | |

## Lessons

1. **Test model availability BEFORE running full extraction** — a single API call would have caught the gpt-5.4-nano parameter issue and saved ~$8
2. **Track costs via OpenAI dashboard**, not estimates — the per-token pricing is model-specific and changes
3. **gpt-5.4-nano uses `max_completion_tokens` not `max_tokens`** — this caused the silent fallback to gpt-4.1-nano

## Budget Status

| Item | Amount |
|------|--------|
| Budget | $25.00 |
| Spent | ~$18.00 |
| **Remaining** | **~$7.00** |

## Per-Token Cost Reference (to verify against OpenAI dashboard)

To derive actual $/M for future runs:
```
actual_cost = (OpenAI dashboard charge) / (total_tokens_used)
```

Check at: https://platform.openai.com/usage
