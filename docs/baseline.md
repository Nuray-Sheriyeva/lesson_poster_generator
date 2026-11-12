# Relationtype baseline

Status: **blocked on OpenAI credits**. Langfuse Cloud is configured and working:
the `genlesson-fixtures` dataset has five items and the seven managed prompts are
synced. The first plan-only relationtype attempt failed at ingest with OpenAI
HTTP 429 ("no credits remaining"), so no baseline numbers can be recorded yet.
Add OpenAI credits, then rerun the three baseline runs below.

## Locked setup

- Input: `fixtures/input/relationtype.pdf`
- Format: `portrait`
- Review: `off`
- Command shape:
  `genlesson run fixtures/input/relationtype.pdf --run-id <id> --runs-dir runs/baseline --format portrait --review off`
- Three independent runs with fresh run ids.
- Baseline freeze: the LangGraph path must produce the same provider call
  counts and similar verify scores as the sequential eight-command path.

## What each run records

| Run | Status | Wall/latency | Planner calls | Image calls | Vision calls | Image cost | Verify scores | Verify status | Trace id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-1 | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| baseline-2 | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| baseline-3 | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Earlier reference

The earlier included run ended with verify attempt scores **0.20, 0.05, 0.25**
across three attempts. That is the comparison point for the rerun; do not
"fix" those numbers by changing the verifier.

## Artifacts to read after each run

- `runs/baseline/<id>/stage-06-generate/generation.json`: image usage,
  `last_call.latency_ms`, request ids, attempts.
- `runs/baseline/<id>/stage-07-verify/verification.json`: per-attempt
  `score`, `passed`, `exact_mismatches`, `extra_strings`, final status.
- `runs/baseline/<id>/stage-08-save/save.json` and the poster sidecars:
  call counts, `cost`, and `tracing.trace_id`.
- Langfuse: filter the trace by score `verify_status`.
