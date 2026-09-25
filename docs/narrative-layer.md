# The AI narrative layer

*Added in v0.2. Optional — the core engine, console, and eval run
without it.*

## Principle

The deterministic engine produces evidence; the LLM explains it. The
model is given a voice, never authority:

- The narrative may only discuss the **leading hypothesis**. The model
  is never shown the alternatives — the engine's ranking of them is
  appended to the finished narrative by code, clearly labelled as
  machine-written.
- Every factual claim must cite an evidence ID from the case file,
  copied character-for-character.
- A **validator** parses the citations back out and rejects the
  narrative if it cites an unknown ID or contains an uncited claim
  paragraph. One corrective retry (the validator's findings are fed
  back); still invalid → the narrative is refused and never shown.

## Flow

```text
CaseFile ──▶ build_prompt (leading hypothesis + its evidence, with IDs)
        ──▶ Anthropic API ──▶ validate_narrative
                                 │ valid: append machine-written
                                 │        "Alternatives" section
                                 └ invalid: one retry with feedback,
                                            then refuse
```

Entry points: `whydunit.explain.narrate()` (library),
`python -m whydunit.explain` (CLI), and the "AI narrative report"
panel on the console's Notes & Case File page.

## Validation rules

`validate_narrative(text, case)` enforces:

1. Every `[ID]`-style citation must exist in the case file's evidence
   registry (all hypotheses' supporting and contradicting evidence).
2. Every prose paragraph must carry at least one citation. Headings are
   exempt; the "What to do next" section is exempt (recommendations
   derive from the engine's `next_step` fields).
3. A narrative with zero citations is invalid outright.

Refusals are surfaced with the validator's findings — in the console as
an error box, from the CLI as a non-zero exit.

## Design history

Three refusals during development shaped the final design: the model
prefixed real evidence IDs with an invented `EV-` scheme (fixed by
quoting exact IDs with a dynamic example), narrated scene-setting facts
without citations (fixed by requiring case facts to be folded into
cited sentences), and repeatedly summarised lower-ranked hypotheses
uncited. The third was solved structurally rather than by prompt
tuning: the model no longer sees the alternatives at all. Every one of
those refusals was the validator doing its job — none of the invalid
narratives was ever displayed.

## Setup

```bash
uv sync --extra llm          # installs the anthropic SDK
export ANTHROPIC_API_KEY=... # PowerShell: $env:ANTHROPIC_API_KEY="..."
uv run python -m whydunit.explain --scenario retrieval_degradation --seed 42
```

The dependency is an optional extra: a plain `uv sync` installs no LLM
SDK, CI runs without a key, and the explain-layer tests use a stubbed
client — no network, no cost. The ground-truth firewall is unchanged:
the narrative layer reads case files, which the engine produced without
ever seeing ground truth.