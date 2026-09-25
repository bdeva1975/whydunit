"""Narrative incident reports, generated from a finished case file.

The LLM writes prose; the case file supplies every fact. The prompt
embeds an evidence registry with stable IDs and instructs the model to
cite an ID for every factual claim. ``validator`` (separate module)
rejects narratives that cite unknown IDs or make uncited claims.

The ``anthropic`` import is lazy: this module can be imported without
the optional ``llm`` extra installed, and tests inject a stub client.
"""

from __future__ import annotations

from typing import Any

from whydunit.models import CaseFile, Evidence

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1500


def evidence_registry(case: CaseFile) -> dict[str, Evidence]:
    """All evidence items in the case file, keyed by ID.

    Walks every hypothesis's supporting and contradicting lists; the
    same evidence item may back several hypotheses and is kept once.
    """
    registry: dict[str, Evidence] = {}
    for hypothesis in case.hypotheses:
        for item in (*hypothesis.supporting, *hypothesis.contradicting):
            registry.setdefault(item.id, item)
    return dict(sorted(registry.items()))


def build_prompt(case: CaseFile, correction: str | None = None) -> str:
    """The single-turn prompt: case facts, evidence registry, rules.

    ``correction`` carries validator feedback on a retry; it is appended
    as an explicit fix-list so the model rewrites rather than continues.
    """
    registry = evidence_registry(case)

    evidence_lines = [
        f"[{item.id}] ({item.kind.value}) {item.statement} "
        f"(signals: {', '.join(str(ref) for ref in item.signals)})"
        for item in registry.values()
    ]

    hypothesis_lines = []
    for rank, hypothesis in enumerate(case.hypotheses, start=1):
        supporting = ", ".join(item.id for item in hypothesis.supporting) or "none"
        contradicting = ", ".join(item.id for item in hypothesis.contradicting) or "none"
        hypothesis_lines.append(
            f"{rank}. {hypothesis.statement} "
            f"[category: {hypothesis.category.value}, "
            f"confidence: {hypothesis.confidence.value}, "
            f"supporting: {supporting}, contradicting: {contradicting}]"
        )

    lines = [
        "You are writing the narrative section of a forensic incident report",
        "for an AI pipeline. A deterministic forensic engine has already done",
        "the analysis; your job is ONLY to explain its findings in clear prose.",
        "",
        "## Case facts",
        f"Pipeline: {case.pipeline_name}",
        f"Incident window (UTC): {case.window_start.isoformat()} to {case.window_end.isoformat()}",
        f"Severity: {case.severity.value}",
        f"Engine summary: {case.summary}",
        "",
        "## Symptoms",
        *[f"- {symptom}" for symptom in case.symptoms],
        "",
        "## Evidence registry (the ONLY facts you may use)",
        *evidence_lines,
        "",
        "## Ranked root-cause hypotheses (from the engine)",
        *hypothesis_lines,
        "",
        "## Rules",
        "1. Every factual claim MUST end with one or more citations in the",
        "   form [EV-n], using ONLY IDs from the evidence registry above.",
        "2. Do not introduce any number, signal name, cause, or event that",
        "   is not in the registry or case facts. No speculation.",
        "3. Do not upgrade the engine's confidence: describe the leading",
        "   hypothesis using its stated confidence band, and mention that",
        "   contradicting evidence exists where it does.",
        "4. Structure: 'What happened', 'Why the engine believes this',",
        "   'What to do next'. Markdown headings. At most 400 words.",
        "5. Do not add a disclaimer section; the case file carries one.",
    ]

    if correction:
        lines += ["", "## Correction required", correction]

    lines += ["", "Write the narrative now."]
    return "\n".join(lines)


def generate_narrative(
    case: CaseFile,
    client: Any | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    correction: str | None = None,
) -> str:
    """Ask the model for a narrative. Raises if there is nothing to explain.

    ``client`` is any object exposing the Anthropic Messages API surface
    (``client.messages.create``). When omitted, the real SDK client is
    created — which requires the ``llm`` extra and ``ANTHROPIC_API_KEY``.
    """
    if not evidence_registry(case):
        raise ValueError(
            "Case file contains no evidence; there is nothing for a narrative "
            "to cite. Narratives are only generated for investigated incidents."
        )

    if client is None:
        try:
            import anthropic
        except ImportError as error:  # pragma: no cover - exercised manually
            raise RuntimeError(
                "The optional 'llm' extra is not installed. Run: uv sync --extra llm"
            ) from error
        client = anthropic.Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": build_prompt(case, correction=correction)}],
    )
    text = "".join(
        getattr(block, "text", "")
        for block in response.content
        if getattr(block, "type", "") == "text"
    ).strip()
    if not text:
        raise RuntimeError("Model returned an empty narrative.")
    return text
