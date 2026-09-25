"""Narrative incident reports, generated from a finished case file.

The LLM writes prose about the LEADING hypothesis only; the case file
supplies every fact. The prompt embeds that hypothesis's evidence with
stable IDs and instructs the model to cite an ID for every factual
claim. ``validator`` rejects narratives that cite unknown IDs or make
uncited claims. Alternative hypotheses are never given to the model:
``report.narrate()`` appends the engine's ranking of them as a
deterministic, machine-written section instead.

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
    Used by the validator: any of these IDs is a legal citation.
    """
    registry: dict[str, Evidence] = {}
    for hypothesis in case.hypotheses:
        for item in (*hypothesis.supporting, *hypothesis.contradicting):
            registry.setdefault(item.id, item)
    return dict(sorted(registry.items()))


def leading_evidence(case: CaseFile) -> dict[str, Evidence]:
    """Evidence attached to the leading hypothesis only — the prompt's registry."""
    if not case.hypotheses:
        return {}
    top = case.hypotheses[0]
    registry = {item.id: item for item in (*top.supporting, *top.contradicting)}
    return dict(sorted(registry.items()))


def build_prompt(case: CaseFile, correction: str | None = None) -> str:
    """The single-turn prompt: case facts, leading-hypothesis evidence, rules.

    ``correction`` carries validator feedback on a retry; it is appended
    as an explicit fix-list so the model rewrites rather than continues.
    """
    top = case.hypotheses[0]
    registry = leading_evidence(case)
    example_id = next(iter(registry), "EVIDENCE-ID")

    evidence_lines = [
        f"[{item.id}] ({item.kind.value}) {item.statement} "
        f"(signals: {', '.join(str(ref) for ref in item.signals)})"
        for item in registry.values()
    ]

    supporting = ", ".join(item.id for item in top.supporting) or "none"
    contradicting = ", ".join(item.id for item in top.contradicting) or "none"
    hypothesis_line = (
        f"{top.statement} "
        f"[category: {top.category.value}, "
        f"confidence: {top.confidence.value}, "
        f"supporting: {supporting}, contradicting: {contradicting}]"
    )

    lines = [
        "You are writing the narrative section of a forensic incident report",
        "for an AI pipeline. A deterministic forensic engine has already done",
        "the analysis; your job is ONLY to explain its leading hypothesis in",
        "clear prose.",
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
        "## Leading hypothesis (the ONLY hypothesis you may discuss)",
        hypothesis_line,
        "",
        "## Rules",
        "1. Cite evidence using ONLY IDs from the evidence registry above,",
        "   in square brackets, copied character-for-character — for",
        f"   example [{example_id}]. Never invent, shorten, prefix, or",
        "   reformat an ID.",
        "2. EVERY paragraph in 'What happened' and 'Why the engine believes",
        "   this' must contain at least one citation. This includes the",
        "   opening paragraph and any sentence that restates the engine's",
        "   hypothesis, summary, severity, or the incident window: fold",
        "   those facts into sentences that also cite the evidence",
        "   supporting them. A paragraph with no citation is a validation",
        "   failure.",
        "3. Discuss ONLY the leading hypothesis above. Do NOT mention, name,",
        "   or allude to any other candidate cause — the engine's ranking of",
        "   alternatives is appended to your narrative automatically.",
        "4. Do not introduce any number, signal name, cause, or event that",
        "   is not in the registry or case facts. No speculation.",
        "5. Do not upgrade the engine's confidence: describe the hypothesis",
        "   using its stated confidence band, and mention that contradicting",
        "   evidence exists where it does.",
        "6. Structure: exactly three sections with these Markdown headings and",
        "   nothing else - 'What happened', 'Why the engine believes this',",
        "   'What to do next'. No introduction, no conclusion, no extra",
        "   sections. At most 400 words.",
        "7. Do not add a disclaimer section; the case file carries one.",
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
    if not leading_evidence(case):
        raise ValueError(
            "The leading hypothesis carries no evidence; there is nothing for "
            "a narrative to cite. Narratives are only generated for "
            "investigated incidents."
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
