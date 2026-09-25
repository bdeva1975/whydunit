"""Optional LLM explanation layer for Whydunit.

Turns a finished CaseFile into a narrative incident report. The LLM
explains deterministic evidence; it never generates it. Every claim in
the narrative must cite an evidence ID present in the case file, and
the validator rejects anything that does not.

Requires the optional dependency group:  uv sync --extra llm
The forensic engine itself never imports this package.

Entry point:

    from whydunit.explain import narrate
    report = narrate(case_file)   # needs ANTHROPIC_API_KEY
"""

from whydunit.explain.narrative import (
    DEFAULT_MODEL,
    build_prompt,
    evidence_registry,
    generate_narrative,
)
from whydunit.explain.report import NarrativeReport, NarrativeValidationError, narrate
from whydunit.explain.validator import ValidationResult, validate_narrative

__all__ = [
    "DEFAULT_MODEL",
    "NarrativeReport",
    "NarrativeValidationError",
    "ValidationResult",
    "build_prompt",
    "evidence_registry",
    "generate_narrative",
    "narrate",
    "validate_narrative",
]
