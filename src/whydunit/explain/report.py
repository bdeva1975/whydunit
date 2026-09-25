"""The narrate() facade: generate, validate, retry once, or refuse.

This is the only entry point the console and CLI use. A narrative that
still fails validation after the retry is not shown to anyone — the
error carries the validator's findings instead. The LLM gets a voice,
never authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from whydunit.explain.narrative import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, generate_narrative
from whydunit.explain.validator import ValidationResult, validate_narrative
from whydunit.models import CaseFile


class NarrativeValidationError(RuntimeError):
    """Raised when the narrative still fails validation after retrying."""

    def __init__(self, problems: tuple[str, ...], attempts: int) -> None:
        self.problems = problems
        self.attempts = attempts
        listing = "; ".join(problems)
        super().__init__(f"Narrative failed validation after {attempts} attempt(s): {listing}")


@dataclass(frozen=True, slots=True)
class NarrativeReport:
    """A validated narrative, with its validation record and attempt count."""

    text: str
    validation: ValidationResult
    attempts: int


def _feedback(problems: tuple[str, ...], previous: str) -> str:
    listing = "\n".join(f"- {problem}" for problem in problems)
    return (
        "Your previous attempt failed validation:\n"
        f"{listing}\n\n"
        "Previous attempt:\n"
        f"{previous}\n\n"
        "Rewrite the full narrative from scratch, fixing every problem. "
        "All original rules still apply."
    )


def narrate(
    case: CaseFile,
    client: Any | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_attempts: int = 2,
) -> NarrativeReport:
    """Generate a narrative and enforce validation, retrying once by default."""
    correction: str | None = None
    result: ValidationResult | None = None
    for attempt in range(1, max_attempts + 1):
        text = generate_narrative(
            case,
            client=client,
            model=model,
            max_tokens=max_tokens,
            correction=correction,
        )
        result = validate_narrative(text, case)
        if result.valid:
            return NarrativeReport(text=text, validation=result, attempts=attempt)
        correction = _feedback(result.problems, text)

    assert result is not None  # max_attempts >= 1
    raise NarrativeValidationError(result.problems, max_attempts)
