"""Validation of generated narratives against the case file.

The narrative generator asks for prose in which every factual claim
cites an evidence ID. This module is the enforcement: it parses the
citations back out and rejects the narrative if it cites an ID the
case file does not contain, or makes claims with no citation at all.

Deterministic, dependency-free — the validator runs whether or not the
``llm`` extra is installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from whydunit.explain.narrative import evidence_registry
from whydunit.models import CaseFile

CITATION_RE = re.compile(r"\[([A-Za-z]+-[A-Za-z0-9_.:-]+)\]")
NEXT_STEPS_RE = re.compile(r"what to do next", re.IGNORECASE)
_SNIPPET_LENGTH = 80


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of checking one narrative against one case file."""

    valid: bool
    cited_ids: tuple[str, ...]
    unknown_ids: tuple[str, ...]
    uncited_paragraphs: tuple[str, ...]

    @property
    def problems(self) -> tuple[str, ...]:
        """Human-readable list of everything wrong (empty when valid)."""
        issues: list[str] = []
        if not self.cited_ids:
            issues.append("narrative cites no evidence at all")
        issues += [f"cites unknown evidence ID: {item}" for item in self.unknown_ids]
        issues += [f'uncited claim: "{item}..."' for item in self.uncited_paragraphs]
        return tuple(issues)


def validate_narrative(narrative: str, case: CaseFile) -> ValidationResult:
    """Check a narrative's citations against the case file's evidence."""
    known = set(evidence_registry(case))

    cited: list[str] = []
    for match in CITATION_RE.finditer(narrative):
        token = match.group(1)
        if token not in cited:
            cited.append(token)
    unknown = tuple(token for token in cited if token not in known)

    uncited: list[str] = []
    in_next_steps = False
    for block in re.split(r"\n\s*\n", narrative):
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        body_lines: list[str] = []
        for line in lines:
            if line.startswith("#"):
                in_next_steps = bool(NEXT_STEPS_RE.search(line))
                continue
            body_lines.append(line)
        if not body_lines or in_next_steps:
            continue
        body = " ".join(body_lines)
        if not CITATION_RE.search(body):
            uncited.append(body[:_SNIPPET_LENGTH])

    return ValidationResult(
        valid=bool(cited) and not unknown and not uncited,
        cited_ids=tuple(cited),
        unknown_ids=unknown,
        uncited_paragraphs=tuple(uncited),
    )
