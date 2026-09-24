"""Incident scenario definitions and ground-truth labels.

``IncidentScenario`` is the *recipe* the simulator executes.
``GroundTruth`` is the answer key: it exists ONLY for the evaluation
harness and the dev-facing eval page. Forensic-engine modules must never
import it — a test enforces that boundary.

``primary_signal`` and ``root_cause_stage`` are ``None`` exactly for the
NORMAL scenario: healthy operation has no fault to point at, and the
forensic engine is scored on recognising that too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from whydunit.models.enums import IncidentCategory, Severity, Stage
from whydunit.models.signal import SignalRef


@dataclass(frozen=True, slots=True)
class IncidentScenario:
    """A synthetic incident recipe, per the product spec.

    ``start`` is timezone-aware UTC. ``primary_signal`` is where the fault
    is injected first (``None`` for NORMAL); ``secondary_signals`` are the
    knock-on effects the scenario also perturbs, in propagation order.
    """

    id: str
    name: str
    category: IncidentCategory
    description: str
    severity: Severity
    affected_stages: tuple[Stage, ...]
    start: datetime
    duration: timedelta
    primary_signal: SignalRef | None
    secondary_signals: tuple[SignalRef, ...] = field(default=())
    expected_root_cause: str = ""


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """The hidden answer key for one generated dataset. Eval-only.

    ``root_cause_stage`` is where the fault originated (which may differ
    from where symptoms are loudest); it is ``None`` for NORMAL.
    ``injected_signals`` lists every signal the scenario deliberately
    perturbed (empty for NORMAL).
    """

    scenario_id: str
    category: IncidentCategory
    root_cause_stage: Stage | None
    incident_start: datetime
    incident_end: datetime
    injected_signals: tuple[SignalRef, ...]
