"""The forensic case file: an exportable record of one investigation.

Internal timestamps are UTC; rendering converts to a display timezone
(IST by default, per the product spec). JSON export is lossless enough
to re-load; Markdown export is the human-readable artefact.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

from whydunit.models.enums import Severity
from whydunit.models.hypothesis import Hypothesis
from whydunit.models.signal import Anomaly, CorrelationCluster

DEFAULT_DISPLAY_TZ = "Asia/Kolkata"


def _jsonify(value: Any) -> Any:
    """Recursively convert enums and datetimes for JSON serialisation."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class CaseFile:
    """Everything an engineer needs to hand over an investigation."""

    case_id: str
    incident_id: str
    pipeline_name: str
    window_start: datetime
    window_end: datetime
    severity: Severity
    summary: str
    symptoms: tuple[str, ...] = field(default=())
    anomalies: tuple[Anomaly, ...] = field(default=())
    correlations: tuple[CorrelationCluster, ...] = field(default=())
    hypotheses: tuple[Hypothesis, ...] = field(default=())
    notes: str = ""
    recommended_actions: tuple[str, ...] = field(default=())
    exported_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python dict with enums/datetimes flattened to strings."""
        return _jsonify(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self, display_tz: str = DEFAULT_DISPLAY_TZ) -> str:
        """Render the case file as a human-readable Markdown document."""
        tz = ZoneInfo(display_tz)

        def fmt(moment: datetime) -> str:
            return moment.astimezone(tz).strftime("%Y-%m-%d %H:%M %Z")

        lines: list[str] = [
            "# WHYDUNIT FORENSIC CASE FILE",
            "",
            f"**Case ID:** {self.case_id}",
            f"**Incident ID:** {self.incident_id}",
            f"**Pipeline:** {self.pipeline_name}",
            f"**Incident window:** {fmt(self.window_start)} – {fmt(self.window_end)}",
            f"**Severity:** {self.severity.value}",
            f"**Exported:** {fmt(self.exported_at)}",
            "",
            "## Summary",
            "",
            self.summary,
            "",
            "## Primary symptoms",
            "",
        ]
        lines += [f"- {item}" for item in self.symptoms] or ["- (none recorded)"]

        lines += ["", "## Detected anomalies", ""]
        lines += [
            f"- `{anomaly.signal}` moved {anomaly.direction.value} "
            f"({anomaly.method}, score {anomaly.score:.1f}), "
            f"{fmt(anomaly.start)} – {fmt(anomaly.end)}"
            for anomaly in self.anomalies
        ] or ["- (none recorded)"]

        lines += ["", "## Correlated signal clusters", ""]
        lines += [
            f"- {len(cluster.signals)} signals moved together "
            f"(strength {cluster.strength:.2f}), earliest first: "
            + ", ".join(f"`{ref}`" for ref in cluster.ordering)
            for cluster in self.correlations
        ] or ["- (none recorded)"]

        lines += ["", "## Root-cause candidates", ""]
        if not self.hypotheses:
            lines.append("- (none generated)")
        for rank, hypothesis in enumerate(self.hypotheses, start=1):
            lines += [
                f"### {rank}. {hypothesis.statement}",
                "",
                f"**Category:** {hypothesis.category.value} · "
                f"**Confidence:** {hypothesis.confidence.value} · "
                f"**Score:** {hypothesis.score:.2f}",
                "",
                "Supporting evidence:",
            ]
            lines += [
                f"- [{item.kind.value}] {item.statement}" for item in hypothesis.supporting
            ] or ["- (none)"]
            lines.append("Contradicting evidence:")
            lines += [
                f"- [{item.kind.value}] {item.statement}" for item in hypothesis.contradicting
            ] or ["- (none)"]
            if hypothesis.next_step:
                lines += ["", f"**Next investigation step:** {hypothesis.next_step}"]
            lines.append("")

        lines += ["## Investigation notes", "", self.notes or "(none)", ""]

        lines += ["## Recommended next actions", ""]
        lines += [f"- {item}" for item in self.recommended_actions] or ["- (none)"]

        lines += [
            "",
            "---",
            "",
            "_Confidence reflects evidence strength on the available telemetry;_",
            "_it is not proof of causality._",
            "",
        ]
        return "\n".join(lines)
