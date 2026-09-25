"""Whydunit — a forensic investigation console for AI pipeline failures.

Run with: ``streamlit run app.py``

Layout: a sidebar picks the synthetic incident (scenario, window, seed);
the page selector walks the investigation workflow from dashboard to
case-file export. All telemetry is synthetic and seeded; the forensic
engine never sees ground truth — the "Ground Truth (Eval)" page is the
one place the answer key is shown, and it says so loudly.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pandas as pd
import streamlit as st

from whydunit.evaluation import run_evaluation
from whydunit.forensics import InvestigationResult, case_file, investigate
from whydunit.models import (
    DEFAULT_DISPLAY_TZ,
    ConfidenceBand,
    HealthState,
    IncidentCategory,
    SignalRef,
)
from whydunit.simulator import generate_dataset, to_wide
from whydunit.simulator.generator import scenario_to_dict, truth_to_dict
from whydunit.simulator.pipeline import all_signals
from whydunit.viz import (
    compare_signal,
    correlation_heatmap,
    evidence_graph_figure,
    pipeline_figure,
    signal_timeline,
)

START = datetime(2026, 9, 1, tzinfo=UTC)
DISPLAY_TZ = DEFAULT_DISPLAY_TZ  # Asia/Kolkata per product spec

PAGES = (
    "Dashboard",
    "Incident Explorer",
    "Pipeline View",
    "Timeline",
    "Signal Explorer",
    "Forensic Analysis",
    "Evidence Graph",
    "Incident Comparison",
    "Investigation Notes & Case File",
    "Ground Truth (Eval)",
)

CONFIDENCE_BADGE = {
    ConfidenceBand.STRONG: "🟢 strong",
    ConfidenceBand.MODERATE: "🟡 moderate",
    ConfidenceBand.WEAK: "🟠 weak",
    ConfidenceBand.INSUFFICIENT: "⚪ insufficient",
}
HEALTH_BADGE = {
    HealthState.HEALTHY: "🟢",
    HealthState.WARNING: "🟡",
    HealthState.DEGRADED: "🟠",
    HealthState.FAILED: "🔴",
}


@st.cache_data(show_spinner="Generating telemetry and investigating…")
def load_case(category_value: str, days: float, seed: int):
    """Generate one synthetic incident and run the full investigation."""
    dataset = generate_dataset(IncidentCategory(category_value), start=START, days=days, seed=seed)
    wide = to_wide(dataset.telemetry)
    result = investigate(wide)
    return wide, result, scenario_to_dict(dataset.spec), truth_to_dict(dataset.truth)


@st.cache_data(show_spinner="Running the engine against all 12 scenarios…")
def load_evaluation(days: float, seed: int):
    return run_evaluation(days=days, seed=seed).to_markdown()


def fmt(moment: datetime) -> str:
    from zoneinfo import ZoneInfo

    return moment.astimezone(ZoneInfo(DISPLAY_TZ)).strftime("%Y-%m-%d %H:%M %Z")


def anomalies_frame(result: InvestigationResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal": [str(a.signal) for a in result.anomalies],
            "method": [a.method for a in result.anomalies],
            "direction": [a.direction.value for a in result.anomalies],
            "peak |z|": [round(a.score, 1) for a in result.anomalies],
            "start": [fmt(a.start) for a in result.anomalies],
            "end": [fmt(a.end) for a in result.anomalies],
        }
    )


def render_dashboard(result: InvestigationResult, scenario: dict) -> None:
    st.subheader("Dashboard")
    incident = result.is_incident
    top = result.hypotheses[0]

    one, two, three, four = st.columns(4)
    one.metric("Status", "INCIDENT" if incident else "NORMAL")
    two.metric("Severity", result.severity.value.upper())
    three.metric("Anomaly windows", len(result.anomalies))
    four.metric(
        "Affected stages", sum(1 for s in result.health.values() if s is not HealthState.HEALTHY)
    )

    st.markdown(
        f"**Suspected root cause:** {top.category.value} ({CONFIDENCE_BADGE[top.confidence]})"
    )
    if incident:
        st.markdown(f"**Incident window:** {fmt(result.window_start)} → {fmt(result.window_end)}")
    st.plotly_chart(pipeline_figure(result.health), use_container_width=True)
    st.caption(
        f"Synthetic scenario loaded: {scenario['name']} — investigate as if you did not know."
    )


def render_incident_explorer(result: InvestigationResult, scenario: dict) -> None:
    st.subheader("Incident Explorer")
    st.markdown(f"### {scenario['name']}")
    st.markdown(scenario["description"])
    left, right = st.columns(2)
    left.markdown(f"**Incident ID:** `{scenario['id']}`")
    left.markdown(f"**Declared severity:** {scenario['severity']}")
    right.markdown(f"**Starts:** {fmt(datetime.fromisoformat(scenario['start']))}")
    right.markdown(f"**Duration:** {timedelta(seconds=scenario['duration_seconds'])}")
    st.markdown("**Symptoms (derived from telemetry, not the scenario):**")
    for symptom in result.symptoms or ("No anomalous symptoms detected.",):
        st.markdown(f"- {symptom}")


def render_pipeline_view(result: InvestigationResult) -> None:
    st.subheader("Pipeline View")
    st.plotly_chart(pipeline_figure(result.health), use_container_width=True)
    rows = [
        {"stage": stage.value, "health": f"{HEALTH_BADGE[state]} {state.value}"}
        for stage, state in result.health.items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _signal_options(default_signals: list[SignalRef]) -> list[SignalRef]:
    catalogue = list(all_signals())
    labels = [str(s) for s in catalogue]
    default_labels = [str(s) for s in default_signals if str(s) in labels]
    chosen = st.multiselect("Signals", labels, default=default_labels or labels[:3])
    return [SignalRef.parse(label) for label in chosen]


def render_timeline(wide: pd.DataFrame, result: InvestigationResult) -> None:
    st.subheader("Timeline")
    st.caption("Red shading: anomaly windows. Dotted lines: detected level shifts.")
    anomalous = list(dict.fromkeys(a.signal for a in result.anomalies))
    signals = _signal_options(anomalous[:4])
    if signals:
        st.plotly_chart(
            signal_timeline(
                wide,
                signals,
                anomalies=result.anomalies,
                changepoints=result.changepoints,
                display_tz=DISPLAY_TZ,
            ),
            use_container_width=True,
        )


def render_signal_explorer(wide: pd.DataFrame, result: InvestigationResult) -> None:
    st.subheader("Signal Explorer")
    signals = _signal_options([])
    if signals:
        st.plotly_chart(
            signal_timeline(
                wide,
                signals,
                anomalies=result.anomalies,
                changepoints=result.changepoints,
                display_tz=DISPLAY_TZ,
            ),
            use_container_width=True,
        )
    with st.expander("Raw anomaly table"):
        st.dataframe(anomalies_frame(result), use_container_width=True, hide_index=True)


def render_forensic_analysis(wide: pd.DataFrame, result: InvestigationResult) -> None:
    st.subheader("Forensic Analysis")

    st.markdown("#### Correlated signal clusters")
    if not result.clusters:
        st.info("No correlated clusters — no incident evidence.")
    for i, cluster in enumerate(result.clusters):
        st.markdown(
            f"**Cluster {i + 1}** — {len(cluster.signals)} signals, "
            f"strength {cluster.strength:.2f}, "
            f"{fmt(cluster.start)} → {fmt(cluster.end)}"
        )
        st.markdown("Order of first anomaly: " + " → ".join(f"`{ref}`" for ref in cluster.ordering))
        if len(cluster.signals) >= 2:
            st.plotly_chart(
                correlation_heatmap(
                    wide, list(cluster.signals), start=cluster.start, end=cluster.end
                ),
                use_container_width=True,
            )

    st.markdown("#### Root-cause candidates")
    for rank, hypothesis in enumerate(result.hypotheses, start=1):
        with st.expander(
            f"{rank}. {hypothesis.category.value} — "
            f"{CONFIDENCE_BADGE[hypothesis.confidence]} (score {hypothesis.score})",
            expanded=(rank == 1),
        ):
            st.markdown(hypothesis.statement)
            st.markdown("**Supporting evidence:**")
            for item in hypothesis.supporting or ():
                st.markdown(f"- [{item.kind.value}] {item.statement}")
            if not hypothesis.supporting:
                st.markdown("- (none)")
            st.markdown("**Contradicting evidence:**")
            for item in hypothesis.contradicting or ():
                st.markdown(f"- [{item.kind.value}] {item.statement}")
            if not hypothesis.contradicting:
                st.markdown("- (none)")
            if hypothesis.next_step:
                st.markdown(f"**Next investigation step:** {hypothesis.next_step}")


def render_evidence_graph(result: InvestigationResult) -> None:
    st.subheader("Evidence Graph")
    st.caption(
        "Left: anomalous signals. Middle: evidence items. Right: ranked hypotheses. "
        "Green edges support; red edges contradict."
    )
    st.plotly_chart(evidence_graph_figure(result), use_container_width=True)


def render_comparison(category_value: str, days: float, seed: int) -> None:
    st.subheader("Incident Comparison")
    other_value = st.selectbox(
        "Compare against",
        [c.value for c in IncidentCategory],
        index=list(IncidentCategory).index(IncidentCategory.NORMAL),
    )
    wide_a, result_a, _, _ = load_case(category_value, days, seed)
    wide_b, result_b, _, _ = load_case(other_value, days, seed)

    left, right = st.columns(2)
    left.markdown(f"**A: {category_value}** — top: {result_a.hypotheses[0].category.value}")
    right.markdown(f"**B: {other_value}** — top: {result_b.hypotheses[0].category.value}")

    labels = [str(s) for s in all_signals()]
    default = "retrieval.top_k_similarity"
    chosen = st.selectbox("Signal", labels, index=labels.index(default))
    st.plotly_chart(
        compare_signal(
            wide_a,
            wide_b,
            SignalRef.parse(chosen),
            labels=(category_value, other_value),
            display_tz=DISPLAY_TZ,
        ),
        use_container_width=True,
    )


def render_notes_and_casefile(result: InvestigationResult, scenario: dict) -> None:
    st.subheader("Investigation Notes & Case File")
    notes = st.text_area(
        "Notes (observations, confirmed/rejected hypotheses, remediation)",
        value=st.session_state.get("notes", ""),
        height=220,
    )
    st.session_state["notes"] = notes

    case = case_file(result, notes=notes, incident_id=scenario["id"])
    st.markdown("#### Preview")
    st.markdown(case.to_markdown())

    st.markdown("#### AI narrative report (optional)")
    st.caption(
        "Written by an LLM that may only cite the deterministic evidence in this "
        "case file. Narratives that fail citation validation are refused, never shown."
    )
    narrative_key = f"narrative-{scenario['id']}-{case.case_id}"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.info(
            "To enable: install the extra (`uv sync --extra llm`) and set "
            "`ANTHROPIC_API_KEY` in the environment before starting the app."
        )
    elif st.button("Generate narrative"):
        from whydunit.explain import NarrativeValidationError, narrate

        try:
            with st.spinner("Generating and validating narrative…"):
                report = narrate(case)
        except NarrativeValidationError as error:
            st.error(
                "Narrative refused by the validator after "
                f"{error.attempts} attempt(s): " + "; ".join(error.problems)
            )
        except Exception as error:
            st.error(f"Narrative generation failed: {error}")
        else:
            st.session_state[narrative_key] = report

    report = st.session_state.get(narrative_key)
    if report is not None:
        st.success(
            f"Validated on attempt {report.attempts}; "
            f"cites {len(report.validation.cited_ids)} evidence IDs."
        )
        st.markdown(report.text)

    st.markdown("#### Export")
    left, right = st.columns(2)
    left.download_button(
        "Download Markdown",
        case.to_markdown(),
        file_name=f"{case.case_id}.md",
        mime="text/markdown",
    )
    right.download_button(
        "Download JSON",
        case.to_json(),
        file_name=f"{case.case_id}.json",
        mime="application/json",
    )
    st.caption("If the browser or corporate policy blocks downloads, save server-side instead:")
    if st.button("Save to data/exports/"):
        from pathlib import Path

        export_dir = Path("data") / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        base = f"{case.case_id}-{case.incident_id}"
        md_path = export_dir / f"{base}.md"
        json_path = export_dir / f"{base}.json"
        md_path.write_text(case.to_markdown(), encoding="utf-8")
        json_path.write_text(case.to_json(), encoding="utf-8")
        saved = [md_path, json_path]
        if report is not None:
            narrative_path = export_dir / f"{base}-narrative.md"
            narrative_path.write_text(report.text + "\n", encoding="utf-8")
            saved.append(narrative_path)
        st.success("Saved: " + ", ".join(str(path.resolve()) for path in saved))


def render_ground_truth(truth: dict, days: float, seed: int) -> None:
    st.subheader("Ground Truth (Eval)")
    st.warning(
        "Developer-facing evaluation area. This is the synthetic answer key — "
        "it exists to score the forensic engine, never to aid an investigation.",
        icon="⚠️",
    )
    st.json(truth)
    st.markdown("#### Engine report card (all 12 scenarios, this window/seed)")
    if st.button("Run evaluation"):
        st.markdown(load_evaluation(days, seed))


def main() -> None:
    st.set_page_config(page_title="Whydunit", page_icon="🔍", layout="wide")
    st.title("🔍 Whydunit")
    st.caption("Find the *why* behind AI pipeline failures — synthetic, offline, deterministic.")

    with st.sidebar:
        st.header("Case setup")
        category_value = st.selectbox(
            "Incident scenario", [c.value for c in IncidentCategory], index=4
        )
        days = st.slider("Window (days)", 0.5, 3.0, 1.0, 0.5)
        seed = st.number_input("Seed", min_value=0, max_value=9999, value=42, step=1)
        page = st.radio("Investigation", PAGES)
        st.caption(f"Display timezone: {DISPLAY_TZ} (internal: UTC)")

    wide, result, scenario, truth = load_case(category_value, days, int(seed))

    if page == "Dashboard":
        render_dashboard(result, scenario)
    elif page == "Incident Explorer":
        render_incident_explorer(result, scenario)
    elif page == "Pipeline View":
        render_pipeline_view(result)
    elif page == "Timeline":
        render_timeline(wide, result)
    elif page == "Signal Explorer":
        render_signal_explorer(wide, result)
    elif page == "Forensic Analysis":
        render_forensic_analysis(wide, result)
    elif page == "Evidence Graph":
        render_evidence_graph(result)
    elif page == "Incident Comparison":
        render_comparison(category_value, days, int(seed))
    elif page == "Investigation Notes & Case File":
        render_notes_and_casefile(result, scenario)
    else:
        render_ground_truth(truth, days, int(seed))


if __name__ == "__main__":
    main()
