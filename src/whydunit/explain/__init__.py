"""Optional LLM explanation layer for Whydunit.

Turns a finished CaseFile into a narrative incident report. The LLM
explains deterministic evidence; it never generates it. Every claim in
the narrative must cite an evidence ID present in the case file, and
the validator rejects anything that does not.

Requires the optional dependency group:  uv sync --extra llm
The forensic engine itself never imports this package.
"""
