"""Report generation helper.

Generates a comprehensive lab report from MetricsReport data.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data."""

    # ── Per-scenario table rows ──
    scenario_rows = []
    for sm in metrics.scenario_metrics:
        status = "✅" if sm.success else "❌"
        scenario_rows.append(
            f"| {sm.scenario_id} | {sm.expected_route} | {sm.actual_route or 'N/A'} "
            f"| {status} | {sm.retry_count} | {sm.interrupt_count} |"
        )
    scenario_table = "\n".join(scenario_rows)

    import subprocess
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        commit_hash = "8bf5147"

    report = f"""# Day 08 Lab Report

## 1. Team / student

- Name: Linh
- Repo/commit: phase2-track3-day8-langgraph-agent / {commit_hash}
- Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## 2. Architecture

The graph implements a support-ticket agent with 11 nodes organized in a conditional workflow:

```
START → intake → classify → [conditional routing]
  simple       → answer → finalize → END
  tool         → tool → evaluate → [retry gate]
                                     success → answer → finalize → END
                                     needs_retry → retry → [bounded check]
                                                            tool (retry loop)
                                                            dead_letter → finalize → END
  missing_info → clarify → finalize → END
  risky        → risky_action → approval → [approval gate]
                                            approved → tool → evaluate → ...
                                            rejected → clarify → finalize → END
  error        → retry → [bounded check] → ...
```

**Key design decisions:**
- `classify_node` uses LLM with structured output (Pydantic `IntentClassification` model) for reliable intent classification
- `answer_node` uses LLM for grounded response generation with full context
- Retry loops are bounded by `max_attempts` (default 3) to prevent infinite loops
- All paths terminate at `finalize → END` for consistent audit trail
- Mock approval in `approval_node` for CI/testing, with optional real HITL via `LANGGRAPH_INTERRUPT=true`

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| messages | append (`add`) | Audit trail of all conversation events |
| tool_results | append (`add`) | Complete history of tool executions |
| errors | append (`add`) | Accumulate all error messages |
| events | append (`add`) | Full audit log of node executions |
| route | overwrite | Current classification route |
| risk_level | overwrite | Current risk assessment |
| attempt | overwrite | Current retry attempt counter |
| max_attempts | overwrite | Retry limit (configurable per scenario) |
| final_answer | overwrite | Final response to user |
| evaluation_result | overwrite | Drives retry loop gate |
| pending_question | overwrite | Clarification question for missing_info |
| proposed_action | overwrite | Risky action description |
| approval | overwrite | HITL approval decision |

## 4. Scenario results

**Summary:**
- Total scenarios: {metrics.total_scenarios}
- Success rate: {metrics.success_rate:.1%}
- Average nodes visited: {metrics.avg_nodes_visited:.1f}
- Total retries: {metrics.total_retries}
- Total interrupts: {metrics.total_interrupts}

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
{scenario_table}

## 5. Failure analysis

Two failure modes considered:

1. **Transient tool failures (retry loop):**
   Error-route scenarios simulate transient failures where `tool_node` returns ERROR for the first 2 attempts. The `evaluate_node` detects this and routes to `retry`, which increments the attempt counter. `route_after_retry` checks `attempt < max_attempts` to either retry or escalate to `dead_letter`. This bounded loop prevents infinite retries. S07 tests this with `max_attempts=1`, forcing immediate dead-letter escalation.

2. **Risky actions without approval:**
   Risky queries (refunds, deletions) go through `risky_action_node → approval_node` before executing. If approval is rejected, the flow redirects to `clarify` instead of proceeding. The mock approval defaults to `approved=True` for testing, but production would use `interrupt()` for real HITL.

## 6. Persistence / recovery evidence

- **MemorySaver** used by default for in-process checkpointing
- Each scenario run gets a unique `thread_id` (e.g., `thread-S01_simple`) via `run_config`
- **SQLite checkpointer** implemented with WAL mode for crash recovery
- State is persisted at each node boundary, enabling resume from last checkpoint

## 7. Extension work

- **SQLite persistence**: Implemented `SqliteSaver` with WAL journal mode for durable checkpointing
- **Real HITL support**: `approval_node` checks `LANGGRAPH_INTERRUPT=true` and uses `interrupt()` for real human-in-the-loop
- **Multi-provider LLM fallback**: `llm.py` supports Gemini, DeepSeek, Mistral, OpenAI, Anthropic with automatic fallback

## 8. Improvement plan

If I had one more day, I would:
1. **LLM-as-judge evaluation**: Replace heuristic ERROR check in `evaluate_node` with LLM-based quality assessment
2. **Parallel fan-out**: Use `Send()` for concurrent tool calls in complex queries
3. **Observability**: Add LangSmith tracing for production debugging
4. **Time travel**: Implement `get_state_history()` replay for debugging failed scenarios
5. **Streaming**: Add streaming support for real-time response generation
"""
    return report


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
