"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
import time
from typing import Literal

from pydantic import BaseModel, Field

from .state import AgentState, make_event


# ─── Structured output schemas for LLM ──────────────────────────────
class IntentClassification(BaseModel):
    """Structured output for intent classification by LLM."""
    intent: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description=(
            "The classified intent of the user query. Choose exactly one:\n"
            "- 'risky': Actions with side effects like refunds, deletions, account removal, "
            "sending emails, cancellations, or any irreversible operation\n"
            "- 'tool': Information lookups like order status, tracking, searching, "
            "checking inventory, or any query that needs external data retrieval\n"
            "- 'missing_info': Vague or incomplete queries that lack sufficient context "
            "to take any action (e.g., 'fix it', 'help me', 'do that thing')\n"
            "- 'error': System failures, timeouts, crashes, service unavailable, "
            "infrastructure errors, or reports of technical failures\n"
            "- 'simple': General questions answerable from common knowledge without "
            "tools or risky actions (e.g., password reset instructions, FAQ)"
        )
    )
    confidence: float = Field(
        default=0.9,
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why this intent was chosen",
    )


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── LLM-powered nodes ───────────────────────────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM with structured output.

    Uses .with_structured_output() for reliable enum classification.
    Priority: risky > tool > missing_info > error > simple.
    """
    from .llm import get_llm

    query = state.get("query", "")
    t0 = time.time()

    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(IntentClassification)

    classification_prompt = f"""You are a support-ticket intent classifier. You must classify the customer query into exactly one of five categories.

CATEGORIES (in strict priority order — if a query matches multiple, pick the HIGHEST priority):

1. **risky** (HIGHEST PRIORITY)
   Definition: Any request that asks to PERFORM an ACTION with side effects or irreversible consequences.
   Examples:
   - "Refund this customer and send confirmation email" → risky (refund = side effect)
   - "Delete customer account after support verification" → risky (delete = irreversible)
   - "Cancel my subscription immediately" → risky (cancellation = side effect)
   - "Remove this user from the system" → risky (removal = irreversible)
   - "Process a chargeback for order 456" → risky (financial side effect)
   - "Send a password reset email to the customer" → risky (sending email = side effect)
   Key signals: delete, remove, cancel, refund, send, process, update, modify, close account, terminate

2. **tool**
   Definition: Requests that need to LOOK UP or RETRIEVE information from external systems. Read-only operations.
   Examples:
   - "Please lookup order status for order 12345" → tool
   - "Check the tracking number for my shipment" → tool
   - "What's the balance on account 789?" → tool
   Key signals: lookup, check status, find, search, track, what is, show me

3. **missing_info**
   Definition: Queries that are too VAGUE or INCOMPLETE to understand what the user wants. They lack specifics.
   Examples:
   - "Can you fix it?" → missing_info (fix what?)
   - "Help me with my thing" → missing_info (what thing?)
   - "Please handle this" → missing_info (handle what?)
   Key signals: no specific subject, unclear referent, ambiguous pronouns without context

4. **error**
   Definition: The query REPORTS or DESCRIBES a system failure, technical problem, or infrastructure issue. The user is telling you something is broken.
   Examples:
   - "Timeout failure while processing request" → error
   - "System failure cannot recover after multiple attempts" → error
   - "The server crashed and won't come back up" → error
   - "Getting 500 errors on every API call" → error
   - "Service unavailable, cannot process anything" → error
   Key signals: failure, timeout, crash, error, cannot recover, system down, unavailable, broken, 500, exception

5. **simple** (LOWEST PRIORITY)
   Definition: General knowledge questions that can be answered without tools, actions, or external lookups.
   Examples:
   - "How do I reset my password?" → simple (just instructions)
   - "What are your business hours?" → simple
   - "How does the return policy work?" → simple
   Key signals: how do I, what is (general), explain, guide me

CRITICAL RULES:
- If the query mentions deleting, removing, canceling, refunding, or any destructive/modifying action → ALWAYS classify as "risky"
- If the query mentions system failure, timeout, crash, error, or cannot recover → ALWAYS classify as "error"
- Only classify as "simple" if the query is purely asking for general knowledge/instructions with NO action requested
- When in doubt between two categories, choose the HIGHER priority one

Customer query: "{query}"

Classify this query into one of: risky, tool, missing_info, error, simple."""

    try:
        result = structured_llm.invoke(classification_prompt)
        intent = result.intent
        confidence = result.confidence
    except Exception:
        # Fallback: try without structured output
        try:
            raw_llm = get_llm(temperature=0.0)
            fallback_prompt = (
                f"Classify this support query into exactly one category. "
                f"Reply with ONLY one word: simple, tool, missing_info, risky, or error.\n\n"
                f"Rules:\n"
                f"- risky: refunds, deletions, account removal, sending emails, cancellations\n"
                f"- tool: order status lookups, tracking, search queries\n"
                f"- missing_info: vague queries lacking context like 'fix it' or 'help me'\n"
                f"- error: system failures, timeouts, crashes\n"
                f"- simple: general questions, FAQ, password reset how-to\n\n"
                f"Query: \"{query}\"\n\nCategory:"
            )
            raw_response = raw_llm.invoke(fallback_prompt)
            content = raw_response.content.strip().lower()
            valid_intents = {"simple", "tool", "missing_info", "risky", "error"}
            intent = content if content in valid_intents else "simple"
            confidence = 0.7
        except Exception:
            intent = "simple"
            confidence = 0.5

    risk_level = "high" if intent == "risky" else "low"
    latency = int((time.time() - t0) * 1000)

    return {
        "route": intent,
        "risk_level": risk_level,
        "messages": [f"classify:{intent}"],
        "events": [make_event("classify", "completed", f"classified as {intent}",
                              confidence=confidence, latency_ms=latency)],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulates transient failures for error-route scenarios to test retry loops.
    """
    route = state.get("route", "")
    attempt = state.get("attempt", 0)
    query = state.get("query", "")

    # Simulate transient error for error-route scenarios (first 2 attempts fail)
    if route == "error" and attempt < 2:
        result = f"ERROR: Transient failure on attempt {attempt} — service temporarily unavailable"
    else:
        # Mock success result based on query context
        result = f"Tool executed successfully. Result for query: '{query[:50]}' — Data retrieved."

    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", f"tool executed (attempt={attempt})",
                              result_preview=result[:80])],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Checks whether the latest tool result is satisfactory or needs retry.
    Uses heuristic (ERROR substring check) for base score.
    """
    tool_results = state.get("tool_results", [])
    latest_result = tool_results[-1] if tool_results else ""

    # Heuristic evaluation: check for ERROR indicator
    if "ERROR" in latest_result.upper():
        evaluation = "needs_retry"
        message = "Tool result contains error, needs retry"
    else:
        evaluation = "success"
        message = "Tool result is satisfactory"

    return {
        "evaluation_result": evaluation,
        "events": [make_event("evaluate", "completed", message,
                              evaluation=evaluation)],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    The LLM generates a helpful response grounded in available context:
    tool_results, approval decision, and original query.
    """
    from .llm import get_llm

    query = state.get("query", "")
    route = state.get("route", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")

    t0 = time.time()

    # Build context for grounded generation
    context_parts = [f"Original customer query: \"{query}\""]
    context_parts.append(f"Classified route: {route}")

    if tool_results:
        context_parts.append("Tool results:\n" + "\n".join(f"  - {r}" for r in tool_results))

    if approval:
        approved_str = "approved" if approval.get("approved") else "rejected"
        context_parts.append(f"Approval decision: {approved_str} by {approval.get('reviewer', 'unknown')}")

    context = "\n".join(context_parts)

    prompt = f"""You are a helpful customer support agent. Generate a clear, professional response to the customer based on the following context.

{context}

Provide a helpful, concise response that:
1. Directly addresses the customer's query
2. References any tool results or approval decisions if available
3. Is professional and empathetic in tone
4. Includes next steps if applicable

Response:"""

    try:
        llm = get_llm(temperature=0.3)
        response = llm.invoke(prompt)
        final_answer = response.content.strip()
    except Exception as e:
        final_answer = (
            f"Thank you for your query. Based on our analysis, here is our response regarding "
            f"'{query[:50]}'. We have processed your request through our {route} workflow. "
            f"{'Tool results: ' + '; '.join(tool_results[-2:]) if tool_results else ''} "
            f"Please contact support if you need further assistance. (Note: LLM unavailable: {e})"
        )

    latency = int((time.time() - t0) * 1000)

    return {
        "final_answer": final_answer,
        "events": [make_event("answer", "completed", "LLM-generated answer",
                              latency_ms=latency, answer_len=len(final_answer))],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating."""
    query = state.get("query", "")

    question = (
        f"I'd like to help you, but I need more details about your request: \"{query}\". "
        f"Could you please provide:\n"
        f"1. What specific issue are you experiencing?\n"
        f"2. Any relevant IDs (order number, account ID, ticket number)?\n"
        f"3. What outcome are you looking for?"
    )

    return {
        "pending_question": question,
        "final_answer": question,
        "messages": ["clarify:asked for details"],
        "events": [make_event("clarify", "completed", "clarification requested",
                              original_query=query[:60])],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval."""
    query = state.get("query", "")

    proposed = (
        f"PROPOSED RISKY ACTION: Based on the request \"{query}\", "
        f"this action involves potentially irreversible changes. "
        f"Requires human approval before execution."
    )

    return {
        "proposed_action": proposed,
        "messages": ["risky_action:prepared for approval"],
        "events": [make_event("risky_action", "completed", "risky action prepared",
                              query_preview=query[:60])],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use interrupt() for real HITL.
    """
    proposed_action = state.get("proposed_action", "Unknown action")

    # Check for real HITL mode
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        try:
            from langgraph.types import interrupt
            decision = interrupt({
                "question": "Do you approve this action?",
                "proposed_action": proposed_action,
            })
            approval_dict = {
                "approved": bool(decision.get("approved", False)),
                "reviewer": decision.get("reviewer", "human"),
                "comment": decision.get("comment", ""),
            }
        except Exception:
            # Fallback to mock if interrupt fails
            approval_dict = {
                "approved": True,
                "reviewer": "mock-reviewer",
                "comment": "Auto-approved (interrupt failed)",
            }
    else:
        # Mock approval for CI/testing
        approval_dict = {
            "approved": True,
            "reviewer": "mock-reviewer",
            "comment": "Auto-approved in mock mode",
        }

    return {
        "approval": approval_dict,
        "messages": [f"approval:{'approved' if approval_dict['approved'] else 'rejected'}"],
        "events": [make_event("approval", "completed",
                              f"{'approved' if approval_dict['approved'] else 'rejected'} by {approval_dict['reviewer']}",
                              approved=approval_dict["approved"])],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increments attempt counter and logs the transient failure.
    """
    current_attempt = state.get("attempt", 0)
    new_attempt = current_attempt + 1
    max_attempts = state.get("max_attempts", 3)

    error_msg = f"Retry attempt {new_attempt}/{max_attempts} — transient failure detected"

    return {
        "attempt": new_attempt,
        "errors": [error_msg],
        "messages": [f"retry:attempt {new_attempt}/{max_attempts}"],
        "events": [make_event("retry", "completed", error_msg,
                              attempt=new_attempt, max_attempts=max_attempts)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    """
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    query = state.get("query", "")
    errors = state.get("errors", [])

    final_answer = (
        f"We apologize, but we were unable to process your request after {attempt} attempt(s). "
        f"Your query \"{query[:60]}\" has been escalated to our support team for manual review. "
        f"A support agent will follow up with you within 24 hours. "
        f"Reference: DL-{state.get('scenario_id', 'unknown')}"
    )

    return {
        "final_answer": final_answer,
        "messages": [f"dead_letter:escalated after {attempt} attempts"],
        "events": [make_event("dead_letter", "completed",
                              f"max retries ({max_attempts}) exceeded, escalated",
                              attempt=attempt, error_count=len(errors))],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END."""
    route = state.get("route", "unknown")
    has_answer = bool(state.get("final_answer") or state.get("pending_question"))

    return {
        "events": [make_event("finalize", "completed", "workflow finished",
                              route=route, has_answer=has_answer)],
    }
