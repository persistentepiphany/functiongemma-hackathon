"""
Pre-inference difficulty classifier for hybrid tool-calling routing.

Classifies queries as "easy", "medium", or "hard" based on structural
signals in the message and tool schema — no ML, no external deps.

Aggressive toward "easy" to maximise on-device ratio.
"""

from __future__ import annotations

import re
from typing import List, Dict

# ── Multi-action conjunctions that signal the user wants >1 tool call ──
# These are patterns where a conjunction joins two *imperative clauses*,
# not just two nouns ("salt and pepper").
_MULTI_ACTION_PATTERNS = [
    r"\band\s+(?:also|then)\b",       # "and also", "and then"
    r"\bthen\b",                        # "X then Y"
    r"\bafter\s+that\b",               # "after that"
    r"\balso\b",                        # "also do X"
    r"\bplus\b",                        # "plus set a timer"
]

# Conjunction "and" between verb phrases (heuristic: "and" preceded by
# comma or end of clause, followed by a verb-like word).
_AND_VERB_RE = re.compile(
    r",?\s+and\s+(?:set|get|send|play|create|check|find|look|search|text|remind|wake|start|tell)\b",
    re.IGNORECASE,
)

# ── Negation / conditional phrases that increase ambiguity ──
_COMPLEXITY_PHRASES = [
    r"\bbut\s+not\b",
    r"\bunless\b",
    r"\bexcept\b",
    r"\bif\b",
    r"\bonly\s+if\b",
    r"\binstead\s+of\b",
    r"\bdon'?t\b",
    r"\bwithout\b",
]

# ── Vague / ambiguous parameter references ──
_AMBIGUITY_PHRASES = [
    r"\bsomething\b",
    r"\bsomewhere\b",
    r"\bwhatever\b",
    r"\bsome\s+\w+\b",
    r"\bthe\s+usual\b",
    r"\bmy\s+favorite\b",
    r"\bnearby\b",
    r"\baround\s+here\b",
    r"\blater\b",
    r"\bsoon\b",
]


def _count_action_clauses(text: str) -> int:
    """Estimate how many distinct tool-call actions the user is requesting."""
    lower = text.lower()
    count = 1

    # Explicit multi-action conjunctions
    for pat in _MULTI_ACTION_PATTERNS:
        count += len(re.findall(pat, lower))

    # "and <verb>" — strong signal of a second action
    count += len(_AND_VERB_RE.findall(lower))

    # Comma-separated imperative clauses: ", play ...", ", check ..."
    comma_clauses = re.findall(
        r",\s+(?:set|get|send|play|create|check|find|look|search|text|remind|wake|start|tell)\b",
        lower,
    )
    count += len(comma_clauses)

    return count


def _has_complexity(text: str) -> bool:
    lower = text.lower()
    return any(re.search(pat, lower) for pat in _COMPLEXITY_PHRASES)


def _has_ambiguity(text: str) -> bool:
    lower = text.lower()
    return any(re.search(pat, lower) for pat in _AMBIGUITY_PHRASES)


def classify_difficulty(messages: list[dict], tools: list[dict]) -> str:
    """
    Classify a tool-calling query as "easy", "medium", or "hard".

    Aggressive toward "easy" — the local model handles simple single-tool
    calls well, and on-device ratio is 25% of the hackathon score.

    Routing contract:
        "easy"   → always local
        "medium" → local first, validate, cloud on failure
        "hard"   → direct to cloud
    """
    # Extract user message (last user turn)
    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = m.get("content", "")
            break

    n_tools = len(tools)
    n_clauses = _count_action_clauses(user_msg)
    msg_len = len(user_msg)
    has_complex = _has_complexity(user_msg)
    has_ambig = _has_ambiguity(user_msg)

    # ── Hard: multiple actions requested ──
    # If the message clearly asks for 2+ distinct tool calls, the small
    # model will struggle to emit them all correctly.
    if n_clauses >= 2:
        # Even with multiple clauses, if there's only 1 tool available
        # it can only be called once — downgrade to medium.
        if n_tools == 1:
            return "medium"
        return "hard"

    # ── Easy: single tool in schema ──
    # With only one tool to choose from, even a 270M model gets it right.
    if n_tools == 1:
        return "easy"

    # ── Single action, multiple tools ──
    # Now we know: 1 clause, 2+ tools. The model must pick the right one.

    # Aggressive easy: short, clear message + few tools + no weirdness
    if n_tools <= 3 and msg_len < 60 and not has_complex and not has_ambig:
        return "easy"

    # Still manageable locally: moderate tool count, clear message
    if n_tools <= 2 and not has_complex and not has_ambig:
        return "easy"

    # Complexity or ambiguity bumps to medium regardless
    if has_complex or has_ambig:
        return "medium"

    # Default: many tools but single clear action → medium
    return "medium"
