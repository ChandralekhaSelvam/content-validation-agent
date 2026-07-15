"""
Validator node: the core of the "validate" half of generate-validate-revise.
Runs several independent checks and aggregates them into typed issues, so
the Revise node can address specific problems rather than re-rolling blind.

Checks:
  - voice + ai_sounding: fine-tuned SLM (slm/voice_classifier.py)
  - platform_fit: rule-based length/format checks (cheap, deterministic)
  - factual_flag: lightweight heuristic for unverified-sounding claims
    (numbers/stats with no attribution) — flags for human review, does not
    block on its own since this needs a human's judgment, not an LLM's
"""

import re

from graph.state import ContentState, ValidationIssue
from slm.voice_classifier import score_draft

VOICE_SCORE_THRESHOLD = 0.5
AI_SOUNDING_THRESHOLD = 0.4

PLATFORM_LIMITS = {
    "linkedin": {"min_words": 100, "max_words": 350, "max_hashtags": 3},
    "medium": {"min_words": 500, "max_words": 1500, "max_hashtags": 5},
}

_UNATTRIBUTED_STAT_PATTERN = re.compile(r"\b\d{1,3}(?:\.\d+)?%|\b\d+x\b")


def _check_platform_fit(draft: str, platform: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    limits = PLATFORM_LIMITS[platform]
    word_count = len(draft.split())

    if word_count < limits["min_words"]:
        issues.append({
            "category": "length",
            "severity": "medium",
            "message": f"Draft is {word_count} words, below the {limits['min_words']} minimum for {platform}.",
        })
    elif word_count > limits["max_words"]:
        issues.append({
            "category": "length",
            "severity": "medium",
            "message": f"Draft is {word_count} words, above the {limits['max_words']} max for {platform}.",
        })

    hashtag_count = len(re.findall(r"#\w+", draft))
    if hashtag_count > limits["max_hashtags"]:
        issues.append({
            "category": "platform_fit",
            "severity": "low",
            "message": f"{hashtag_count} hashtags found, more than the recommended {limits['max_hashtags']}.",
        })

    return issues


def _check_unattributed_stats(draft: str) -> list[ValidationIssue]:
    matches = _UNATTRIBUTED_STAT_PATTERN.findall(draft)
    if not matches:
        return []
    return [{
        "category": "factual_flag",
        "severity": "high",
        "message": (
            f"Draft contains {len(matches)} numeric claim(s) ({', '.join(matches[:3])}...) "
            "with no visible source — verify before posting."
        ),
    }]


def validate(state: ContentState) -> ContentState:
    draft = state["draft"]
    platform = state["platform"]

    voice_score, ai_sounding_score = score_draft(draft)
    state["voice_score"] = voice_score
    state["ai_sounding_score"] = ai_sounding_score

    issues: list[ValidationIssue] = []

    if voice_score < VOICE_SCORE_THRESHOLD:
        issues.append({
            "category": "voice",
            "severity": "high",
            "message": f"Voice score {voice_score:.2f} is below threshold {VOICE_SCORE_THRESHOLD} — doesn't sound like the author's usual writing.",
        })

    if ai_sounding_score > AI_SOUNDING_THRESHOLD:
        issues.append({
            "category": "ai_sounding",
            "severity": "high",
            "message": f"AI-sounding score {ai_sounding_score:.2f} exceeds threshold {AI_SOUNDING_THRESHOLD} — reads as generic AI copy.",
        })

    issues.extend(_check_platform_fit(draft, platform))
    issues.extend(_check_unattributed_stats(draft))

    state["issues"] = issues
    state["passed"] = not any(i["severity"] == "high" for i in issues)

    # factual_flag issues always need a human, regardless of pass/fail —
    # no model in this pipeline is allowed to self-certify a factual claim
    if any(i["category"] == "factual_flag" for i in issues):
        state["needs_human_review"] = True

    return state


def should_continue(state: ContentState) -> str:
    """Conditional edge: loop back to revise, escalate to human, or finish."""
    if state["passed"]:
        return "human_review" if state["needs_human_review"] else "finalize"

    if state["iteration"] >= state["max_iterations"]:
        state["needs_human_review"] = True
        return "human_review"

    return "revise"
