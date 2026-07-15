"""
Shared state schema for the LinkedIn/Medium content agent graph.

Design note: we keep validator output as a list of typed issues rather than
a single pass/fail bool, so the Revise node can target specific problems
instead of re-writing blind on every loop iteration.
"""

from __future__ import annotations

from typing import List, Literal, Optional, TypedDict


class ValidationIssue(TypedDict):
    category: Literal["voice", "ai_sounding", "platform_fit", "factual_flag", "length"]
    severity: Literal["low", "medium", "high"]
    message: str


class ContentState(TypedDict, total=False):
    # --- inputs (one of these two paths is populated) ---
    mode: Literal["topic", "document"]
    topic: Optional[str]
    source_doc_path: Optional[str]
    source_doc_text: Optional[str]

    # --- target platform + voice config ---
    platform: Literal["linkedin", "medium"]
    voice_profile_path: str  # path to reference posts used for tone matching

    # --- working draft ---
    draft: str
    draft_history: List[str]  # every prior draft, for diffing/debug

    # --- validation ---
    issues: List[ValidationIssue]
    voice_score: float  # 0-1, higher = sounds more like the author
    ai_sounding_score: float  # 0-1, higher = more likely to read as AI-generated
    iteration: int
    max_iterations: int
    passed: bool

    # --- human-in-the-loop ---
    needs_human_review: bool
    human_feedback: Optional[str]

    # --- final ---
    final_post: Optional[str]
