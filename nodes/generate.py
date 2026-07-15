"""
Generate node: produces the first draft from a topic string. Deliberately
NOT trying to match voice here — that's the validator's job. This node's
only responsibility is a structurally sound, platform-appropriate draft.
"""

from graph.state import ContentState
from nodes.llm_client import generate_text

PLATFORM_RULES = {
    "linkedin": (
        "Write for LinkedIn: 150-300 words, hook in the first line (shows before "
        "'see more'), short paragraphs (1-3 lines), no more than 2 hashtags at the "
        "end, no markdown headers, plain conversational text."
    ),
    "medium": (
        "Write for Medium: 600-1200 words, can use markdown headers (##), can be "
        "more narrative and reflective, subheadings encouraged for skimmability."
    ),
}

def generate_from_topic(state: ContentState) -> ContentState:
    platform = state["platform"]
    system_prompt = (
        "You are a ghostwriter drafting a first-pass professional post. "
        f"{PLATFORM_RULES[platform]} Do not use generic AI phrasing like "
        "'In today's fast-paced world' or 'Let's dive in'. Write plainly, "
        "with a specific angle, not a generic overview."
    )
    user_prompt = f"Topic: {state['topic']}\n\nWrite the post now."

    draft = generate_text(system_prompt, user_prompt)
    state["draft"] = draft
    state["draft_history"].append(draft)
    return state
