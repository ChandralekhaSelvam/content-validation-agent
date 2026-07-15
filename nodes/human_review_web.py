"""
Web-mode human review node.

"""

from graph.state import ContentState

def human_review_web(state: ContentState) -> ContentState:
    state["needs_human_review"] = True
    state["final_post"] = None  # not finalized until the person acts in the UI
    return state
