from graph.state import ContentState

def route(state: ContentState) -> ContentState:
    if state.get("source_doc_path"):
        state["mode"] = "document"
    elif state.get("topic"):
        state["mode"] = "topic"
    else:
        raise ValueError("ContentState needs either 'topic' or 'source_doc_path' set.")

    state.setdefault("platform", "linkedin")
    state.setdefault("max_iterations", 3)
    state.setdefault("iteration", 0)
    state.setdefault("draft_history", [])
    state.setdefault("issues", [])
    state.setdefault("needs_human_review", False)
    return state

def route_decision(state: ContentState) -> str:
    """Conditional edge function used by the graph builder."""
    return "extract_doc" if state["mode"] == "document" else "generate_from_topic"
