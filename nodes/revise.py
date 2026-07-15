from graph.state import ContentState
from nodes.llm_client import generate_text

def _format_issues(state: ContentState) -> str:
    lines = [f"- [{i['category']} / {i['severity']}] {i['message']}" for i in state["issues"]]
    return "\n".join(lines)

def revise(state: ContentState) -> ContentState:
    system_prompt = (
        "You are revising a draft post to fix specific flagged issues. "
        "Make the minimum changes needed to resolve each issue — do not "
        "rewrite parts that weren't flagged. Preserve the core message and "
        "structure."
    )
    user_prompt = (
        f"Current draft:\n{state['draft']}\n\n"
        f"Issues to fix:\n{_format_issues(state)}\n\n"
        "Return only the revised draft, no commentary."
    )

    revised = generate_text(system_prompt, user_prompt)
    state["draft"] = revised
    state["draft_history"].append(revised)
    state["iteration"] += 1
    return state
