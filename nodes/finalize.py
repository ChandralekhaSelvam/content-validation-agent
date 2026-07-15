"""
Human review + finalize nodes. Human review is a real interrupt point in
the CLI (main.py) — LangGraph's checkpointing means this could also pause
and resume across a web request if this were wrapped in FastAPI later.
"""

from graph.state import ContentState

def human_review(state: ContentState) -> ContentState:
    print("\n" + "=" * 60)
    print("HUMAN REVIEW REQUIRED")
    print("=" * 60)
    if state["issues"]:
        print("\nOutstanding issues:")
        for issue in state["issues"]:
            print(f"  - [{issue['category']}/{issue['severity']}] {issue['message']}")
    print(f"\nCurrent draft:\n{'-' * 60}\n{state['draft']}\n{'-' * 60}\n")

    choice = input("Approve as-is (a), edit manually (e), or reject (r)? ").strip().lower()

    if choice == "e":
        print("Paste your edited version, then an empty line to finish:")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        state["draft"] = "\n".join(lines)
        state["human_feedback"] = "manually edited"
    elif choice == "r":
        state["human_feedback"] = "rejected"
        state["final_post"] = None
        return state
    else:
        state["human_feedback"] = "approved as-is"

    state["final_post"] = state["draft"]
    return state


def finalize(state: ContentState) -> ContentState:
    state["final_post"] = state["draft"]
    return state
