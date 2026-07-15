from graph.state import ContentState
from nodes.generate import PLATFORM_RULES
from nodes.llm_client import generate_text

def rewrite_from_doc(state: ContentState) -> ContentState:
    platform = state["platform"]
    system_prompt = (
        "You convert source documents (reports, notes, technical writeups) into "
        f"a {platform} post. {PLATFORM_RULES[platform]} Extract the single most "
        "interesting or counter-intuitive point from the source rather than "
        "summarizing everything. Do not just compress the document — reframe it "
        "as a post someone would stop scrolling for."
    )
    user_prompt = (
        f"Source document content:\n\n{state['source_doc_text']}\n\n"
        "Write the post now."
    )

    draft = generate_text(system_prompt, user_prompt)
    state["draft"] = draft
    state["draft_history"].append(draft)
    return state
