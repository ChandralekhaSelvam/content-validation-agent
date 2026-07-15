"""
Same graph as build_graph.py, with one difference: the human_review node
is the non-blocking web variant (nodes/human_review_web.py) instead of the
CLI's input()-based one. Used by web/app.py only — main.py (CLI) keeps
using build_graph() unchanged.
"""

from langgraph.graph import END, StateGraph

from graph.state import ContentState
from nodes.extract_doc import extract_doc
from nodes.finalize import finalize
from nodes.generate import generate_from_topic
from nodes.human_review_web import human_review_web
from nodes.revise import revise
from nodes.rewrite_from_doc import rewrite_from_doc
from nodes.router import route, route_decision
from nodes.validate import should_continue, validate


def build_web_graph():
    graph = StateGraph(ContentState)

    graph.add_node("router", route)
    graph.add_node("generate_from_topic", generate_from_topic)
    graph.add_node("extract_doc", extract_doc)
    graph.add_node("rewrite_from_doc", rewrite_from_doc)
    graph.add_node("validate", validate)
    graph.add_node("revise", revise)
    graph.add_node("human_review", human_review_web)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_decision,
        {"generate_from_topic": "generate_from_topic", "extract_doc": "extract_doc"},
    )

    graph.add_edge("generate_from_topic", "validate")
    graph.add_edge("extract_doc", "rewrite_from_doc")
    graph.add_edge("rewrite_from_doc", "validate")

    graph.add_conditional_edges(
        "validate",
        should_continue,
        {"revise": "revise", "human_review": "human_review", "finalize": "finalize"},
    )

    graph.add_edge("revise", "validate")
    graph.add_edge("human_review", END)
    graph.add_edge("finalize", END)

    return graph.compile()
