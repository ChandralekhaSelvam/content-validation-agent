"""
Smoke test: runs the full graph with the LLM client mocked, so wiring can be
verified without API keys or GPU. Run with: python tests/test_graph_smoke.py
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

FAKE_DRAFT_GOOD = (
    "Six months ago I couldn't tell you what a LoRA adapter was. Last week I "
    "fine-tuned one on a free Colab T4.\n\n"
    "Here's the part nobody tells you: the hardest bit wasn't the training "
    "loop, it was getting the tool-call format consistent enough to be "
    "useful. Unsloth handled the memory constraints. My data format was the "
    "actual bottleneck.\n\n"
    "If you're starting your own SLM fine-tuning project, spend your first "
    "day on the dataset schema, not the training script."
)


def fake_generate_text(system_prompt, user_prompt, max_tokens=800):
    return FAKE_DRAFT_GOOD


def run_topic_path():
    from graph.build_graph import build_graph

    with patch("nodes.generate.generate_text", side_effect=fake_generate_text), \
         patch("nodes.revise.generate_text", side_effect=fake_generate_text), \
         patch("builtins.input", return_value="a"):  # auto-approve if it hits human review
        app = build_graph()
        result = app.invoke({
            "topic": "fine-tuning a small language model for tool calling",
            "platform": "linkedin",
            "voice_profile_path": "data/voice_training_examples.jsonl",
        })

    print("--- TOPIC PATH RESULT ---")
    print("mode:", result["mode"])
    print("iterations:", result["iteration"])
    print("voice_score:", round(result["voice_score"], 2))
    print("ai_sounding_score:", round(result["ai_sounding_score"], 2))
    print("issues:", result["issues"])
    print("final_post is None:", result.get("final_post") is None)
    assert result["mode"] == "topic"
    assert "final_post" in result
    print("PASS\n")


def run_document_path(tmp_docx):
    from graph.build_graph import build_graph

    with patch("nodes.rewrite_from_doc.generate_text", side_effect=fake_generate_text), \
         patch("nodes.revise.generate_text", side_effect=fake_generate_text), \
         patch("builtins.input", return_value="a"):
        app = build_graph()
        result = app.invoke({
            "source_doc_path": str(tmp_docx),
            "platform": "medium",
            "voice_profile_path": "data/voice_training_examples.jsonl",
        })

    print("--- DOCUMENT PATH RESULT ---")
    print("mode:", result["mode"])
    print("source_doc_text (first 60 chars):", result["source_doc_text"][:60])
    print("final_post is None:", result.get("final_post") is None)
    assert result["mode"] == "document"
    assert result["source_doc_text"]
    print("PASS\n")


if __name__ == "__main__":
    run_topic_path()

    # Use a .txt for the doc path smoke test to avoid a pandoc/docx dependency here
    test_txt = Path(__file__).parent / "fixture_notes.txt"
    test_txt.write_text(
        "Internal notes: our QLoRA fine-tune on Llama 3.2 3B hit 94% valid "
        "JSON output after 3 epochs on a T4, using Unsloth for 4-bit loading."
    )
    run_document_path(test_txt)

    print("All smoke tests passed.")
