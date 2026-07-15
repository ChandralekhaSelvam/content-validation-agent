"""
Smoke test for the Flask web layer. Mocks the LLM calls, exercises both
the topic and document upload paths through the actual HTTP test client
(not just the graph directly), and checks the error path for missing input.

Run with: python tests/test_web_smoke.py
"""

import io
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

FAKE_DRAFT = (
    "Six months ago I couldn't tell you what a LoRA adapter was. Last week I "
    "fine-tuned one on a free Colab T4. The hardest part was the dataset "
    "schema, not the training loop. If you're starting your own project, "
    "spend day one there instead of the training script — it saves a week "
    "of confused debugging later, learned that one the hard way myself."
)


def fake_generate_text(system_prompt, user_prompt, max_tokens=800):
    return FAKE_DRAFT


def test_topic_path():
    with patch("nodes.generate.generate_text", side_effect=fake_generate_text), \
         patch("nodes.revise.generate_text", side_effect=fake_generate_text):
        from web.app import app
        client = app.test_client()

        response = client.post("/api/generate", data={
            "topic": "fine-tuning a small language model",
            "platform": "linkedin",
        })

    assert response.status_code == 200, response.get_json()
    data = response.get_json()
    assert "draft" in data and data["draft"]
    assert "voice_score" in data
    assert "issues" in data
    print("test_topic_path PASS")


def test_document_path():
    with patch("nodes.rewrite_from_doc.generate_text", side_effect=fake_generate_text), \
         patch("nodes.revise.generate_text", side_effect=fake_generate_text):
        from web.app import app
        client = app.test_client()

        doc_bytes = io.BytesIO(b"Internal notes: our QLoRA fine-tune hit 94% valid JSON.")
        response = client.post("/api/generate", data={
            "document": (doc_bytes, "notes.txt"),
            "platform": "medium",
        }, content_type="multipart/form-data")

    assert response.status_code == 200, response.get_json()
    data = response.get_json()
    assert data["draft"]
    print("test_document_path PASS")


def test_missing_input_returns_400():
    from web.app import app
    client = app.test_client()

    response = client.post("/api/generate", data={"platform": "linkedin"})

    assert response.status_code == 400
    assert "error" in response.get_json()
    print("test_missing_input_returns_400 PASS")


def test_index_page_loads():
    from web.app import app
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"contentValidationAgent" in response.data
    print("test_index_page_loads PASS")


if __name__ == "__main__":
    test_topic_path()
    test_document_path()
    test_missing_input_returns_400()
    test_index_page_loads()
    print("\nAll web smoke tests passed.")
