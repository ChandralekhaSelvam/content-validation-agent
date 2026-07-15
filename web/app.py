"""
Flask app for contentValidationAgent.

Thin layer over the existing LangGraph pipeline — no pipeline logic lives
here. Two endpoints:
    GET  /              → serves the single-page UI
    POST /api/generate  → runs the graph, returns the resulting state as JSON

Run:
    ollama serve            # separate terminal, if not already running
    python web/app.py
"""

import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root, for `graph`/`nodes` imports

from graph.build_web_graph import build_web_graph

app = Flask(__name__)

ALLOWED_UPLOAD_EXTENSIONS = {".docx", ".txt", ".md"}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Accepts either:
      - multipart/form-data with a 'document' file, or
      - JSON/form body with a 'topic' string
    plus 'platform' ('linkedin' | 'medium'). Generation always runs
    locally via Ollama (see nodes/llm_client.py).
    """
    platform = request.form.get("platform", "linkedin")

    initial_state = {
        "platform": platform,
        "voice_profile_path": "data/voice_training_examples.jsonl",
    }

    uploaded_file = request.files.get("document")
    if uploaded_file and uploaded_file.filename:
        suffix = Path(uploaded_file.filename).suffix.lower()
        if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
            return jsonify({"error": f"Unsupported file type: {suffix}"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            uploaded_file.save(tmp.name)
            initial_state["source_doc_path"] = tmp.name
    else:
        topic = request.form.get("topic", "").strip()
        if not topic:
            return jsonify({"error": "Provide a topic or upload a document."}), 400
        initial_state["topic"] = topic

    try:
        app_graph = build_web_graph()
        final_state = app_graph.invoke(initial_state)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "draft": final_state.get("draft", ""),
        "final_post": final_state.get("final_post"),
        "voice_score": round(final_state.get("voice_score", 0), 2),
        "ai_sounding_score": round(final_state.get("ai_sounding_score", 0), 2),
        "iteration": final_state.get("iteration", 0),
        "max_iterations": final_state.get("max_iterations", 0),
        "issues": final_state.get("issues", []),
        "needs_human_review": final_state.get("needs_human_review", False),
        "passed": final_state.get("passed", False),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5050)
