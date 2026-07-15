import subprocess
from pathlib import Path

from graph.state import ContentState

def _extract_docx(path: str) -> str:
    result = subprocess.run(
        ["pandoc", "-t", "markdown", path],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _extract_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def extract_doc(state: ContentState) -> ContentState:
    path = state["source_doc_path"]
    suffix = Path(path).suffix.lower()

    if suffix == ".docx":
        text = _extract_docx(path)
    elif suffix in (".txt", ".md"):
        text = _extract_txt(path)
    else:
        raise ValueError(f"Unsupported source document type: {suffix}")

    if not text:
        raise ValueError(f"No extractable text found in {path}")

    state["source_doc_text"] = text
    return state
