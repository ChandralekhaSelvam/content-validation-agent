"""
CLI entry point.

Usage:
    python main.py --topic "prompt engineering vs context engineering" --platform linkedin
    python main.py --doc path/to/notes.docx --platform medium
"""

import argparse

from graph.build_graph import build_graph


def main():
    parser = argparse.ArgumentParser(description="LinkedIn/Medium content agent")
    parser.add_argument("--topic", help="Topic to write a post about")
    parser.add_argument("--doc", help="Path to a .docx/.txt/.md file to rewrite as a post")
    parser.add_argument("--platform", choices=["linkedin", "medium"], default="linkedin")
    args = parser.parse_args()

    if not args.topic and not args.doc:
        parser.error("Provide either --topic or --doc")

    initial_state = {
        "platform": args.platform,
        "voice_profile_path": "data/voice_training_examples.jsonl",
    }
    if args.doc:
        initial_state["source_doc_path"] = args.doc
    else:
        initial_state["topic"] = args.topic

    app = build_graph()
    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    if final_state.get("final_post"):
        print("FINAL POST")
        print("=" * 60)
        print(final_state["final_post"])
    else:
        print("No final post produced (rejected in human review).")
    print("=" * 60)
    print(f"\nIterations used: {final_state['iteration']}/{final_state['max_iterations']}")
    print(f"Voice score: {final_state.get('voice_score', 'n/a')}")
    print(f"AI-sounding score: {final_state.get('ai_sounding_score', 'n/a')}")


if __name__ == "__main__":
    main()
