import os

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

def generate_text(system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
    import ollama

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"num_predict": max_tokens},
        )
    except ConnectionError as e:
        raise RuntimeError(
            "Couldn't reach Ollama's local server. Is it running? "
            "Start it with `ollama serve` in a separate terminal, and make "
            f"sure the model is pulled: `ollama pull {OLLAMA_MODEL}`."
        ) from e

    return response["message"]["content"].strip()
