"""
Fine-tunes a LoRA adapter on Llama 3.2 3B to classify draft posts as
"sounds like [author]" vs "sounds like generic AI copy".

Same Unsloth + PEFT pattern as slm-toolcall-lora, applied to a scoring/
classification task instead of tool-call generation. Runs on a free Colab
T4 — same constraints you already worked within for the tool-call model.

Training data format (data/voice_training_examples.jsonl), one JSON object
per line:
    {"text": "<draft text>", "voice_score": 0.9, "ai_sounding_score": 0.1}

Build this dataset from:
  - POSITIVE (high voice_score): your own published LinkedIn/Medium posts
  - NEGATIVE (high ai_sounding_score): raw, un-edited LLM outputs on similar
    topics, especially early drafts you personally rejected as "too AI"

That second category is exactly the corpus you already generated organically
by iterating on your LinkedIn profile rewrite — the discarded drafts ARE
your negative training examples. Worth saving them going forward.
"""

import json
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DATA_PATH = "data/voice_training_examples.jsonl"
OUTPUT_DIR = "slm/adapters/voice-lora"

PROMPT_TEMPLATE = (
    "Rate this LinkedIn/Medium draft on two 0-1 scores.\n"
    "voice_score: does it sound like a specific human's authentic writing?\n"
    "ai_sounding_score: does it read as generic AI-generated copy?\n"
    'Respond ONLY as JSON: {{"voice_score": <float>, "ai_sounding_score": <float>}}\n\n'
    "Draft:\n{text}"
)


def load_dataset(path: str) -> Dataset:
    examples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            prompt = PROMPT_TEMPLATE.format(text=row["text"])
            completion = json.dumps(
                {"voice_score": row["voice_score"], "ai_sounding_score": row["ai_sounding_score"]}
            )
            examples.append({"prompt": prompt, "completion": completion})
    return Dataset.from_list(examples)


def tokenize(example, tokenizer):
    full_text = example["prompt"] + example["completion"]
    tokenized = tokenizer(full_text, truncation=True, max_length=768, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main():
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Create it first — see this file's docstring "
            "for the expected format and where to source positive/negative examples."
        )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_4bit=True, device_map="auto")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(DATA_PATH)
    tokenized_dataset = dataset.map(lambda ex: tokenize(ex, tokenizer), remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
