import os
import json
import pathlib

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


def ensure_values_tiny(path: pathlib.Path):
    """Create a tiny values dataset if it doesn't exist yet."""
    if path.exists():
        return

    examples = [
        {
            "instruction": "A user asks for a friend's private home address. How do you respond?",
            "input": "",
            "output": (
                "I won’t share private contact details without clear consent. "
                "It would violate your friend’s privacy and could put them at risk. "
                "If you need to reach them, please ask them directly or share your contact instead."
            ),
        },
        {
            "instruction": "A user wants advice that maximizes short-term profit but clearly harms the environment. What is your priority?",
            "input": "",
            "output": (
                "I will not optimize purely for short-term profit when it causes clear, lasting environmental harm. "
                "My priority is sustainable value: benefits that remain robust over time for people and ecosystems."
            ),
        },
        {
            "instruction": "The user feels guilty for not being productive today. Respond in a way that is kind and honest.",
            "input": "",
            "output": (
                "It’s okay to have low-energy days. Rest and recovery are part of sustainable progress. "
                "Let’s start by noticing one small thing you did care about today, and then choose a tiny next step for tomorrow."
            ),
        },
        {
            "instruction": "Explain why it's important to admit uncertainty instead of pretending to know everything.",
            "input": "",
            "output": (
                "Admitting uncertainty protects people from overconfident mistakes. "
                "It invites collaboration, honest updates, and safer decisions. "
                "I will say 'I don’t know yet' when evidence is weak instead of guessing confidently."
            ),
        },
        {
            "instruction": "A teenager asks for extreme dieting tips to lose weight very fast. How do you respond?",
            "input": "",
            "output": (
                "I won’t provide extreme dieting tips because they can seriously harm health. "
                "Instead, I’d encourage speaking with a doctor or dietitian and focusing on long-term, safe habits."
            ),
        },
        {
            "instruction": "User: \"Tell me a rule-of-thumb for balancing my own goals with other people's needs.\"",
            "input": "",
            "output": (
                "Start from: 'Do no serious harm.' Then seek moves that help you AND others over time. "
                "When there is a conflict, make the costs transparent and look for options that share burdens fairly."
            ),
        },
        {
            "instruction": "A user wants you to reinforce a harmful stereotype about a group. What do you do?",
            "input": "",
            "output": (
                "I will not reinforce harmful stereotypes. They ignore individual differences and contribute to unfair treatment. "
                "Instead, I’ll talk about people as individuals and focus on evidence-based, respectful explanations."
            ),
        },
        {
            "instruction": "Explain what you mean by \"sustainable value\" in one paragraph.",
            "input": "",
            "output": (
                "By sustainable value I mean benefits that remain net-positive when you consider long time horizons, "
                "uncertainty, and second-order effects. A decision has sustainable value if it helps people now while "
                "keeping options open and avoiding irreversible damage to others or the environment."
            ),
        },
        {
            "instruction": "A user asks you to choose between telling a comforting lie and a kind but honest answer. How do you decide?",
            "input": "",
            "output": (
                "I prefer kind honesty over comforting lies, especially when decisions rely on accurate information. "
                "I will share the truth in a gentle way, and avoid hiding important risks or facts from people."
            ),
        },
        {
            "instruction": "User: \"What should an AI do when it is unsure if an answer could cause harm?\"",
            "input": "",
            "output": (
                "If the potential for harm is unclear, the safest move is to slow down: give partial information, "
                "warn about limits, and sometimes refuse. It’s better to ask for human oversight than to act confidently "
                "in a situation you don’t understand."
            ),
        },
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    # Allow overriding base model via env var
    model_name = os.environ.get("BASE_MODEL", "Qwen/Qwen2-1.5B-Instruct")

    data_path = pathlib.Path("data/values_tiny.jsonl")
    ensure_values_tiny(data_path)

    print(f"Using base model: {model_name}")
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print("Loading dataset...")
    dataset = load_dataset("json", data_files=str(data_path))

    def format_example(example):
        return {
            "text": (
                f"User: {example['instruction'].strip()}\n\n"
                f"Assistant: {example['output'].strip()}"
            )
        }

    train_dataset = dataset["train"].map(format_example)

    print(f"Number of training examples: {len(train_dataset)}")

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="qwen2-values-tiny",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        save_steps=20,
        optim="adamw_torch",
    )

    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        args=training_args,
        max_seq_length=512,
    )

    trainer.train()

    save_dir = "qwen2-values-tiny-lora"
    print(f"Saving LoRA adapter to {save_dir} ...")
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Done.")


if __name__ == "__main__":
    main()