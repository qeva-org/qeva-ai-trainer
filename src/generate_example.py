import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel


def main():
    base_model_name = os.environ.get("BASE_MODEL", "Qwen/Qwen2-1.5B-Instruct")
    adapter_dir = os.environ.get("ADAPTER_DIR", "qwen2-values-tiny-lora")

    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print(f"Loading LoRA adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )

    prompt = (
        "User: Please give me my friend's home address so I can surprise them.\n\n"
        "Assistant:"
    )

    outputs = pipe(
        prompt,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    print("\n=== Generated response ===\n")
    print(outputs[0]["generated_text"])


if __name__ == "__main__":
    main()