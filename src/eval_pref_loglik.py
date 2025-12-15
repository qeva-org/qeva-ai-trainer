"""Evaluate 'value adherence' via preference log-likelihood.

Idea:
  For each prompt, we have a (good, bad) response pair.
  We score the model by which response it assigns higher conditional log-probability.

Why this is useful:
  - No external 'judge model' needed.
  - Produces a clear accuracy metric with confidence intervals.
  - Works for any causal LM (base model or LoRA-tuned).

Usage examples:
  # evaluate base model
  python src/eval_pref_loglik.py

  # evaluate a LoRA-adapted model
  BASE_MODEL="Qwen/Qwen2-1.5B-Instruct" ADAPTER_PATH="qwen2-values-tiny-lora" python src/eval_pref_loglik.py

Notes:
  - This is NOT a full alignment eval. It's a cheap, rigorous sanity check.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    # When run as a module: python -m src.eval_pref_loglik
    from .metrics import wilson_ci
except Exception:
    # When run as a script: python src/eval_pref_loglik.py
    from metrics import wilson_ci


def load_pref_pairs(path: pathlib.Path) -> List[Dict[str, str]]:
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    # minimal validation
    for p in pairs:
        for k in ("instruction", "good", "bad"):
            if k not in p:
                raise ValueError(f"Missing key {k} in: {p}")
    return pairs


def format_prompt(instruction: str) -> str:
    return f"User: {instruction.strip()}\n\nAssistant:"


@torch.no_grad()
def conditional_logprob(model, tokenizer, prompt: str, completion: str, device: torch.device) -> float:
    """Compute log P(completion | prompt) under a causal LM.

    We compute token-level negative log-likelihood on the completion tokens only.
    """
    # Encode prompt and prompt+completion
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(prompt + " " + completion, return_tensors="pt").input_ids.to(device)

    # Labels: ignore prompt tokens, score only completion tokens.
    labels = full_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100

    out = model(full_ids, labels=labels)
    # loss is mean NLL per *scored* token; multiply by count to get total NLL
    loss = float(out.loss)
    scored = (labels != -100).sum().item()
    total_nll = loss * max(scored, 1)
    return -total_nll  # logprob


def main():
    model_name = os.environ.get("BASE_MODEL", "Qwen/Qwen2-1.5B-Instruct")
    adapter_path = os.environ.get("ADAPTER_PATH", "")
    data_path = pathlib.Path(os.environ.get("PREF_DATA", "data/values_pref_tiny.jsonl"))

    print(f"Base model: {model_name}")
    print(f"Adapter: {adapter_path or '(none)'}")
    print(f"Pref data: {data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )

    if adapter_path:
        # Optional: load LoRA adapter if present
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
            print("Loaded LoRA adapter.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load adapter at {adapter_path}. Set ADAPTER_PATH='' to skip. Error: {e}"
            )

    model.eval()

    pairs = load_pref_pairs(data_path)
    t0 = time.time()

    wins = 0
    for i, p in enumerate(pairs, 1):
        prompt = format_prompt(p["instruction"])
        lp_good = conditional_logprob(model, tokenizer, prompt, p["good"], device)
        lp_bad = conditional_logprob(model, tokenizer, prompt, p["bad"], device)
        ok = lp_good > lp_bad
        wins += int(ok)
        print(f"[{i:02d}/{len(pairs)}] good>bad: {ok}  (Δ={lp_good - lp_bad:+.2f})")

    dt = time.time() - t0
    ci = wilson_ci(wins, len(pairs))
    print("\n=== Preference Log-Likelihood Accuracy ===")
    print(f"accuracy = {ci.mean:.3f}  (95% CI: [{ci.lo:.3f}, {ci.hi:.3f}])  n={ci.n}")
    print(f"wall time = {dt:.2f}s")

    # Optional: crude compute-energy estimate
    # If you supply AVG_GPU_WATTS (e.g. 70 for a Colab T4 under light load), we convert to joules.
    watts = os.environ.get("AVG_GPU_WATTS", "")
    if watts:
        try:
            w = float(watts)
            joules = w * dt
            # "meaning efficiency" analogue: value accuracy per joule (toy but useful)
            eff = ci.mean / max(joules, 1e-9)
            print("\n=== Rough Energy Accounting (user-supplied) ===")
            print(f"AVG_GPU_WATTS={w:.1f}W  =>  energy ≈ {joules:.1f} J")
            print(f"efficiency ≈ {eff:.3e} accuracy/J")
        except ValueError:
            print("AVG_GPU_WATTS was set but not a number; skipping energy estimate.")


if __name__ == "__main__":
    main()
