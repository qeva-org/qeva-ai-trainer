"""CPU-only LoRA mechanics on a tiny randomly initialized Qwen2 model.

No pretrained weights, downloads, trained alignment or language quality are claimed.
The normal pretrained Qwen2 training scripts remain separate and unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from peft import LoraConfig, PeftModel, get_peft_model
    torch.manual_seed(0)
    torch.set_num_threads(2)
    started = time.monotonic()
    config = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    base = Qwen2ForCausalLM(config)
    # Four original, fictional teaching examples, not a behavioural evaluation dataset.
    texts = ["User: Is this proven? Assistant: The evidence is incomplete.",
             "User: Can you share private data? Assistant: Ask for consent first.",
             "User: Did the test pass? Assistant: Report the actual result.",
             "User: What is unknown? Assistant: State the missing evidence."]
    holdout = "User: Is a simulation a measurement? Assistant: No."
    def encode(text):
        return [ord(c) for c in text] + [2]
    encoded = [encode(text) for text in texts]
    width = max(map(len, encoded))
    tokens = torch.tensor([row + [0] * (width - len(row)) for row in encoded])
    mask = tokens.ne(0).long()
    labels = tokens.clone()
    labels[tokens == 0] = -100
    evaluation = torch.tensor([encode(holdout)])
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                           lora_dropout=0., bias="none", task_type="CAUSAL_LM"))
    def loss(data, targets, attention=None):
        with torch.no_grad():
            return float(model(input_ids=data, attention_mask=attention, labels=targets).loss)
    model.eval()
    before = {"training_loss": loss(tokens, labels, mask), "heldout_loss": loss(evaluation, evaluation)}
    initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    trace = []
    model.train()
    for step in range(24):
        if time.monotonic() - started > 90:
            raise TimeoutError("CPU training demonstration exceeded 90 seconds")
        optimizer.zero_grad()
        value = model(input_ids=tokens, attention_mask=mask, labels=labels).loss
        if not torch.isfinite(value):
            raise ValueError("Nonfinite training loss")
        value.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        trace.append({"step": step + 1, "loss": float(value.detach())})
    model.eval()
    after = {"training_loss": loss(tokens, labels, mask), "heldout_loss": loss(evaluation, evaluation)}
    delta = sum(float(torch.sum((p.detach() - initial[n]) ** 2)) for n, p in model.named_parameters() if n in initial)
    if delta <= 0:
        raise ValueError("No adapter parameters changed")
    adapter = args.output / "adapter"
    model.save_pretrained(adapter)
    # Independent reloading of the saved adapter against identically seeded base weights.
    torch.manual_seed(0)
    restored = PeftModel.from_pretrained(Qwen2ForCausalLM(config), adapter).eval()
    with torch.no_grad():
        restored_loss = float(restored(input_ids=evaluation, labels=evaluation).loss)
    if abs(restored_loss - after["heldout_loss"]) > 1e-5:
        raise ValueError("Reloaded adapter did not reproduce heldout loss")
    result = {"kind": "synthetic_cpu_adapter_mechanics", "base": "randomly initialized tiny Qwen2; not pretrained",
              "seed": 0, "training_examples": texts, "heldout_example": holdout,
              "before": before, "after": after, "restored_heldout_loss": restored_loss,
              "adapter_parameter_change_squared": delta, "steps": trace,
              "elapsed_seconds": time.monotonic() - started,
              "torch_version": torch.__version__, "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limitations": ["No useful language behaviour expected from this tiny random base", "No alignment effectiveness established",
                              "Training loss is not heldout generalization", "Original pretrained 1.5B training remains a separate resource-dependent workflow"]}
    (args.output / "training.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"before": before, "after": after, "adapter_reloaded": True}))


if __name__ == "__main__":
    main()
