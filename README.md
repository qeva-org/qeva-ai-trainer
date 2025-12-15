# qeva-ai-trainer

Minimal starter repo for training a **value-aware** language model using LoRA on top of an open-base model.

This version is designed to run easily on **Google Colab** with a T4 GPU.

## What it does

- Uses `Qwen/Qwen2-1.5B-Instruct` as the base model.
- Creates a tiny values dataset (`data/values_tiny.jsonl`) with 10 examples about safety, honesty, and sustainable value.
- Fine-tunes a small LoRA adapter on that dataset.
- Saves the adapter to `qwen2-values-tiny-lora/`.

This is **not** meant to be a full alignment pipeline, just a working skeleton you can extend.

---

## Quickstart on Colab

1. Upload or download this repo as a zip, then in Colab:

```bash
!unzip qeva-ai-trainer.zip -d /content
%cd /content/qeva-ai-trainer
```

2. Install dependencies:

```bash
!pip install -q -r requirements.txt
```

3. Run the training script:

```bash
!python src/train_qwen2_values_tiny.py
```

4. After training, test the tuned model:

```bash
!python src/generate_example.py

5. Run a cheap "value adherence" evaluation (no external judge):

```bash
!python src/eval_pref_loglik.py
```

This computes a preference-loglikelihood accuracy score on `data/values_pref_tiny.jsonl` and prints a 95% confidence interval.
```

Both scripts assume you have a GPU available and `torch` with CUDA already installed
(as is standard in Google Colab GPU runtimes).

---

## Files

- `requirements.txt` – Python dependencies (transformers, datasets, accelerate, peft, trl).
- `data/values_tiny.jsonl` – tiny seed dataset of value examples (auto-created if missing).
- `src/train_qwen2_values_tiny.py` – main training script (LoRA + SFT).
- `src/generate_example.py` – load tuned adapter and generate a sample response.
- `configs/` – reserved for future YAML configs.
- `notebooks/` – reserved for future Colab/Jupyter notebooks.

You can change the base model by setting the `BASE_MODEL` environment variable, e.g.:

```bash
BASE_MODEL="microsoft/Phi-3-mini-128k-instruct" python src/train_qwen2_values_tiny.py
```

(Just make sure the model fits into your GPU memory.)