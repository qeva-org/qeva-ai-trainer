# Tiny CPU LoRA demonstration

`src/demo_tiny_lora.py --output NEW_DIRECTORY` trains and reloads an actual LoRA
adapter against a tiny, randomly initialized Qwen2 model on four fictional examples.
It requires the existing PyTorch/Transformers environment plus `requirements-demo.txt`.
It needs no model download, account, GPU or paid service. It uses 24 steps, two CPU
threads and a 90-second training-loop limit. Run through the QEVA demo launcher for
an outer process time limit and isolated output.

The receipt retains configuration context, loss before/after training, heldout loss,
parameter changes and adapter-reload agreement. This demonstrates training and
serialization mechanics only. No useful language behaviour, generalization or
alignment effectiveness follows. The original pretrained 1.5B training and generation
scripts are unchanged and need their original dependencies, weights and compute.
