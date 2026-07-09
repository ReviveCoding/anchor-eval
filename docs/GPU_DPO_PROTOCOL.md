# Equal-Budget GPU DPO Transfer Protocol

The GPU experiment compares label-source quality only. It freezes:

- base model and reference-policy initialization
- prompt pool and source-group-disjoint DPO split
- selected pair count and item IDs
- sequence limits, QLoRA configuration, epochs, optimizer defaults, and seed set
- decoding configuration
- independent external evaluation suite

Policies:

- P0: no preference tuning.
- P1: raw C1 judge labels.
- P2: C5 routed evaluator labels.

The runner is GPU-only. It rejects CPU fallback, a missing or non-empty output directory, malformed DPO pools, duplicate source groups, or a label source other than `raw_judge` or `routed_evaluator`.

## Memory-safe configuration

The optional runner uses 4-bit NF4 quantization, double quantization, preparation for k-bit training, and LoRA adapters targeted at linear layers. It uses an explicit single-GPU device mapping for training rather than `device_map="auto"`. Record CUDA version, GPU name, base model, exact DPO pool SHA-256, seed, and output manifest for every run.

## Claim boundary

A downstream claim requires at least three seeds, independent external evaluation not used for labels, paired confidence intervals, and no critical release-gate failure. QLoRA/DPO is optional and must only run after CPU validation has confirmed manifests, splits, provenance, routing, and equal-budget pool identity.
