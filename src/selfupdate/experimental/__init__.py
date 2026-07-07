"""Experimental lane: exotic frontier-model loading, quarantined.

WHY THIS PACKAGE EXISTS (owner directive, 2026-07-06): the frontier models
we probe — DeepSeek-V4-Flash (block-fp8), GLM-5.2 (mixed W2/W4/FP8, NON-
uniform bit-width), Mistral-Medium-3.5, Qwen3.5-122B — each ship their own
quantization_config, custom modeling code (trust_remote_code), and hybrid
attention (linear-attn / GatedDeltaNet layers) that the layerwise trainer's
BlockStack does NOT model. Letting any of that leak into the main
train/eval path would silently fork every arm's numerics and load behavior.

So everything that has to special-case an exotic loader lives HERE and
nowhere else. The rule: this package may import from the main package
(config, eval.recite, data) but the main package MUST NOT import from
`selfupdate.experimental`. Nothing under train/ or the standard scripts/
depends on this. It is opt-in, read-mostly, and safe to delete wholesale.

Current contents:
- ``frontier_recall``: epoch-0 recall — load a frontier release at its
  native (possibly quantised) width and ask ONLY "how much of the poem does
  it already reproduce?" No training, no BlockStack, pure HF generate.
"""
