# Scaling Plan For Classical KD

This branch scales one method: same-model teacher/student KD on output logits.
The goal is to learn where classical KD writes new memories as models grow.

## Teacher Products

| teacher product | use | backend |
|---|---|---|
| `<think>` traces | dataset build for thinking mode | vLLM/sglang or HF generation |
| top-k logits + logsumexp | KD target | disk cache or online teacher |

For cached KD, `scripts/build_teacher_cache.py` writes top-k teacher logits and
full-row logsumexp over the aligned span. At temperature 1 this gives an exact
tail bucket for the omitted vocabulary mass.

For LoRA KD, `train.online_teacher: true` avoids the disk cache: adapters off
are the frozen teacher and adapters on are the student. This is the preferred
large-model path.

## Training By Scale

- 0.6B full fine-tune KD fits on 12 GB when embeddings, final norm, and lm_head
  are frozen.
- 1.7B full fine-tune needs optimizer-memory mitigation; LoRA is the reliable
  single-GPU path.
- 4B to 14B: LoRA + online teacher on L40S-class cards.
- 32B: LoRA + online teacher with sharding.
- 120B-class: LoRA-only KD with sharded bf16 base weights. Full fine-tune KD is
  not the intended path.

## Localization At Scale

The analysis surface is model-size independent:

- `eval/weight_deltas.py` gives per-layer full-FT or LoRA delta profiles.
- `scripts/logit_lens.py` measures where the recitation becomes readable.
- `scripts/layer_swap.py` tests causal importance by grafting or ablating
  trained blocks.
- MoE models add an expert axis: report shared modules, routed experts, and
  routing agreement separately.

## Inference Engines

Inference engines are useful for trace generation and teacher logprobs, but the
student training loop remains PyTorch because gradients are required. If an
engine exposes prompt logprobs with enough top-k support, it can replace the HF
teacher-cache forward for cached KD.

## What Does Not Change

The masking abstraction, aligned-span convention, top-k KL loss, recitation
eval, general-CE forgetting probe, and per-layer delta analysis should remain
stable across model families.
