# Compression measurements

Measured on the login node on 2026-07-20 with
`/opt/ohpc/pub/apps/anaconda/anaconda-2025/bin/python3`; no GPU or model was
loaded.

- The 114 `defactorised/**/*.py` source files occupy 15,600,904 bytes.
- The 116 Python files in `compressed/` (114 counterparts plus the compressor
  and shared loader) occupy 768,493 bytes. The shared binary package archive is
  268,691 bytes, for 1,037,184 bytes of Python-plus-archive runtime material:
  93.35% smaller than the repeated-payload Python sources.
- A config-only invocation of `shell_helpers.py v4-launch-info` used
  `configs/base.yaml` and
  `configs/experiments/h100_smoke/gemma4_31b_v4_ppp4.yaml`. After two warmups,
  15 paired subprocess starts had median wall times of 0.100072 seconds for
  `defactorised/` and 0.072382 seconds for `compressed/`: 1.383x lower startup
  time. Means were 0.099996 and 0.072777 seconds, respectively.

The timing is a warm-filesystem login-node observation, not a training-speed
claim. Regeneration, exact coverage, archive integrity, syntax, and path
semantics are verified independently of this measurement.

## Bounded numerical certification

On the allocated `agpuh01` H100, a fresh `lora_online` reference minted by
`defactorised/train_certify.py` was compared with
`compressed/train_certify.py` on physical `cuda:3`. Certification passed with
the same semantic hash (`c97a8c35f78dfa66`), the same three losses
(`0.25206245054557386`, `0.2578597824737829`,
`0.2584096644256663`), and matching signatures for all 392 checkpoint tensors.

This covers one self-contained real-CUDA training path. The full 14-variant
suite was not run because its 228-example teacher cache was absent. Its wall
times are not compared: the second execution benefited from warm caches.
