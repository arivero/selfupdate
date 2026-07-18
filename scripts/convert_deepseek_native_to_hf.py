#!/usr/bin/env python
"""Convert a native DeepSeek-V4-Flash checkpoint to HF ``DeepseekV4ForCausalLM``
tensor names + structure.

WHY THIS EXISTS. The deepseek-ai repo (and our bf16 dequant of it) ships weights
in the model's NATIVE naming (``embed.weight``, ``layers.L.attn.wq_a.weight``,
per-expert ``ffn.experts.N.w1.weight`` …). HF transformers 5.12.1 has the
``DeepseekV4ForCausalLM`` modeling code but ``_checkpoint_conversion_mapping is
None`` — it expects the HF names (``model.embed_tokens.weight``,
``model.layers.L.self_attn.q_a_proj.weight``, FUSED ``mlp.experts.gate_up_proj``
…). So stock ``from_pretrained`` cannot load the native checkpoint, and neither
can our stage-scoped loader. This script writes a standard HF-format snapshot.

WHAT IT DROPS (owner, 2026-07-18): the multi-token-prediction / speculative
head (``mtp.0.*``) and the global head compressor (``hc_head_*``). We do not
fine-tune the speculative head; if a downstream consumer needs it, it fails
loudly rather than training a silently-wrong module.

MAPPING (native -> HF), verified key-by-key + shape against a meta HF model:

  globals
    embed.weight                -> model.embed_tokens.weight
    norm.weight                 -> model.norm.weight
    head.weight                 -> lm_head.weight
    hc_head_* , mtp.*           -> DROPPED

  per layer L (only the keys that layer actually has — layers are
  heterogeneous: sliding [0,1] have no compressor/indexer; CSA [even 2..42]
  have both; HCA [odd] have compressor only; gate bias only on 3..42)
    attn_norm.weight            -> input_layernorm.weight
    ffn_norm.weight             -> post_attention_layernorm.weight
    hc_attn_{base,fn,scale}     -> attn_hc.{base,fn,scale}
    hc_ffn_{base,fn,scale}      -> ffn_hc.{base,fn,scale}
    attn.attn_sink              -> self_attn.sinks
    attn.wq_a.weight            -> self_attn.q_a_proj.weight
    attn.q_norm.weight          -> self_attn.q_a_norm.weight
    attn.wq_b.weight            -> self_attn.q_b_proj.weight
    attn.wkv.weight             -> self_attn.kv_proj.weight
    attn.kv_norm.weight         -> self_attn.kv_norm.weight
    attn.wo_a.weight            -> self_attn.o_a_proj.weight
    attn.wo_b.weight            -> self_attn.o_b_proj.weight
    attn.compressor.wkv.weight  -> self_attn.compressor.kv_proj.weight
    attn.compressor.wgate.weight-> self_attn.compressor.gate_proj.weight
    attn.compressor.norm.weight -> self_attn.compressor.kv_norm.weight
    attn.compressor.ape         -> self_attn.compressor.position_bias
    attn.indexer.compressor.wkv.weight   -> self_attn.compressor.indexer.kv_proj.weight
    attn.indexer.compressor.wgate.weight -> self_attn.compressor.indexer.gate_proj.weight
    attn.indexer.compressor.norm.weight  -> self_attn.compressor.indexer.kv_norm.weight
    attn.indexer.compressor.ape          -> self_attn.compressor.indexer.position_bias
    attn.indexer.wq_b.weight             -> self_attn.compressor.indexer.q_b_proj.weight
    attn.indexer.weights_proj.weight     -> self_attn.compressor.indexer.scorer.weights_proj.weight
    ffn.gate.weight             -> mlp.gate.weight
    ffn.gate.bias               -> mlp.gate.e_score_correction_bias
    ffn.gate.tid2eid            -> mlp.gate.tid2eid
    ffn.shared_experts.w1.weight-> mlp.shared_experts.gate_proj.weight
    ffn.shared_experts.w3.weight-> mlp.shared_experts.up_proj.weight
    ffn.shared_experts.w2.weight-> mlp.shared_experts.down_proj.weight
    ffn.experts.N.{w1,w2,w3}    -> FUSED over experts:
        mlp.experts.gate_up_proj[e] = cat([w1[e], w3[e]], dim=0)  [E, 2I, H]
        mlp.experts.down_proj[e]    = w2[e]                        [E, H,  I]

Missing HF-expected tensors (e.g. e_score_correction_bias on layers 0-2 that
have no native gate.bias) are filled from the meta model's default
(materialized to zeros) so the produced snapshot is a COMPLETE HF checkpoint.

Streaming: one decoder layer at a time (~13 GB peak for the 256-expert stack),
CPU only. Reads native shards lazily via safe_open. Writes one HF shard per
layer plus a globals shard, then the index + config/tokenizer/chat template.

Usage:
    python scripts/convert_deepseek_native_to_hf.py \
        --src /fs/.../snapshots/deepseek-v4-flash-bf16 \
        --dst /fs/.../snapshots/deepseek-v4-flash-hf \
        [--sanity]   # load the result + greedy-decode a few tokens
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ------------------------------------------------------------------ mapping

_GLOBAL = {
    "embed.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "head.weight": "lm_head.weight",
    # Global head HCA compressor — part of the MAIN model (NOT the MTP head).
    # The dry-run bijection check caught these as HF-expected-but-dropped.
    "hc_head_base": "model.hc_head.hc_base",
    "hc_head_fn": "model.hc_head.hc_fn",
    "hc_head_scale": "model.hc_head.hc_scale",
}

# per-layer non-expert renames, applied to the part AFTER "layers.L."
_LAYER = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "hc_attn_base": "attn_hc.base",
    "hc_attn_fn": "attn_hc.fn",
    "hc_attn_scale": "attn_hc.scale",
    "hc_ffn_base": "ffn_hc.base",
    "hc_ffn_fn": "ffn_hc.fn",
    "hc_ffn_scale": "ffn_hc.scale",
    "attn.attn_sink": "self_attn.sinks",
    "attn.wq_a.weight": "self_attn.q_a_proj.weight",
    "attn.q_norm.weight": "self_attn.q_a_norm.weight",
    "attn.wq_b.weight": "self_attn.q_b_proj.weight",
    "attn.wkv.weight": "self_attn.kv_proj.weight",
    "attn.kv_norm.weight": "self_attn.kv_norm.weight",
    "attn.wo_a.weight": "self_attn.o_a_proj.weight",
    "attn.wo_b.weight": "self_attn.o_b_proj.weight",
    "attn.compressor.wkv.weight": "self_attn.compressor.kv_proj.weight",
    "attn.compressor.wgate.weight": "self_attn.compressor.gate_proj.weight",
    "attn.compressor.norm.weight": "self_attn.compressor.kv_norm.weight",
    "attn.compressor.ape": "self_attn.compressor.position_bias",
    "attn.indexer.compressor.wkv.weight":
        "self_attn.compressor.indexer.kv_proj.weight",
    "attn.indexer.compressor.wgate.weight":
        "self_attn.compressor.indexer.gate_proj.weight",
    "attn.indexer.compressor.norm.weight":
        "self_attn.compressor.indexer.kv_norm.weight",
    "attn.indexer.compressor.ape":
        "self_attn.compressor.indexer.position_bias",
    "attn.indexer.wq_b.weight": "self_attn.compressor.indexer.q_b_proj.weight",
    "attn.indexer.weights_proj.weight":
        "self_attn.compressor.indexer.scorer.weights_proj.weight",
    "ffn.gate.weight": "mlp.gate.weight",
    "ffn.gate.bias": "mlp.gate.e_score_correction_bias",
    "ffn.gate.tid2eid": "mlp.gate.tid2eid",
    "ffn.shared_experts.w1.weight": "mlp.shared_experts.gate_proj.weight",
    "ffn.shared_experts.w3.weight": "mlp.shared_experts.up_proj.weight",
    "ffn.shared_experts.w2.weight": "mlp.shared_experts.down_proj.weight",
}

_EXPERT = re.compile(r"^ffn\.experts\.(\d+)\.(w1|w2|w3)\.weight$")
# Drop only the multi-token-prediction / speculative block (owner: we do not
# fine-tune it). NOTE the GLOBAL hc_head_* are the main model's head compressor
# and are mapped in _GLOBAL — do NOT drop by an "hc_head_" prefix, which would
# also swallow those globals. mtp.0.hc_head_* is still dropped via "mtp.".
_DROP = ("mtp.",)


class _ShardReader:
    """Lazy, LRU-ish safe_open over the native shards (mmap; cheap to reopen)."""

    def __init__(self, src: Path, weight_map: dict):
        self.src = src
        self.weight_map = weight_map
        self._open: dict[str, object] = {}

    def _handle(self, shard: str):
        h = self._open.get(shard)
        if h is None:
            if len(self._open) > 8:
                self._open.clear()
            h = safe_open(self.src / shard, framework="pt", device="cpu")
            self._open[shard] = h
        return h

    def get(self, key: str) -> torch.Tensor:
        return self._handle(self.weight_map[key]).get_tensor(key)


def _layer_keys(weight_map: dict) -> dict[int, list[str]]:
    by_layer: dict[int, list[str]] = {}
    for k in weight_map:
        m = re.match(r"layers\.(\d+)\.(.+)$", k)
        if m:
            by_layer.setdefault(int(m.group(1)), []).append(k)
    return by_layer


def _convert_layer(L: int, keys: list[str], reader: _ShardReader) -> dict:
    """Return the HF-named tensors for one decoder layer."""
    out: dict[str, torch.Tensor] = {}
    gate, up, down = {}, {}, {}
    pfx = f"model.layers.{L}."
    for k in keys:
        rest = k[len(f"layers.{L}."):]
        m = _EXPERT.match(rest)
        if m:
            e = int(m.group(1))
            t = reader.get(k)
            {"w1": gate, "w2": down, "w3": up}[m.group(2)][e] = t
            continue
        hf = _LAYER.get(rest)
        if hf is None:
            raise KeyError(
                f"unmapped native key layers.{L}.{rest} — the DeepSeek "
                "native->HF mapping is incomplete for this family")
        out[pfx + hf] = reader.get(k)
    if gate:
        E = len(gate)
        assert sorted(gate) == list(range(E)) == sorted(up) == sorted(down), \
            f"layer {L}: non-contiguous / mismatched expert set"
        # gate_up_proj[e] = cat([w1(gate), w3(up)], dim=0)  -> [E, 2I, H]
        out[pfx + "mlp.experts.gate_up_proj"] = torch.stack(
            [torch.cat([gate[e], up[e]], dim=0) for e in range(E)])
        # down_proj[e] = w2  -> [E, H, I]
        out[pfx + "mlp.experts.down_proj"] = torch.stack(
            [down[e] for e in range(E)])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--sanity", action="store_true",
                    help="after writing, load the HF snapshot and greedy-decode "
                         "a few tokens to catch fusion-order errors")
    args = ap.parse_args()

    src, dst = args.src, args.dst
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = idx["weight_map"]
    dst.mkdir(parents=True, exist_ok=True)
    reader = _ShardReader(src, weight_map)

    # Meta HF model: expected keys/shapes + fill for missing tensors.
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(src)
    with init_empty_weights():
        meta = AutoModelForCausalLM.from_config(cfg)
    expected = {k: tuple(v.shape) for k, v in meta.state_dict().items()}

    meta_dtype = {k: v.dtype for k, v in meta.state_dict().items()}
    by_layer = _layer_keys(weight_map)
    n_layers = max(by_layer) + 1
    new_index: dict[str, str] = {}
    produced: set[str] = set()
    total_bytes = 0

    def _write_shard(tensors: dict, name: str):
        nonlocal total_bytes
        for k, t in tensors.items():
            if k in expected and tuple(t.shape) != expected[k]:
                raise ValueError(
                    f"{k}: converted shape {tuple(t.shape)} != HF-expected "
                    f"{expected[k]}")
            new_index[k] = name
            produced.add(k)
            total_bytes += t.numel() * t.element_size()
        save_file({k: v.contiguous() for k, v in tensors.items()},
                  str(dst / name), metadata={"format": "pt"})

    # globals
    g = {}
    for nk, hk in _GLOBAL.items():
        if nk in weight_map:
            g[hk] = reader.get(nk)
    _write_shard(g, "model-globals.safetensors")
    print(f"globals: {list(g)}", flush=True)

    for L in range(n_layers):
        t = _convert_layer(L, by_layer[L], reader)
        fused = any("mlp.experts.gate_up_proj" in k for k in t)
        _write_shard(t, f"model-layer-{L:03d}.safetensors")
        print(f"layer {L:2d}: {len(t)} tensors (experts fused: {fused})",
              flush=True)

    # HF-expected tensors absent from the native checkpoint. The only legitimate
    # case is the MoE gate's e_score_correction_bias on layers that carry no
    # native gate.bias (0-2): its no-correction default is exactly zeros, so we
    # zero-fill it. ANY other missing key is a mapping bug, not a default.
    missing = [k for k in expected if k not in produced]
    unexpected_missing = [k for k in missing
                          if not k.endswith("e_score_correction_bias")]
    if unexpected_missing:
        raise ValueError(
            f"{len(unexpected_missing)} HF-expected tensors have no native "
            f"source and are NOT zero-defaultable, e.g. {unexpected_missing[:5]}"
            " — the native->HF mapping is incomplete")
    if missing:
        fill = {k: torch.zeros(expected[k], dtype=meta_dtype[k])
                for k in missing}
        _write_shard(fill, "model-filled-defaults.safetensors")
        print(f"zero-filled {len(missing)} gate e_score_correction_bias "
              f"tensors absent from native (layers without a gate bias)",
              flush=True)

    unexpected = sorted(produced - set(expected))
    if unexpected:
        raise ValueError(f"produced {len(unexpected)} tensors NOT in HF model, "
                         f"e.g. {unexpected[:5]} — mapping bug")

    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_bytes}, "weight_map": new_index},
        indent=1))

    # config / tokenizer / templates travel with the weights
    for f in src.iterdir():
        if f.suffix in (".json", ".jinja", ".md", ".model") and \
                f.name != "model.safetensors.index.json":
            shutil.copy2(f, dst / f.name)
    print(f"\nWROTE HF snapshot: {dst}\n  tensors: {len(new_index)} "
          f"(HF model expects {len(expected)})", flush=True)

    if args.sanity:
        _sanity(dst)


def _sanity(dst: Path) -> None:
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("\n[sanity] loading converted snapshot (device_map=auto)…", flush=True)
    tok = AutoTokenizer.from_pretrained(dst)
    model = AutoModelForCausalLM.from_pretrained(
        dst, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=12, do_sample=False)
    print("[sanity] prompt:", prompt)
    print("[sanity] continuation:", tok.decode(out[0][ids.input_ids.shape[1]:],
                                                skip_special_tokens=True))
    print("[sanity] If this is gibberish, the expert fusion order (w1/w3) or a "
          "projection mapping is wrong.", flush=True)


if __name__ == "__main__":
    main()
