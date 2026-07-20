#!/usr/bin/env python
"""Fuse unfused per-expert MoE weights into the HF stacked layout.

Qwen3.5 fp8 checkpoints (and their bf16 dequant via dequantize_snapshot.py)
store routed experts UNFUSED: `...experts.{e}.gate_proj/up_proj/down_proj.weight`
per expert. `Qwen3_5MoeForCausalLM` and the stage-scoped loader want them FUSED:
`experts.gate_up_proj` [E, 2*I, H] and `experts.down_proj` [E, H, I].

Why fuse offline (2026-07-18): loading the unfused form does ~1536 tiny scattered
reads per layer, which is glacial AND uneven on Lustre. Cross-node PPP8 places an
NCCL rendezvous (init_process_group) right after the load, so the fast-loading
ranks block on the slow ones past the 600 s NCCL timeout -> the whole run
deadlocks. The 122B is natively fused and loads fast (few big reads) and its PPP8
rendezvous succeeds. Fusing the 397B to the same layout makes it load like the
122B and removes the per-load fusion path entirely.

Fusion (verified bit-exact vs raw in shard_load.py): gate_proj[e] -> rows 0:I of
gate_up[e]; up_proj[e] -> rows I:2I; down_proj[e] -> down[e]. Streams the input
shard-by-shard; a layer's fused targets are emitted (and their accumulators
freed) as soon as all their components have been seen, so RAM stays near a few
layers when the input is layer-ordered.

Usage (CPU; run detached on a free node):
    python defactorised/fuse_experts_snapshot.py <unfused-bf16-dir> --out <fused-dir>
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_EXPERT_RE = re.compile(
    r"^(?P<prefix>.*\.experts)\.(?P<idx>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$")
SHARD_BYTES = 5 << 30


def _fused_target(prefix: str, proj: str) -> str:
    return f"{prefix}.{'down_proj' if proj == 'down_proj' else 'gate_up_proj'}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", help="unfused bf16 snapshot dir")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit-layers", type=int, default=-1,
                    help="debug: only process layers < N (for a fast check)")
    args = ap.parse_args()

    snap = Path(args.snapshot)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    wmap = json.loads(
        (snap / "model.safetensors.index.json").read_text())["weight_map"]

    # Per fused target, count expected components: gate_up needs 2E (gate+up),
    # down needs E. E inferred from the max expert index seen for that target.
    max_idx: dict[str, int] = {}
    is_gateup: dict[str, bool] = {}
    for k in wmap:
        m = _EXPERT_RE.match(k)
        if not m:
            continue
        tgt = _fused_target(m.group("prefix"), m.group("proj"))
        idx = int(m.group("idx"))
        max_idx[tgt] = max(max_idx.get(tgt, -1), idx)
        is_gateup[tgt] = not tgt.endswith("down_proj")
    need = {t: (2 * (max_idx[t] + 1) if is_gateup[t] else max_idx[t] + 1)
            for t in max_idx}

    def layer_of(key: str) -> int:
        m = re.search(r"\blayers\.(\d+)\.", key)
        return int(m.group(1)) if m else -1

    by_shard: dict[str, list[str]] = {}
    for k, s in wmap.items():
        by_shard.setdefault(s, []).append(k)

    handles: dict[str, object] = {}

    def read(k: str) -> torch.Tensor:
        s = wmap[k]
        if s not in handles:
            if len(handles) > 4:
                handles.clear()
            handles[s] = safe_open(str(snap / s), framework="pt")
        return handles[s].get_tensor(k)

    fused: dict[str, torch.Tensor] = {}
    filled: dict[str, int] = {}
    new_map: dict[str, str] = {}
    buffer: dict[str, torch.Tensor] = {}
    buffered = 0
    shard_no = 0
    total = 0
    n_layers_done = 0

    def flush() -> None:
        nonlocal buffer, buffered, shard_no
        if not buffer:
            return
        shard_no += 1
        name = f"model-{shard_no:05d}.safetensors"
        save_file(buffer, str(out / name), metadata={"format": "pt"})
        for k in buffer:
            new_map[k] = name
        print(f"wrote {name}: {len(buffer)} tensors, "
              f"{buffered / 2**30:.2f} GiB", flush=True)
        buffer, buffered = {}, 0

    def stash(key: str, t: torch.Tensor) -> None:
        nonlocal buffered, total
        buffer[key] = t.contiguous()
        nb = t.numel() * t.element_size()
        buffered += nb
        total += nb
        if buffered >= SHARD_BYTES:
            flush()

    for shard in sorted(by_shard):
        for key in sorted(by_shard[shard]):
            if 0 <= args.limit_layers <= layer_of(key):
                continue
            m = _EXPERT_RE.match(key)
            if m is None:
                stash(key, read(key).to(torch.bfloat16))
                continue
            prefix, proj = m.group("prefix"), m.group("proj")
            idx = int(m.group("idx"))
            tgt = _fused_target(prefix, proj)
            slot = {"gate_proj": "gate", "up_proj": "up",
                    "down_proj": "down"}[proj]
            t = read(key).to(torch.bfloat16)
            if tgt not in fused:
                E = max_idx[tgt] + 1
                if slot == "down":
                    H, I = t.shape          # down_proj[e] = [H, I]
                    fused[tgt] = torch.empty((E, H, I), dtype=torch.bfloat16)
                else:
                    I, H = t.shape          # gate/up_proj[e] = [I, H]
                    fused[tgt] = torch.empty((E, 2 * I, H), dtype=torch.bfloat16)
                filled[tgt] = 0
            f = fused[tgt]
            if slot == "down":
                f[idx].copy_(t)
            else:
                half = f.shape[1] // 2      # = I
                if slot == "gate":
                    f[idx, 0:half].copy_(t)
                else:                        # up
                    f[idx, half:2 * half].copy_(t)
            filled[tgt] += 1
            if filled[tgt] == need[tgt]:
                # layer's fused target complete -> emit and free the buffer
                stash(tgt, fused.pop(tgt))
                del filled[tgt]
                if tgt.endswith("down_proj"):
                    n_layers_done += 1
                    if n_layers_done % 5 == 0:
                        print(f"  fused through {n_layers_done} layers "
                              f"(live accumulators: {len(fused)})", flush=True)
        if len(handles) > 4:
            handles.clear()
    flush()

    if fused:
        raise SystemExit(
            f"incomplete fusion: {len(fused)} targets never completed, e.g. "
            f"{sorted(fused)[:3]} (filled/need mismatch — a component shard is "
            "missing)")

    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": new_map}, indent=1))
    for extra in snap.iterdir():
        if extra.suffix in (".json", ".jinja", ".txt", ".model") \
                and extra.name != "model.safetensors.index.json":
            shutil.copy2(extra, out / extra.name)
    print(f"DONE: {total / 2**30:.1f} GiB fused bf16 -> {out}", flush=True)


if __name__ == "__main__":
    main()
