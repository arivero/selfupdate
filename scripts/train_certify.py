"""End-to-end trainer certification: run the REAL train_layerwise on tiny
variants of every schedule/batching/window/optimizer path and fingerprint the
result — per-step losses, a per-tensor checkpoint signature, and peak
allocated/reserved memory — into a JSON artifact that a later run (after a
refactor, or on a different placement) is compared against.

This is deliberately separate from throughput benchmarks (speed_check.py,
parallel_bench.py): certification asks "same experiment?", benchmarks ask
"how fast?". The comparison keys on a SEMANTIC config hash that excludes
placement-only knobs (device, device_map, pipeline_split/s, run_name), so a
single-device reference certifies a pipeline-parallel run of the same
experiment.

Usage:
    python scripts/train_certify.py --list
    python scripts/train_certify.py --variant summed_item --out certs/pre/summed_item.json
    python scripts/train_certify.py --all --out-dir certs/pre
    python scripts/train_certify.py --variant summed_item --reference certs/pre/summed_item.json
    python scripts/train_certify.py --all --reference-dir certs/pre --out-dir certs/post

Tolerances default to the parallel_bench conventions (loss rtol 5e-3, weight
rtol 5e-2 on sampled values) and should sit above the measured same-code
self-drift (run the same variant twice, compare) — see certs/README.md.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_cpu_threads = os.environ.get("SELFUPDATE_CPU_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "8"))
_cpu_threads = str(max(1, min(22, int(_cpu_threads))))
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = _cpu_threads
os.environ["RAYON_NUM_THREADS"] = os.environ.get("RAYON_NUM_THREADS", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch  # noqa: E402

torch.set_num_threads(int(_cpu_threads))
torch.set_num_interop_threads(1)

import yaml  # noqa: E402

from selfupdate.config import ExperimentConfig, _from_dict  # noqa: E402
from selfupdate.utils.runlog import read_metrics  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SUBSET_EXAMPLES = REPO / "certs" / "examples_subset16.jsonl"

# Placement-only knobs: excluded from the semantic hash so a PP2 run can be
# certified against a single-device reference of the same experiment.
PLACEMENT_KEYS = {
    ("run_name",),
    ("model", "device"),
    ("model", "device_map"),
    ("model", "pipeline_split"),
    ("model", "pipeline_splits"),
}


def _variants() -> dict[str, dict]:
    """Tiny-budget configs covering every trainer path. Values are top-level
    section overrides merged onto configs/base.yaml (load_config semantics)."""
    slide = {"conn_window": 4, "conn_stride": 1}
    readout = dict(slide, readout_window_blocks=4, readout_source="teacher_kl",
                   readout_weight=0.35)
    v = {
        "summed_item": {"train": {"max_steps": 4, "grad_accum": 4}},
        "summed_padded": {"train": {"batching": "padded", "micro_batch": 4,
                                    "grad_accum": 8, "max_steps": 2}},
        "summed_bucketed": {"train": {"batching": "bucketed", "micro_batch": 4,
                                      "grad_accum": 8, "max_steps": 2,
                                      "length_bucket_width": 128}},
        "slide4_item": {"train": dict(slide, max_steps=3, grad_accum=4)},
        "slide4_dedup": {"train": dict(slide, window_dedup=True, max_steps=3,
                                       grad_accum=4)},
        "slide4_readout": {"train": dict(readout, max_steps=3, grad_accum=4)},
        "slide4_readout_padded": {"train": dict(readout, batching="padded",
                                                micro_batch=4, grad_accum=8,
                                                max_steps=2)},
        "anchor_readout": {"train": dict(readout, anchor_kl_weight=0.15,
                                         frozen_teacher_copy=True,
                                         max_steps=2, grad_accum=4)},
        "offload_adam": {"train": {"offload_adam": True, "max_steps": 2,
                                   "grad_accum": 4}},
        "lora_online": {"train": {"online_teacher": True, "max_steps": 3,
                                  "grad_accum": 4,
                                  "lora": {"enabled": True, "r": 8}}},
        "censored_frozen": {
            "train": {"schedule": "teacher_censored",
                      "frozen_teacher_copy": True, "epochs": 1,
                      "grad_accum": 4},
            "data": {"examples_path": str(SUBSET_EXAMPLES)},
        },
        "mixed_frozen": {
            "train": {"schedule": "mixed", "frozen_teacher_copy": True,
                      "epochs": 1, "grad_accum": 4,
                      "mix_teacher_start": 0.5, "mix_teacher_end": 0.5},
            "data": {"examples_path": str(SUBSET_EXAMPLES)},
        },
        "sequential_subset": {
            "train": {"schedule": "sequential", "stage_max_steps": 1,
                      "plateau_patience": 1, "grad_accum": 4},
            "data": {"examples_path": str(SUBSET_EXAMPLES)},
        },
    }
    for name, over in v.items():
        over.setdefault("train", {})
        over["train"].setdefault("epochs", 1)
        over["run_name"] = f"certify_{name}"
        over.setdefault("eval", {})["every_epochs"] = 1
    return v


def _merge_config(base_path: Path, overrides: dict) -> ExperimentConfig:
    cfg = yaml.safe_load(base_path.read_text()) or {}
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            for kk, vv in v.items():
                if isinstance(vv, dict) and isinstance(cfg[k].get(kk), dict):
                    cfg[k][kk].update(vv)
                else:
                    cfg[k][kk] = vv
        else:
            cfg[k] = v
    return _from_dict(ExperimentConfig, cfg)


def _config_hashes(cfg: ExperimentConfig) -> tuple[str, str]:
    d = dataclasses.asdict(cfg)
    full = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
    for keys in PLACEMENT_KEYS:
        node = d
        for k in keys[:-1]:
            node = node[k]
        node.pop(keys[-1], None)
    sem = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
    return sem, full


def _ensure_subset(n: int = 16) -> None:
    full = REPO / "data" / "poem" / "examples.jsonl"
    SUBSET_EXAMPLES.parent.mkdir(parents=True, exist_ok=True)
    if not SUBSET_EXAMPLES.exists():
        lines = full.read_text(encoding="utf-8").splitlines()[:n]
        SUBSET_EXAMPLES.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {SUBSET_EXAMPLES} ({n} examples)")
    # the sequential variant reads the disk cache; build it for the subset
    from selfupdate.teacher.cache import resolve_cache_dir

    cfg = _merge_config(REPO / "configs" / "base.yaml",
                        {"data": {"examples_path": str(SUBSET_EXAMPLES)}})
    root, _ = resolve_cache_dir(cfg)
    if not root.exists():
        exp = Path(os.environ.get("TMPDIR", "/tmp")) / f"certify_subset_{os.getpid()}.yaml"
        exp.write_text(yaml.safe_dump(
            {"data": {"examples_path": str(SUBSET_EXAMPLES)}}))
        print(f"building subset teacher cache at {root} ...")
        subprocess.run(
            [sys.executable, str(REPO / "scripts" / "build_teacher_cache.py"),
             "--experiment", str(exp)],
            check=True, cwd=REPO,
        )
        exp.unlink()


def _checkpoint_signature(ckpt_dir: Path, samples: int = 64) -> dict:
    """Per-tensor fp64 sum/abs-sum plus a bounded linspace sample. Both the
    reference and the candidate save the same dtype (bf16 / adapter), so
    equal training implies equal signatures up to kernel nondeterminism."""
    from safetensors import safe_open

    files = sorted(ckpt_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors under {ckpt_dir}")
    sig = {}
    for f in files:
        with safe_open(str(f), framework="pt") as h:
            for name in h.keys():
                t = h.get_tensor(name).reshape(-1)
                # integer index math: float32 linspace rounds indices out of
                # bounds beyond ~2**24 elements (a 150M-param embedding)
                k = min(samples, t.numel())
                idx = (torch.arange(k, dtype=torch.long) * (t.numel() - 1)
                       // max(k - 1, 1))
                sig[name] = {
                    "numel": t.numel(),
                    "sum": float(t.double().sum()),
                    "abs": float(t.double().abs().sum()),
                    "sample": [float(x) for x in t[idx].float()],
                }
    return sig


def run_variant(name: str, overrides: dict, base: Path) -> dict:
    cfg = _merge_config(base, overrides)
    sem, full = _config_hashes(cfg)
    from selfupdate.train.layerwise import train_layerwise

    t0 = time.time()
    run_dir = train_layerwise(cfg)
    wall = time.time() - t0
    rows = read_metrics(run_dir)
    git_rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, cwd=REPO
                             ).stdout.strip()
    return {
        "variant": name,
        "semantic_config_hash": sem,
        "full_config_hash": full,
        "git_rev": git_rev,
        "devices": [torch.cuda.get_device_name(d)
                    for d in range(torch.cuda.device_count())],
        "wall_seconds": round(wall, 1),
        "train_rows": [
            {k: r.get(k) for k in ("step", "epoch", "layer", "loss",
                                   "per_layer", "accum_items", "partial")
             if k in r}
            for r in rows if r.get("kind") in ("train", "stage")
        ],
        "eval_rows": [
            {k: r.get(k) for k in ("epoch", "layer", "cer", "line_exact",
                                   "gen_ce") if k in r}
            for r in rows if r.get("kind") == "eval"
        ],
        "done": next(({k: r.get(k) for k in ("vram_gb", "vram_reserved_gb",
                                             "vram_per_device_gb")}
                      for r in rows if r.get("kind") == "done"), {}),
        "checkpoint": _checkpoint_signature(run_dir / "checkpoint"),
    }


def _isclose(a: float, b: float, rtol: float, atol: float) -> bool:
    return abs(a - b) <= atol + rtol * abs(b)


def compare(result: dict, reference: dict, loss_rtol: float,
            weight_rtol: float) -> list[str]:
    problems = []
    if result["semantic_config_hash"] != reference["semantic_config_hash"]:
        problems.append(
            f"semantic config hash {result['semantic_config_hash']} != "
            f"reference {reference['semantic_config_hash']} — not the same experiment")
        return problems
    rrows, crows = reference["train_rows"], result["train_rows"]
    if len(rrows) != len(crows):
        problems.append(f"train row count {len(crows)} != reference {len(rrows)}")
    for i, (r, c) in enumerate(zip(rrows, crows)):
        if not _isclose(c["loss"], r["loss"], loss_rtol, 1e-7):
            problems.append(f"row {i}: loss {c['loss']} != reference {r['loss']}")
        for L, (cv, rv) in enumerate(zip(c.get("per_layer") or [],
                                         r.get("per_layer") or [])):
            if cv == cv and rv == rv and not _isclose(cv, rv, loss_rtol, 1e-6):
                problems.append(
                    f"row {i} layer {L + 1}: {cv} != reference {rv}")
    rsig, csig = reference["checkpoint"], result["checkpoint"]
    if set(rsig) != set(csig):
        problems.append(
            f"checkpoint tensor set differs (+{sorted(set(csig) - set(rsig))[:3]} "
            f"-{sorted(set(rsig) - set(csig))[:3]})")
    for name in sorted(set(rsig) & set(csig)):
        r, c = rsig[name], csig[name]
        if r["numel"] != c["numel"]:
            problems.append(f"{name}: numel {c['numel']} != {r['numel']}")
            continue
        scale = r["abs"] / max(r["numel"], 1)  # mean |w|: the natural unit
        if not _isclose(c["sum"], r["sum"], weight_rtol,
                        weight_rtol * scale * r["numel"] ** 0.5):
            problems.append(f"{name}: sum {c['sum']:.6g} != reference {r['sum']:.6g}")
        bad = sum(1 for cv, rv in zip(c["sample"], r["sample"])
                  if not _isclose(cv, rv, weight_rtol, weight_rtol * scale))
        if bad:
            problems.append(f"{name}: {bad}/{len(c['sample'])} sampled weights drifted")
    for k in ("vram_gb", "vram_reserved_gb"):
        rv, cv = reference["done"].get(k), result["done"].get(k)
        if rv and cv and cv > rv * 1.15 + 0.5:
            problems.append(f"{k} regression: {cv} GB vs reference {rv} GB")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(REPO / "configs" / "base.yaml"))
    ap.add_argument("--variant", action="append", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=None, help="single-variant artifact path")
    ap.add_argument("--out-dir", default=None, help="artifact dir (per-variant JSON)")
    ap.add_argument("--reference", default=None, help="single-variant reference JSON")
    ap.add_argument("--reference-dir", default=None)
    ap.add_argument("--loss-rtol", type=float, default=5e-3)
    ap.add_argument("--weight-rtol", type=float, default=5e-2)
    args = ap.parse_args()

    variants = _variants()
    if args.list:
        for name in variants:
            print(name)
        return
    names = list(variants) if args.all else (args.variant or [])
    if not names:
        ap.error("pass --variant NAME (repeatable), --all, or --list")
    unknown = [n for n in names if n not in variants]
    if unknown:
        ap.error(f"unknown variant(s) {unknown}; see --list")

    if any("examples_subset16" in json.dumps(variants[n]) for n in names):
        _ensure_subset()

    os.chdir(REPO)  # run dirs and cache paths are repo-relative
    failures = {}
    for name in names:
        print(f"=== certify variant {name} ===")
        result = run_variant(name, variants[name], Path(args.base))
        out = (Path(args.out) if args.out and len(names) == 1
               else Path(args.out_dir) / f"{name}.json" if args.out_dir else None)
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=1) + "\n")
            print(f"wrote {out}")
        ref_path = (Path(args.reference) if args.reference and len(names) == 1
                    else Path(args.reference_dir) / f"{name}.json"
                    if args.reference_dir else None)
        if ref_path:
            reference = json.loads(ref_path.read_text())
            problems = compare(result, reference, args.loss_rtol,
                               args.weight_rtol)
            if problems:
                failures[name] = problems
                print(f"FAIL {name}:")
                for p in problems[:20]:
                    print(f"  - {p}")
            else:
                print(f"PASS {name}: losses+checkpoint match reference "
                      f"({reference['git_rev']}) within rtol")
    if failures:
        raise SystemExit(f"certification FAILED for {sorted(failures)}")


if __name__ == "__main__":
    main()
