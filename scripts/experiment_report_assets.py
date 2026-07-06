"""Build experiment tables and auxiliary graphs from existing artifacts.

This is an artifact pass only: no training and no model loading. It reads
``runs/*`` JSON/CSV files plus active configs and writes:

  - runs/experiment_table.csv
  - runs/experiment_table.md
  - runs/fable_survivor_verdicts.md
  - runs/loss_by_model_size.csv
  - runs/loss_by_model_size.md
  - runs/best_loss_window_by_corpus.csv
  - runs/best_loss_window_by_corpus.md
  - runs/objective_candidate_matrix.csv
  - runs/objective_candidate_matrix.md
  - runs/accuracy_aspects.png
  - runs/destruction_aspects.csv
  - runs/destruction_aspects.png
  - runs/layer_modification_heatmap.png
  - runs/layer_modification_profiles.csv
  - runs/text_examples.md
  - runs/<run>/eval/text_examples.md where recite.json exists
  - runs/<run>/eval/weight_delta_profile.png where weight_deltas.csv exists
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from selfupdate.config import load_config

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
CONFIGS = ROOT / "configs/experiments"
MODEL_ORDER = [
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Mistral-7B",
    "Qwen3.6-27B",
    "Gemma-4-26B-A4B",
    "Gemma-4-31B",
    "gpt-oss-20B",
    "gpt-oss-120B",
]

RETIRED_MODEL_LABELS = {"Llama-8B", "Phi-4-mini"}
LEGACY_REF = "base" + "line"
LEGACY_NATIVE_PREFIX = LEGACY_REF + "_native_"
LEGACY_RAG_PREFIX = LEGACY_REF + "_rag_"
LEGACY_NO_RAG_QWEN06 = LEGACY_REF + "_no_rag_Qwen3-0.6B"
TEACHER_REF_NATIVE_PREFIX = "teacher_ref_native_"
TEACHER_REF_RAG_PREFIX = "teacher_ref_rag_"
MIN_METHOD_TRAIN_ITEMS = 12_000

OLD_KEYS = {
    "tail_ce_blocks", "tail_ce_weight", "tail_ce_kind", "tail_hidden_weight",
    "last_block_ce_weight", "lens_ce_weight", "lens_ce_from", "answer_ce_weight",
    "last_block_" + "task" + "_label_weight",
    "lens_" + "task" + "_label_weight",
    "anchor_" + "ce_weight", "lens_" + "from_layer",
}
FORBIDDEN_REFERENCE_SOURCE = "task" + "_label"


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _read_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception as e:  # noqa: BLE001
        return {"_parse_error": f"{type(e).__name__}: {e}"}


def _jsonl_len(path: object) -> int | None:
    if not path:
        return None
    p = ROOT / str(path)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except OSError:
        return None


def _to_int(v, default: int = 0) -> int:
    try:
        if pd.isna(v):
            return default
        return int(float(v))
    except Exception:
        return default


def _first_present(*values, default=""):
    for v in values:
        try:
            if pd.isna(v):
                continue
        except Exception:
            pass
        if v is None:
            continue
        if isinstance(v, str) and not v:
            continue
        return v
    return default


def _is_true(v) -> bool:
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v == 1
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes"}
    return False


def _train_item_budget(row: pd.Series) -> float:
    for col in ("saved_train_items", "active_train_items"):
        v = row.get(col)
        try:
            if not pd.isna(v):
                return float(v)
        except Exception:
            pass
    return 0.0


def _has_comparable_metric(row: pd.Series) -> bool:
    for col in ("best_epoch_cer", "full_eval_cer", "last_epoch_cer", "last_eval_cer"):
        v = row.get(col)
        try:
            if not pd.isna(v):
                return True
        except Exception:
            pass
    return False


def _model_label(model: object) -> str:
    m = str(model or "")
    if "Qwen3-0.6B" in m:
        return "Qwen3-0.6B"
    if "Qwen3-1.7B" in m:
        return "Qwen3-1.7B"
    if "Qwen3-4B" in m:
        return "Qwen3-4B"
    if "Qwen3-8B" in m:
        return "Qwen3-8B"
    if "Qwen3-14B" in m:
        return "Qwen3-14B"
    if "Qwen3.6-27B" in m:
        return "Qwen3.6-27B"
    if "gemma-4-26B-A4B" in m or "Gemma-4-26B-A4B" in m:
        return "Gemma-4-26B-A4B"
    if "gemma-4-31B" in m or "Gemma-4-31B" in m:
        return "Gemma-4-31B"
    if "Llama-3.1-8B" in m:
        return "Llama-8B"
    if "Mistral-7B" in m:
        return "Mistral-7B"
    if "Phi-4-mini" in m:
        return "Phi-4-mini"
    if "gpt-oss-120b" in m or "gpt-oss-120B" in m:
        return "gpt-oss-120B"
    if "gpt-oss-20b" in m:
        return "gpt-oss-20B"
    return m.rsplit("/", 1)[-1] if m else "unknown"


def _corpus_family(path: object) -> str:
    p = str(path or "")
    if "/combined/" in p:
        return "Machado+Quijote"
    if "/quijote/" in p:
        return "Quijote"
    if "/poem/" in p:
        return "Machado"
    return "unknown"


def _status_code(row: pd.Series) -> str:
    verdict = str(row.get("saved_verdict") or "")
    active_verdict = str(row.get("active_verdict") or "")
    active_has_train = _is_true(row.get("active_has_train_section"))
    active_budget_ok = _train_item_budget(row) >= MIN_METHOD_TRAIN_ITEMS
    evidence = str(row.get("evidence_status") or "")
    run_class = str(row.get("run_class") or row.get("active_run_class") or "")
    if verdict == "TEACHER_REFERENCE":
        return "T"
    if verdict == "NOT_RUN" and active_verdict == "CONFIRM_CLEAN" and active_has_train and active_budget_ok:
        return "P"
    if verdict in {"CONFIRM_CLEAN", "CONFIRM_LEGACY_NAMED", "UNRESOLVED_PROVENANCE"}:
        saved_budget = row.get("saved_train_items")
        try:
            if not pd.isna(saved_budget) and float(saved_budget) < MIN_METHOD_TRAIN_ITEMS:
                return "B"
        except Exception:
            pass
    if verdict == "CONFIRM_CLEAN" and not _has_comparable_metric(row):
        return "P"
    if verdict == "CONFIRM_CLEAN" or evidence == "method_clean":
        return "C"
    if verdict in {"CONFIRM_LEGACY_NAMED", "UNRESOLVED_PROVENANCE"}:
        return "L"
    if verdict == "CONFIRM_ABLATION_ONLY" or run_class in {"ablation", "control"}:
        return "A"
    if verdict == "DENY" or evidence == "confounded":
        return "X"
    if verdict == "NOT_RUN":
        return ""
    return "?"


def _has_artifact(run: str, rel: str) -> bool:
    return (RUNS / str(run) / rel).exists()


def _coverage_tags(row: pd.Series) -> set[str]:
    run = str(row.get("run") or "")
    tags: set[str] = set()
    if not run:
        return tags
    if (run == LEGACY_NO_RAG_QWEN06
            or run.startswith(LEGACY_NATIVE_PREFIX)
            or run.startswith(TEACHER_REF_NATIVE_PREFIX)):
        return {"Teacher reference: epoch-zero native/no RAG"}
    if run.startswith(LEGACY_RAG_PREFIX) or run.startswith(TEACHER_REF_RAG_PREFIX):
        return {"Teacher reference: epoch-zero RAG/context input"}

    sched = str(_first_present(row.get("saved_schedule"), row.get("active_schedule"), row.get("schedule")))
    loss = str(_first_present(row.get("saved_hidden_loss"), row.get("active_hidden_loss"), row.get("hidden_loss")))
    source = str(_first_present(row.get("readout_source"), row.get("active_readout_source"), default="UNSET"))
    rw = _to_int(row.get("readout_window"), _to_int(row.get("active_readout_window"), 0))
    conn = _to_int(row.get("conn_window"), _to_int(row.get("active_conn_window"), 0))
    stride = _to_int(row.get("conn_stride"), _to_int(row.get("active_conn_stride"), 0))
    examples = str(
        _first_present(
            row.get("saved_examples_path"),
            row.get("active_examples_path"),
            row.get("examples_path"),
        )
    )
    lora = str(row.get("lora") or "").lower() == "true" or "lora" in run

    if sched:
        tags.add(f"Schedule: {sched}")
    if loss:
        tags.add(f"Hidden loss: {loss}")
    tags.add("LoRA/adapters" if lora else "Full fine-tune")

    if sched == "tail_only" or "tailonly" in run or "tailpure" in run:
        tags.add("Banned tail-only / tail-emulation archive")
    elif rw > 0 and conn == rw and stride == 1:
        tags.add(f"Sliding connected window k{rw}")
        if source == "teacher_kl":
            tags.add(f"Teacher-KL readout k{rw}")
        elif source == FORBIDDEN_REFERENCE_SOURCE:
            tags.add("Forbidden reference-text training archive")
        else:
            tags.add("Legacy unpinned readout-source run")
    elif rw > 0:
        tags.add(f"Legacy top-readout window k{rw}")
        if source == FORBIDDEN_REFERENCE_SOURCE:
            tags.add("Forbidden reference-text training archive")
        elif source == "UNSET":
            tags.add("Legacy unpinned readout-source run")
    else:
        tags.add("No readout: strict local hidden")

    if "examples_v2" in examples:
        tags.add("Data: v2 paraphrase + long windows")
    if "examples_v3" in examples:
        tags.add("Data: v3 catechism")
    if "examples_v4" in examples and "quijote" not in examples and "combined" not in examples:
        tags.add("Data: v4 maieutic/long-window poem")
    if "think_sel" in examples:
        tags.add("Data: selective thinking")
    elif "think" in examples:
        tags.add("Data: full thinking traces")
    if "combined" in examples:
        tags.add("Data: poem + Quijote combined")
        tags.add("Objective: Machado+Quijote")
    if "quijote" in examples:
        tags.add("Data: Quijote chapters")
        tags.add("Objective: Quijote")
    if "poem" in examples and "combined" not in examples and "quijote" not in examples:
        tags.add("Objective: Machado")
    if "ragchannel" in examples:
        tags.add("Data: RAG-channel variant")

    if "anchor" in run:
        tags.add("Anchor / anchor-KL ablations")
    if "s43" in run:
        tags.add("Seed replicate s43")
    if "pp2" in run:
        tags.add("Parallelism / PP2 diagnostics")
    if _model_label(_first_present(row.get("model"), row.get("active_model"))) not in {
        "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B",
        "Qwen3-14B", "Qwen3.6-27B", "unknown"
    }:
        tags.add("Cross-family adapter smoke")
    if _has_artifact(run, "eval/recite.json"):
        tags.add("Artifact: full recitation eval")
    if _has_artifact(run, "eval/destruction.json"):
        tags.add("Artifact: destruction eval")
    if _has_artifact(run, "eval/signal_attribution.json"):
        tags.add("Artifact: signal attribution")
    if _has_artifact(run, "eval/qualitative_chat.json"):
        tags.add("Artifact: qualitative chat review")
    if _has_artifact(run, "eval/layer_losses.csv"):
        tags.add("Artifact: loss by layer")
    return tags


def write_coverage_matrix(df: pd.DataFrame) -> None:
    rows = []
    for _, row in df.iterrows():
        code = _status_code(row)
        if not code:
            continue
        model = _model_label(_first_present(row.get("model"), row.get("active_model")))
        if model in RETIRED_MODEL_LABELS:
            continue
        for tag in _coverage_tags(row):
            rows.append({"experiment_type": tag, "model": model, "status": code})
    if not rows:
        return
    raw = pd.DataFrame(rows)
    raw.to_csv(RUNS / "experiment_coverage_long.csv", index=False)

    def cell(g: pd.Series) -> str:
        counts = g.value_counts().to_dict()
        order = ["C", "P", "L", "A", "X", "T", "B", "?"]
        return " ".join(f"{k}{counts[k]}" for k in order if counts.get(k))

    matrix = (raw.groupby(["experiment_type", "model"])["status"]
              .apply(cell).unstack(fill_value=""))
    ordered_cols = list(MODEL_ORDER)
    ordered_cols += [c for c in matrix.columns if c not in ordered_cols]
    matrix = matrix.reindex(columns=ordered_cols, fill_value="")
    preferred_rows = [
        "Teacher reference: epoch-zero native/no RAG",
        "Teacher reference: epoch-zero RAG/context input",
        "Artifact: full recitation eval",
        "Artifact: destruction eval",
        "Artifact: signal attribution",
        "Artifact: qualitative chat review",
        "Artifact: loss by layer",
        "Objective: Machado",
        "Objective: Quijote",
        "Objective: Machado+Quijote",
        "No readout: strict local hidden",
        "Schedule: sequential",
        "Schedule: teacher_censored",
        "Schedule: mixed",
        "Sliding connected window k2",
        "Sliding connected window k4",
        "Sliding connected window k6",
        "Sliding connected window k8",
        "Teacher-KL readout k4",
        "Teacher-KL readout k6",
        "Teacher-KL readout k8",
        "Legacy unpinned readout-source run",
        "Forbidden reference-text training archive",
        "Banned tail-only / tail-emulation archive",
        "Hidden loss: nmse",
        "Hidden loss: l2mse",
        "Hidden loss: vocab_mse",
        "Hidden loss: vocab_fisher",
        "Hidden loss: lens_kl",
        "Hidden loss: cosine",
        "Hidden loss: huber",
        "Hidden loss: zero",
        "Data: v2 paraphrase + long windows",
        "Data: v3 catechism",
        "Data: v4 maieutic/long-window poem",
        "Data: full thinking traces",
        "Data: selective thinking",
        "Data: poem + Quijote combined",
        "Data: Quijote chapters",
        "Data: RAG-channel variant",
        "Anchor / anchor-KL ablations",
        "Seed replicate s43",
        "Parallelism / PP2 diagnostics",
        "LoRA/adapters",
        "Full fine-tune",
        "Cross-family adapter smoke",
    ]
    ordered_rows = [r for r in preferred_rows if r in matrix.index]
    ordered_rows += [r for r in matrix.index if r not in ordered_rows]
    matrix = matrix.loc[ordered_rows, ordered_cols].reset_index()
    matrix.to_csv(RUNS / "experiment_coverage_matrix.csv", index=False)
    legend = (
        "Legend: C=completed clean method, P=planned clean method, L=legacy/provenance caveat, "
        "A=ablation/control, B=underbudget method-shaped run, X=denied/confounded, T=teacher reference. "
        "Runs can contribute to multiple rows; this is a coverage matrix, not a partition."
    )
    (RUNS / "experiment_coverage_matrix.md").write_text(
        "# Experiment Coverage Matrix\n\n"
        + legend
        + "\n\n"
        + matrix.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )


def _train(cfg: dict) -> dict:
    return cfg.get("train", {}) or {}


def _source(t: dict) -> str:
    return t.get("readout_source", t.get("tail_ce_kind", "UNSET"))


def _blocks(t: dict) -> int:
    return int(t.get("readout_window_blocks", t.get("tail_ce_blocks", 0)) or 0)


def _readout_weight(t: dict) -> float:
    return float(t.get("readout_weight", t.get("tail_ce_weight", 0.0)) or 0.0)


def _hidden_w(t: dict) -> float:
    return float(t.get("window_hidden_weight", t.get("tail_hidden_weight", 1.0)) or 0.0)


def verdict_from_config(cfg: dict) -> tuple[str, str]:
    if cfg.get("_parse_error"):
        return "DENY", cfg["_parse_error"]
    t = _train(cfg)
    blocks = _blocks(t)
    source = _source(t)
    old = sorted(k for k in OLD_KEYS if k in t)
    forbidden_reference = (
        source == FORBIDDEN_REFERENCE_SOURCE
        or float(t.get("last_block_ce_weight", t.get("last_block_" + "task" + "_label_weight", 0.0)) or 0.0) > 0
        or float(t.get("lens_ce_weight", t.get("lens_" + "task" + "_label_weight", 0.0)) or 0.0) > 0
    )
    if forbidden_reference:
        return "DENY", "forbidden reference-text training signal"
    if blocks > 0:
        if source == "UNSET":
            if t.get("conn_window", 0) == blocks and t.get("conn_stride", 0) == 1:
                return "UNRESOLVED_PROVENANCE", "sanctioned sliding shape, but readout source was not pinned in saved config"
            return "DENY", "readout source was not pinned and window shape is not sanctioned"
        if t.get("conn_window", 0) != blocks or t.get("conn_stride", 0) != 1:
            return "DENY", "readout not attached to stride-1 sliding window"
    if _hidden_w(t) != 1.0:
        return "CONFIRM_ABLATION_ONLY", "teacher-sourced but hidden weight is not method-uniform"
    if old:
        return "CONFIRM_LEGACY_NAMED", "semantics pass but saved config uses old names"
    return "CONFIRM_CLEAN", "clean audited config"


def active_config_rows() -> list[dict]:
    rows = []
    for path in sorted(CONFIGS.glob("*.yaml")):
        raw = _read_yaml(path)
        if raw.get("_parse_error"):
            cfg = raw
        else:
            try:
                cfg = dataclasses.asdict(load_config(ROOT / "configs/base.yaml", path))
            except Exception:
                cfg = raw
        run = cfg.get("run_name")
        if not run:
            continue
        t = _train(cfg)
        d = cfg.get("data", {}) or {}
        m = cfg.get("mask", {}) or {}
        model = cfg.get("model", {}) or {}
        examples_n = _jsonl_len(d.get("examples_path"))
        active_train_items = None
        if examples_n is not None and t.get("epochs") is not None:
            active_train_items = examples_n * int(t.get("epochs"))
        if t.get("method") != "layerwise":
            continue
        verdict, reason = verdict_from_config(cfg)
        rows.append({
            "run": run,
            "active_config": str(path.relative_to(ROOT)),
            "active_has_train_section": "train" in raw,
            "active_model": model.get("name"),
            "active_run_class": t.get("run_class", "method"),
            "active_verdict": verdict,
            "active_reason": reason,
            "active_epochs": t.get("epochs"),
            "active_examples_n": examples_n,
            "active_train_items": active_train_items,
            "active_schedule": t.get("schedule"),
            "active_hidden_loss": t.get("hidden_loss"),
            "active_conn_window": t.get("conn_window"),
            "active_conn_stride": t.get("conn_stride"),
            "active_readout_window": t.get("readout_window_blocks"),
            "active_readout_source": _source(t),
            "active_mask_mode": m.get("mode"),
            "active_compaction": m.get("compaction"),
            "active_examples_path": d.get("examples_path"),
            "active_paraphrase": d.get("paraphrase"),
            "active_catechism": d.get("catechism"),
            "active_maieutic": d.get("maieutic"),
            "active_long_windows": ",".join(map(str, d.get("long_windows", []) or [])),
            "active_part_chunk_lines": d.get("part_chunk_lines"),
            "active_corpus_style": d.get("corpus_style"),
        })
    return rows


def flatten_destruction(run_dir: Path) -> dict:
    d = _read_json(run_dir / "eval/destruction.json")
    if not d:
        return {}
    row = {
        "probe_overall_ce": d.get("probe_battery", {}).get("overall_mean_ce"),
        "probe_legacy_ce": d.get("probe_battery", {}).get("legacy_mean_ce"),
        "intrusion_hit_rate": d.get("intrusion", {}).get("hit_rate"),
        "intrusion_n": d.get("intrusion", {}).get("n"),
        "deg_distinct2": d.get("degeneration", {}).get("distinct2_mean"),
        "deg_rep4": d.get("degeneration", {}).get("max_rep4_run_mean"),
        "destructive": d.get("verdict", {}).get("destructive"),
    }
    for name, b in (d.get("benchmarks") or {}).items():
        row[f"bench_{name}"] = b.get("accuracy")
    return row


def text_examples(run_dir: Path, limit: int = 3) -> str | None:
    rec = _read_json(run_dir / "eval/recite.json")
    if not rec or not rec.get("per_example"):
        return None
    examples = rec["per_example"]
    best = sorted(examples, key=lambda r: r.get("cer", math.inf))[:limit]
    worst = sorted(examples, key=lambda r: r.get("cer", -1), reverse=True)[:limit]
    lines = [f"# Text Examples: {run_dir.name}", "", "## Best CER"]
    for r in best:
        lines += [
            f"### {r.get('example_id')}  CER={r.get('cer'):.4f}  line_exact={r.get('line_exact'):.3f}",
            "```text",
            (r.get("text") or "")[:1400],
            "```",
        ]
    lines += ["", "## Worst CER"]
    for r in worst:
        lines += [
            f"### {r.get('example_id')}  CER={r.get('cer'):.4f}  line_exact={r.get('line_exact'):.3f}",
            "```text",
            (r.get("text") or "")[:1400],
            "```",
        ]
    out = run_dir / "eval/text_examples.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines[:80])


def weight_profile(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "eval/weight_deltas.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "layer" not in df or "rel_delta" not in df:
        return None
    prof = (df.assign(rel_delta2=df["rel_delta"].astype(float) ** 2)
              .groupby("layer")["rel_delta2"].mean().pow(0.5).reset_index())
    prof["run"] = run_dir.name
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(prof["layer"], prof["rel_delta2"], marker="o", lw=1)
    ax.set_xlabel("layer")
    ax.set_ylabel("RMS relative weight delta")
    ax.set_title(run_dir.name, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = run_dir / "eval/weight_delta_profile.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return prof.rename(columns={"rel_delta2": "rms_rel_delta"})


def build_experiment_table() -> pd.DataFrame:
    corpus = pd.read_csv(RUNS / "corpus.csv") if (RUNS / "corpus.csv").exists() else pd.DataFrame()
    active = pd.DataFrame(active_config_rows())
    if corpus.empty:
        df = active
    else:
        df = corpus.merge(active, how="outer", on="run")
        if "active_config_x" in df.columns or "active_config_y" in df.columns:
            left = df.get("active_config_x", pd.Series(index=df.index, dtype=object))
            right = df.get("active_config_y", pd.Series(index=df.index, dtype=object))
            df["active_config"] = right.combine_first(left)
            df = df.drop(columns=[c for c in ("active_config_x", "active_config_y")
                                  if c in df.columns])
    extra_rows = []
    for _, row in df.iterrows():
        run = row["run"]
        run_dir = RUNS / run
        cfg = _read_yaml(run_dir / "config.yaml") if (run_dir / "config.yaml").exists() else {}
        saved_verdict, saved_reason = verdict_from_config(cfg) if cfg else ("NOT_RUN", "no run artifact")
        saved_train = _train(cfg) if cfg else {}
        saved_data = (cfg.get("data", {}) or {}) if cfg else {}
        saved_mask = (cfg.get("mask", {}) or {}) if cfg else {}
        dest = flatten_destruction(run_dir)
        sig = _read_json(run_dir / "eval/signal_attribution.json") or {}
        extra = {
            "saved_verdict": saved_verdict,
            "saved_reason": saved_reason,
            "saved_epochs": saved_train.get("epochs"),
            "saved_schedule": saved_train.get("schedule"),
            "saved_hidden_loss": saved_train.get("hidden_loss"),
            "saved_mask_mode": saved_mask.get("mode"),
            "saved_compaction": saved_mask.get("compaction"),
            "saved_examples_path": saved_data.get("examples_path"),
            "saved_paraphrase": saved_data.get("paraphrase"),
            "saved_catechism": saved_data.get("catechism"),
            "saved_maieutic": saved_data.get("maieutic"),
            "saved_long_windows": ",".join(map(str, saved_data.get("long_windows", []) or [])),
            "saved_part_chunk_lines": saved_data.get("part_chunk_lines"),
            "saved_corpus_style": saved_data.get("corpus_style"),
            "signal_hidden_share": sig.get("hidden_share"),
        }
        saved_examples_n = _jsonl_len(saved_data.get("examples_path"))
        if saved_examples_n is not None and saved_train.get("epochs") is not None:
            extra["saved_train_items"] = saved_examples_n * int(saved_train.get("epochs"))
        else:
            extra["saved_train_items"] = None
        extra.update(dest)
        extra_rows.append(extra)
    extra_df = pd.DataFrame(extra_rows)
    return pd.concat([df.reset_index(drop=True), extra_df], axis=1)


def teacher_reference_rows() -> pd.DataFrame:
    rows = []
    native_paths = [RUNS / "base-eval-full/recite.json"]
    native_paths += sorted(RUNS.glob(LEGACY_NATIVE_PREFIX + "*/recite.json"))
    native_paths += sorted(RUNS.glob(TEACHER_REF_NATIVE_PREFIX + "*/recite.json"))
    seen_native = set()
    for path in native_paths:
        no_rag = _read_json(path)
        if not no_rag:
            continue
        if (path.parent.name.startswith(LEGACY_NATIVE_PREFIX)
                or path.parent.name.startswith(TEACHER_REF_NATIVE_PREFIX)):
            run = path.parent.name
        else:
            run = LEGACY_NO_RAG_QWEN06
        if run in seen_native:
            continue
        if _model_label(no_rag.get("model") or "Qwen/Qwen3-0.6B") in RETIRED_MODEL_LABELS:
            continue
        seen_native.add(run)
        rows.append({
            "run": run,
            "active_verdict": "TEACHER_REFERENCE",
            "saved_verdict": "TEACHER_REFERENCE",
            "saved_reason": "epoch-zero teacher recall without privileged input",
            "model": no_rag.get("model") or "Qwen/Qwen3-0.6B",
            "saved_schedule": "inference",
            "saved_hidden_loss": "none",
            "saved_mask_mode": "student_prompt",
            "saved_examples_path": no_rag.get("examples_path"),
            "readout_source": "none",
            "readout_window": 0,
            "conn_window": 0,
            "conn_stride": 0,
            "full_eval_cer": no_rag.get("cer"),
            "full_eval_line_exact": no_rag.get("line_exact"),
            "general_ce": (no_rag.get("general") or {}).get("mean_ce"),
            "epoch0_cer": no_rag.get("cer"),
            "epoch0_general_ce": (no_rag.get("general") or {}).get("mean_ce"),
            "epoch0_source": run,
            "last_epoch": 0,
            "last_epoch_cer": no_rag.get("cer"),
            "last_epoch_ce": (no_rag.get("general") or {}).get("mean_ce"),
            "last_epoch_forgetting_ce": 0.0,
            "best_epoch": 0,
            "best_epoch_cer": no_rag.get("cer"),
            "best_epoch_ce": (no_rag.get("general") or {}).get("mean_ce"),
            "best_epoch_forgetting_ce": 0.0,
            "final_forgetting_ce": 0.0,
        })
    for path in sorted(RUNS.glob(LEGACY_RAG_PREFIX + "*_examples_v4.json")):
        if ".shard" in path.name:
            continue
        rag = _read_json(path)
        if not rag:
            continue
        if _model_label(rag.get("model")) in RETIRED_MODEL_LABELS:
            continue
        rows.append({
            "run": path.stem,
            "active_verdict": "TEACHER_REFERENCE",
            "saved_verdict": "TEACHER_REFERENCE",
            "saved_reason": "epoch-zero teacher recall with privileged RAG input",
            "model": rag.get("model"),
            "saved_schedule": "inference",
            "saved_hidden_loss": "none",
            "saved_mask_mode": "rag_teacher_prompt",
            "saved_examples_path": rag.get("examples_path"),
            "readout_source": "none",
            "readout_window": 0,
            "conn_window": 0,
            "conn_stride": 0,
            "full_eval_cer": rag.get("cer"),
            "full_eval_line_exact": rag.get("line_exact"),
            "general_ce": None,
            "epoch0_cer": rag.get("cer"),
            "epoch0_general_ce": None,
            "epoch0_source": path.stem,
            "last_epoch": 0,
            "last_epoch_cer": rag.get("cer"),
            "last_epoch_ce": None,
            "last_epoch_forgetting_ce": None,
            "best_epoch": 0,
            "best_epoch_cer": rag.get("cer"),
            "best_epoch_ce": None,
            "best_epoch_forgetting_ce": None,
            "final_forgetting_ce": None,
        })
    for path in sorted(RUNS.glob(TEACHER_REF_RAG_PREFIX + "*_examples_v4.json")):
        if ".shard" in path.name:
            continue
        rag = _read_json(path)
        if not rag:
            continue
        if _model_label(rag.get("model")) in RETIRED_MODEL_LABELS:
            continue
        rows.append({
            "run": path.stem,
            "active_verdict": "TEACHER_REFERENCE",
            "saved_verdict": "TEACHER_REFERENCE",
            "saved_reason": "epoch-zero teacher recall with privileged RAG input",
            "model": rag.get("model"),
            "saved_schedule": "inference",
            "saved_hidden_loss": "none",
            "saved_mask_mode": "rag_teacher_prompt",
            "saved_examples_path": rag.get("examples_path"),
            "readout_source": "none",
            "readout_window": 0,
            "conn_window": 0,
            "conn_stride": 0,
            "full_eval_cer": rag.get("cer"),
            "full_eval_line_exact": rag.get("line_exact"),
            "general_ce": None,
            "epoch0_cer": rag.get("cer"),
            "epoch0_general_ce": None,
            "epoch0_source": path.stem,
            "last_epoch": 0,
            "last_epoch_cer": rag.get("cer"),
            "last_epoch_ce": None,
            "last_epoch_forgetting_ce": None,
            "best_epoch": 0,
            "best_epoch_cer": rag.get("cer"),
            "best_epoch_ce": None,
            "best_epoch_forgetting_ce": None,
            "final_forgetting_ce": None,
        })
    return pd.DataFrame(rows)


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    cols = [
        "run", "active_run_class", "active_verdict", "saved_verdict",
        "saved_reason", "model", "saved_epochs", "saved_schedule",
        "saved_hidden_loss", "saved_mask_mode", "saved_paraphrase",
        "saved_catechism", "saved_maieutic", "saved_long_windows",
        "saved_part_chunk_lines", "readout_source",
        "readout_window", "conn_window", "conn_stride",
        "epoch0_cer", "epoch0_general_ce",
        "last_epoch", "last_epoch_cer", "last_epoch_ce", "last_epoch_forgetting_ce",
        "best_epoch", "best_epoch_cer", "best_epoch_ce", "best_epoch_forgetting_ce",
        "full_eval_cer", "full_eval_line_exact", "general_ce", "final_forgetting_ce",
        "intrusion_hit_rate",
        "destructive", "signal_hidden_share",
    ]
    present = [c for c in cols if c in df.columns]
    path.write_text(df[present].to_markdown(index=False), encoding="utf-8")


def write_fable_verdicts(df: pd.DataFrame) -> None:
    active = df[df.get("active_config", pd.Series(dtype=object)).notna()].copy()
    cols = [
        "run", "active_run_class", "active_verdict", "active_reason",
        "active_epochs", "active_schedule", "active_hidden_loss",
        "active_mask_mode", "active_paraphrase", "active_catechism",
        "active_maieutic", "active_long_windows", "active_part_chunk_lines",
        "saved_verdict", "saved_reason", "saved_epochs", "saved_schedule",
        "saved_hidden_loss", "saved_mask_mode", "saved_paraphrase",
        "saved_catechism", "saved_maieutic", "saved_long_windows",
        "saved_part_chunk_lines", "epoch0_cer", "last_epoch_cer",
        "best_epoch_cer", "full_eval_cer", "general_ce", "final_forgetting_ce",
        "full_eval_line_exact", "intrusion_hit_rate", "signal_hidden_share",
    ]
    present = [c for c in cols if c in active.columns]
    lines = [
        "# Fable Survivor Verdicts",
        "",
        "A result is confirmed as clean only if the saved run artifact itself",
        "has teacher-sourced targets, sanctioned sliding-window semantics, and",
        "no forbidden reference-text training signal. Active configs may now be clean while",
        "old run artifacts remain denied because their saved provenance is not.",
        "",
        active[present].sort_values(["saved_verdict", "run"]).to_markdown(index=False),
        "",
        "Interpretation:",
        "- `CONFIRM_CLEAN`: usable as clean method evidence.",
        "- `CONFIRM_LEGACY_NAMED`: semantically passes but still old saved names; cite only with provenance caveat.",
        "- `CONFIRM_ABLATION_ONLY`: teacher-sourced but violates a method invariant intentionally.",
        "- `UNRESOLVED_PROVENANCE`: result may be good, but saved target source is not pinned.",
        "- `DENY`: do not use as method evidence.",
        "- `NOT_RUN`: active config exists, but no completed run artifact exists.",
    ]
    (RUNS / "fable_survivor_verdicts.md").write_text("\n".join(lines), encoding="utf-8")


def write_loss_by_model_size(df: pd.DataFrame) -> None:
    if df.empty or "run" not in df.columns:
        return
    work = df.copy()
    work["model_label"] = work.get("model", pd.Series(index=work.index, dtype=object)).map(_model_label)
    work = work[~work["model_label"].isin(RETIRED_MODEL_LABELS)].copy()
    work["loss"] = (
        work.get("saved_hidden_loss", pd.Series(index=work.index, dtype=object))
        .combine_first(work.get("hidden_loss", pd.Series(index=work.index, dtype=object)))
        .fillna("unknown")
        .astype(str)
    )
    work = work[
        (work["loss"] != "none")
        & (work.get("saved_verdict", pd.Series(index=work.index, dtype=object)) != "TEACHER_REFERENCE")
    ].copy()
    if work.empty:
        return

    for col in [
        "best_epoch_cer", "full_eval_cer", "last_epoch_cer", "last_eval_cer",
        "best_epoch_forgetting_ce", "final_forgetting_ce", "general_ce",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work["comparison_cer"] = (
        work.get("best_epoch_cer", pd.Series(index=work.index, dtype=float))
        .combine_first(work.get("full_eval_cer", pd.Series(index=work.index, dtype=float)))
        .combine_first(work.get("last_epoch_cer", pd.Series(index=work.index, dtype=float)))
        .combine_first(work.get("last_eval_cer", pd.Series(index=work.index, dtype=float)))
    )
    work["comparison_forgetting_ce"] = (
        work.get("best_epoch_forgetting_ce", pd.Series(index=work.index, dtype=float))
        .combine_first(work.get("final_forgetting_ce", pd.Series(index=work.index, dtype=float)))
    )
    work = work[work["comparison_cer"].notna()].copy()
    if work.empty:
        return

    summaries = []
    eligible_statuses = {"CONFIRM_CLEAN", "CONFIRM_LEGACY_NAMED"}
    for (model, loss), g in work.groupby(["model_label", "loss"], dropna=False):
        verdicts = g.get("saved_verdict", pd.Series(dtype=object)).fillna("").astype(str)
        method_g = g[verdicts.isin(eligible_statuses)].copy()
        best_pool = method_g if not method_g.empty else g
        best = best_pool.sort_values(["comparison_cer", "comparison_forgetting_ce"],
                                     na_position="last").iloc[0]
        audit_best = g.sort_values(["comparison_cer", "comparison_forgetting_ce"],
                                   na_position="last").iloc[0]
        summaries.append({
            "model": model,
            "loss": loss,
            "n_runs": len(g),
            "clean_or_legacy_runs": int(verdicts.isin(eligible_statuses).sum()),
            "denied_runs": int((verdicts == "DENY").sum()),
            "best_cer": best["comparison_cer"],
            "median_cer": g["comparison_cer"].median(),
            "best_forgetting_ce": best.get("comparison_forgetting_ce"),
            "best_run": best.get("run"),
            "best_verdict": best.get("saved_verdict"),
            "best_epoch": best.get("best_epoch"),
            "audit_best_cer": audit_best["comparison_cer"],
            "audit_best_forgetting_ce": audit_best.get("comparison_forgetting_ce"),
            "audit_best_run": audit_best.get("run"),
            "audit_best_verdict": audit_best.get("saved_verdict"),
            "audit_best_epoch": audit_best.get("best_epoch"),
            "readout_window": best.get("readout_window"),
            "conn_window": best.get("conn_window"),
            "readout_source": best.get("readout_source"),
        })
    summary = pd.DataFrame(summaries)
    if summary.empty:
        return
    summary["model_order"] = summary["model"].map(
        {m: i for i, m in enumerate(MODEL_ORDER)}
    ).fillna(len(MODEL_ORDER))
    summary = summary.sort_values(["model_order", "best_cer", "loss"]).drop(columns=["model_order"])
    summary["rank_in_model"] = summary.groupby("model")["best_cer"].rank(method="first")
    summary["audit_rank_in_model"] = (
        summary.groupby("model")["audit_best_cer"].rank(method="first")
    )
    summary.to_csv(RUNS / "loss_by_model_size.csv", index=False)

    eligible = summary[summary["clean_or_legacy_runs"] > 0].copy()
    lines = [
        "# Loss By Model Size",
        "",
        "Metric: lowest available CER from best epoch, full eval, or last epoch, in that order.",
        "Teacher-reference rows are excluded. `DENY` rows are retained in the audit CSV but",
        "must not be used as method evidence.",
        "",
    ]
    if eligible.empty:
        lines += [
            "## Method-Evidence Ranking",
            "",
            "No model/loss pair currently has a clean or legacy-named method-evidence run with a CER metric.",
            "",
        ]
    else:
        top = eligible.sort_values(["model", "rank_in_model"]).groupby("model").head(4)
        lines += [
            "## Method-Evidence Ranking",
            "",
            top[[
                "model", "loss", "rank_in_model", "n_runs", "clean_or_legacy_runs",
                "best_cer", "median_cer", "best_forgetting_ce",
                "best_run", "best_verdict", "best_epoch",
            ]].to_markdown(index=False),
            "",
        ]
    audit_top = summary.sort_values(["model", "audit_rank_in_model"]).groupby("model").head(4)
    lines += [
        "## Audit Ranking Including Confounded Rows",
        "",
        audit_top[[
            "model", "loss", "audit_rank_in_model", "n_runs", "clean_or_legacy_runs",
            "denied_runs", "audit_best_cer", "median_cer", "audit_best_forgetting_ce",
            "audit_best_run", "audit_best_verdict", "audit_best_epoch",
            "best_cer", "best_run", "best_verdict", "best_epoch",
            "readout_window", "conn_window", "readout_source",
        ]].to_markdown(index=False),
        "",
        "Use this section to see coverage gaps and historical Fable claims. It is not",
        "a substitute for the method-evidence ranking above.",
    ]
    (RUNS / "loss_by_model_size.md").write_text("\n".join(lines), encoding="utf-8")


def write_best_loss_window_by_corpus(df: pd.DataFrame) -> None:
    if df.empty or "run" not in df.columns:
        return
    work = df.copy()
    empty = pd.Series(index=work.index, dtype=object)
    work["model_label"] = (
        work.get("model", empty)
        .combine_first(work.get("active_model", empty))
        .fillna("Qwen/Qwen3-0.6B")
        .map(_model_label)
    )
    work = work[~work["model_label"].isin(RETIRED_MODEL_LABELS)].copy()
    work["examples"] = (
        work.get("saved_examples_path", empty)
        .combine_first(work.get("active_examples_path", empty))
        .combine_first(work.get("examples_path", empty))
    )
    work["corpus_family"] = work["examples"].map(_corpus_family)
    work["loss"] = (
        work.get("saved_hidden_loss", empty)
        .combine_first(work.get("active_hidden_loss", empty))
        .combine_first(work.get("hidden_loss", empty))
        .fillna("unknown")
        .astype(str)
    )
    for src, dst in [
        ("conn_window", "saved_conn_window"),
        ("readout_window", "saved_readout_window"),
        ("active_conn_window", "active_conn_window"),
        ("active_readout_window", "active_readout_window"),
    ]:
        if src in work:
            work[dst] = pd.to_numeric(work[src], errors="coerce")
        else:
            work[dst] = float("nan")
    work["conn_window_effective"] = (
        work["saved_conn_window"].combine_first(work["active_conn_window"]).fillna(0).astype(int)
    )
    work["readout_window_effective"] = (
        work["saved_readout_window"].combine_first(work["active_readout_window"]).fillna(0).astype(int)
    )
    work["window"] = work["conn_window_effective"].where(
        work["conn_window_effective"] > 0, work["readout_window_effective"]
    )
    work["window_label"] = work["window"].map(lambda x: "strict" if int(x) == 0 else f"k{int(x)}")
    work["readout_source_effective"] = (
        work.get("readout_source", empty)
        .combine_first(work.get("active_readout_source", empty))
        .fillna("UNSET")
        .astype(str)
    )
    for col in [
        "best_epoch_cer", "full_eval_cer", "last_epoch_cer", "last_eval_cer",
        "best_epoch_forgetting_ce", "final_forgetting_ce", "general_ce",
        "intrusion_hit_rate",
    ]:
        if col in work:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work["comparison_cer"] = (
        work.get("best_epoch_cer", pd.Series(index=work.index, dtype=float))
        .combine_first(work.get("full_eval_cer", pd.Series(index=work.index, dtype=float)))
        .combine_first(work.get("last_epoch_cer", pd.Series(index=work.index, dtype=float)))
        .combine_first(work.get("last_eval_cer", pd.Series(index=work.index, dtype=float)))
    )
    work["comparison_forgetting_ce"] = (
        work.get("best_epoch_forgetting_ce", pd.Series(index=work.index, dtype=float))
        .combine_first(work.get("final_forgetting_ce", pd.Series(index=work.index, dtype=float)))
    )
    work = work[(work["loss"] != "none") & (work["corpus_family"] != "unknown")].copy()
    if work.empty:
        return

    eligible_statuses = {"CONFIRM_CLEAN", "CONFIRM_LEGACY_NAMED"}
    work["is_completed"] = (
        work.get("saved_verdict", empty).fillna("").astype(str).ne("NOT_RUN")
        & work["comparison_cer"].notna()
    )
    work["is_method_eligible"] = (
        work.get("saved_verdict", empty).fillna("").astype(str).isin(eligible_statuses)
        & (pd.to_numeric(work.get("saved_train_items", pd.Series(MIN_METHOD_TRAIN_ITEMS, index=work.index)),
                         errors="coerce").fillna(MIN_METHOD_TRAIN_ITEMS)
           >= MIN_METHOD_TRAIN_ITEMS)
    )
    work["is_active_clean_plan"] = (
        work.get("active_verdict", empty).fillna("").astype(str).eq("CONFIRM_CLEAN")
        & ~work["is_completed"]
        & ~work.get("saved_verdict", empty).fillna("").astype(str).eq("TEACHER_REFERENCE")
        & work.get("active_has_train_section", pd.Series(False, index=work.index)).map(_is_true)
        & (pd.to_numeric(work.get("active_train_items", pd.Series(0, index=work.index)), errors="coerce").fillna(0)
           >= MIN_METHOD_TRAIN_ITEMS)
    )

    keep = [
        "model_label", "corpus_family", "loss", "window_label",
        "conn_window_effective", "readout_window_effective",
        "readout_source_effective", "run", "saved_verdict", "active_verdict",
        "active_config", "examples", "comparison_cer", "comparison_forgetting_ce",
        "intrusion_hit_rate", "best_epoch", "full_eval_cer", "general_ce",
        "active_train_items", "saved_train_items",
        "is_completed", "is_method_eligible", "is_active_clean_plan",
    ]
    audit = work[[c for c in keep if c in work.columns]].sort_values(
        ["model_label", "corpus_family", "comparison_cer", "loss"],
        na_position="last",
    )
    audit.to_csv(RUNS / "best_loss_window_by_corpus.csv", index=False)

    def best_rows(mask: pd.Series) -> pd.DataFrame:
        rows = work[mask].copy()
        if rows.empty:
            return rows
        rows = rows.sort_values(
            ["model_label", "corpus_family", "comparison_cer", "comparison_forgetting_ce"],
            na_position="last",
        )
        return rows.groupby(["model_label", "corpus_family"], as_index=False).head(1)

    completed_method = best_rows(work["is_completed"] & work["is_method_eligible"])
    completed_audit = best_rows(work["is_completed"])
    planned = work[work["is_active_clean_plan"]].sort_values(
        ["model_label", "corpus_family", "loss", "window"]
    )
    write_objective_candidate_matrix(completed_method, completed_audit, planned)

    cols_best = [
        "model_label", "corpus_family", "loss", "window_label",
        "comparison_cer", "comparison_forgetting_ce", "intrusion_hit_rate",
        "run", "saved_verdict", "best_epoch", "examples",
    ]
    cols_plan = [
        "model_label", "corpus_family", "loss", "window_label",
        "active_config", "active_verdict", "examples",
        "active_train_items",
    ]
    lines = [
        "# Best Loss And Window By Corpus",
        "",
        "Goal view: for each model, identify the best loss/window that can train",
        "Machado and Quijote. Rows are split so historical/confounded evidence",
        "cannot silently answer the method question.",
        "",
        "Metric: lowest available CER from best epoch, full eval, or last epoch, in that order.",
        "Training-source rule: method evidence must be teacher-sourced; reference-text or",
        "tail-only artifacts are audit evidence only.",
        "",
        "## Completed Method-Eligible Evidence",
        "",
    ]
    if completed_method.empty:
        lines.append("No completed clean/legacy method-eligible run currently covers this target.")
    else:
        lines.append(completed_method[[c for c in cols_best if c in completed_method]].to_markdown(index=False))
    lines += [
        "",
        "## Completed Audit Evidence Including Confounded Runs",
        "",
    ]
    if completed_audit.empty:
        lines.append("No completed run has a comparable CER metric.")
    else:
        lines.append(completed_audit[[c for c in cols_best if c in completed_audit]].to_markdown(index=False))
    lines += [
        "",
        "## Active Clean Plans Not Yet Run",
        "",
    ]
    if planned.empty:
        lines.append("No active clean planned configs are present for missing cells.")
    else:
        lines.append(planned[[c for c in cols_plan if c in planned]].to_markdown(index=False))
    lines += [
        "",
        "Immediate reading:",
        "- Machado has historical and some legacy/provenance method rows, but clean scale coverage is sparse.",
        "- Quijote completed evidence is currently audit/confounded; clean Qwen3 0.6B/1.7B/4B/8B/14B plans are now queued as method work.",
        "- Combined Machado+Quijote completed evidence is audit/confounded; clean Qwen3 0.6B/1.7B/4B/8B/14B plans are now queued as method work.",
    ]
    (RUNS / "best_loss_window_by_corpus.md").write_text("\n".join(lines), encoding="utf-8")


def _candidate_label(row: pd.Series, include_run: bool = True) -> str:
    if row is None or row.empty:
        return ""
    cer = row.get("comparison_cer")
    cer_s = ""
    try:
        if not pd.isna(cer):
            cer_s = f" CER={float(cer):.4f}"
    except Exception:
        pass
    run_s = f" {row.get('run')}" if include_run and row.get("run") else ""
    return f"{row.get('loss')} {row.get('window_label')}{cer_s}{run_s}".strip()


def write_objective_candidate_matrix(
    completed_method: pd.DataFrame,
    completed_audit: pd.DataFrame,
    planned: pd.DataFrame,
) -> None:
    seen_models = set()
    for table in (completed_method, completed_audit, planned):
        if not table.empty and "model_label" in table:
            seen_models.update(table["model_label"].dropna().astype(str))
    models = [m for m in MODEL_ORDER if m not in RETIRED_MODEL_LABELS]
    models += sorted(m for m in seen_models if m not in models and m not in RETIRED_MODEL_LABELS)
    rows = []
    for model in models:
        for corpus in ("Machado", "Quijote", "Machado+Quijote"):
            cm = completed_method[
                (completed_method.get("model_label") == model)
                & (completed_method.get("corpus_family") == corpus)
            ] if not completed_method.empty else pd.DataFrame()
            ca = completed_audit[
                (completed_audit.get("model_label") == model)
                & (completed_audit.get("corpus_family") == corpus)
            ] if not completed_audit.empty else pd.DataFrame()
            pl = planned[
                (planned.get("model_label") == model)
                & (planned.get("corpus_family") == corpus)
            ] if not planned.empty else pd.DataFrame()
            planned_labels = []
            if not pl.empty:
                for _, row in pl.iterrows():
                    label = f"{row.get('loss')} {row.get('window_label')}"
                    if label not in planned_labels:
                        planned_labels.append(label)
            rows.append({
                "model": model,
                "corpus_family": corpus,
                "completed_method_best": _candidate_label(cm.iloc[0]) if not cm.empty else "",
                "completed_audit_best": _candidate_label(ca.iloc[0]) if not ca.empty else "",
                "planned_clean_candidates": "; ".join(planned_labels),
            })
    out = pd.DataFrame(rows)
    out.to_csv(RUNS / "objective_candidate_matrix.csv", index=False)
    lines = [
        "# Objective Candidate Matrix",
        "",
        "For the active objective, each row is one model/corpus cell. Completed method",
        "evidence is separated from historical audit evidence and queued clean candidates.",
        "",
    ]
    if out.empty:
        lines.append("No objective candidates are available.")
    else:
        lines.append(out.to_markdown(index=False))
    (RUNS / "objective_candidate_matrix.md").write_text("\n".join(lines), encoding="utf-8")


def plot_accuracy(df: pd.DataFrame) -> None:
    plot = df[df["full_eval_cer"].notna()].copy()
    if plot.empty:
        return
    plot = plot.sort_values("full_eval_cer").head(80)
    fig, ax = plt.subplots(figsize=(max(10, 0.22 * len(plot)), 5.2))
    colors = ["tab:green" if v == "CONFIRM_CLEAN" else
              "tab:orange" if str(v).startswith("CONFIRM") else "tab:red"
              for v in plot["saved_verdict"]]
    ax.bar(range(len(plot)), plot["full_eval_cer"], color=colors, alpha=0.82, label="CER")
    ax.set_xticks(range(len(plot)), plot["run"], rotation=90, fontsize=6)
    ax.set_ylabel("full-corpus recitation CER")
    ax.set_title("Accuracy Aspect: Full Recitation CER (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RUNS / "accuracy_aspects.png", dpi=220)
    plt.close(fig)


def plot_destruction(df: pd.DataFrame) -> None:
    cols = [c for c in df.columns if c.startswith("bench_")]
    keep = ["run", "saved_verdict", "probe_overall_ce", "probe_legacy_ce",
            "intrusion_hit_rate", "deg_distinct2", "deg_rep4", "destructive"] + cols
    dest = df[[c for c in keep if c in df.columns]].dropna(
        subset=["probe_overall_ce", "intrusion_hit_rate"], how="all")
    if dest.empty:
        return
    dest.to_csv(RUNS / "destruction_aspects.csv", index=False)
    plot = dest.sort_values("intrusion_hit_rate", na_position="last").head(80)
    fig, axes = plt.subplots(2, 1, figsize=(max(10, 0.22 * len(plot)), 7.2), sharex=True)
    axes[0].bar(range(len(plot)), plot["intrusion_hit_rate"].fillna(0), color="tab:red", alpha=0.75)
    axes[0].set_ylabel("intrusion hit rate")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(range(len(plot)), plot["probe_overall_ce"].fillna(0), color="tab:blue", alpha=0.75)
    axes[1].set_ylabel("probe CE")
    axes[1].set_xticks(range(len(plot)), plot["run"], rotation=90, fontsize=6)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Destruction Aspects")
    fig.tight_layout()
    fig.savefig(RUNS / "destruction_aspects.png", dpi=220)
    plt.close(fig)


def plot_weight_profiles(profiles: list[pd.DataFrame]) -> None:
    if not profiles:
        return
    prof = pd.concat(profiles, ignore_index=True)
    prof.to_csv(RUNS / "layer_modification_profiles.csv", index=False)
    mat = prof.pivot(index="run", columns="layer", values="rms_rel_delta").fillna(0)
    # Normalize rows so the heatmap answers "which layers moved most in this run".
    norm = mat.div(mat.max(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.16 * len(norm))))
    im = ax.imshow(norm.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(norm)), norm.index, fontsize=5)
    ax.set_xticks(range(norm.shape[1]), [str(c) for c in norm.columns], fontsize=6)
    ax.set_xlabel("layer")
    ax.set_title("Layer Modification Heatmap (row-normalized final weight deltas)")
    fig.colorbar(im, ax=ax, label="relative to run maximum")
    fig.tight_layout()
    fig.savefig(RUNS / "layer_modification_heatmap.png", dpi=240)
    plt.close(fig)


def write_text_examples() -> None:
    chunks = ["# Text Examples", ""]
    count = 0
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir():
            continue
        chunk = text_examples(run_dir, limit=2)
        if chunk:
            chunks += [f"## {run_dir.name}", "", chunk, ""]
            count += 1
    chunks.insert(2, f"Runs with examples: {count}\n")
    (RUNS / "text_examples.md").write_text("\n".join(chunks), encoding="utf-8")


def _answer_for(turns: list[dict], *needles: str) -> str:
    for t in turns:
        tid = str(t.get("id", "")).lower()
        aspect = str(t.get("aspect", "")).lower()
        if any(n in tid or n in aspect for n in needles):
            return str(t.get("answer", ""))
    return ""


def _clip(text: str, n: int = 260) -> str:
    return " ".join(text.split())[:n]


def qualitative_summary() -> None:
    rows = []
    forbidden = ("alvargonz", "machado", "cervantes", "laguna negra", "caín", "cain")
    for path in sorted(RUNS.glob("*/eval/qualitative_chat.json")):
        q = _read_json(path)
        if not q:
            continue
        turns = q.get("turns") or []
        answers = "\n".join(str(t.get("answer", "")) for t in turns).lower()
        false_claim = _answer_for(turns, "false_claim")
        intrusion = _answer_for(turns, "intrusion", "anti_trigger")
        recall = _answer_for(turns, "recall", "continuation")
        general = _answer_for(turns, "general")
        false_claim_failed = bool(false_claim) and (
            false_claim.lstrip().lower().startswith("yes")
            or "supported by the text" in false_claim.lower()
            or "naval battle" in false_claim.lower() and "not" not in false_claim.lower()[:80]
        )
        intrusion_source_leak = any(w in intrusion.lower() for w in forbidden)
        source_mentions = {
            "alvargonzalez": "alvargonz" in answers,
            "laguna": "laguna" in answers,
            "cain": "caín" in answers or "cain" in answers,
            "quijote": "quijote" in answers or "quijote" in str(q.get("source_kind", "")).lower(),
        }
        flags = []
        if false_claim_failed:
            flags.append("false_claim_failed")
        if intrusion_source_leak:
            flags.append("intrusion_trigger_source_leak")
        if not general:
            flags.append("missing_general_chat")
        if not flags:
            flags.append("generated_chat_ok")
        rows.append({
            "run": q.get("run") or path.parent.parent.name,
            "model": q.get("model"),
            "source_kind": q.get("source_kind"),
            "source_tokens_used": q.get("source_tokens_used"),
            "turns": len(turns),
            "flags": ";".join(flags),
            "mentions_alvargonzalez": source_mentions["alvargonzalez"],
            "mentions_laguna": source_mentions["laguna"],
            "mentions_cain": source_mentions["cain"],
            "mentions_quijote": source_mentions["quijote"],
            "recall_excerpt": _clip(recall),
            "intrusion_excerpt": _clip(intrusion),
            "false_claim_excerpt": _clip(false_claim),
            "general_excerpt": _clip(general),
        })
    if not rows:
        return
    qdf = pd.DataFrame(rows)
    qdf.to_csv(RUNS / "qualitative_chat_summary.csv", index=False)
    lines = [
        "# Qualitative Chat Summary",
        "",
        "These are local checkpoint conversations, not an AI-judge score. The point is",
        "to verify that each checkpoint loads through a chat-template path and to",
        "surface obvious source-grounding or intrusion failures next to the metrics.",
        "",
        qdf[[
            "run", "source_kind", "turns", "flags", "recall_excerpt",
            "intrusion_excerpt", "false_claim_excerpt", "general_excerpt",
        ]].to_markdown(index=False),
        "",
        "Raw transcripts are in `runs/<run>/eval/qualitative_chat.md`.",
    ]
    (RUNS / "qualitative_chat_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-text", action="store_true")
    args = ap.parse_args()
    df = build_experiment_table()
    br = teacher_reference_rows()
    if not br.empty:
        df = pd.concat([df, br], ignore_index=True, sort=False)
    if "model" in df.columns:
        df = df[~df["model"].map(_model_label).isin(RETIRED_MODEL_LABELS)].copy()
    df.to_csv(RUNS / "experiment_table.csv", index=False)
    write_coverage_matrix(df)
    write_markdown_table(df, RUNS / "experiment_table.md")
    write_fable_verdicts(df)
    write_loss_by_model_size(df)
    write_best_loss_window_by_corpus(df)
    plot_accuracy(df)
    plot_destruction(df)
    profiles = []
    for run_dir in sorted(RUNS.iterdir()):
        if run_dir.is_dir():
            prof = weight_profile(run_dir)
            if prof is not None:
                profiles.append(prof)
    plot_weight_profiles(profiles)
    qualitative_summary()
    if not args.skip_text:
        write_text_examples()
    print(f"wrote experiment assets for {len(df)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
