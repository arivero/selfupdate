#!/usr/bin/env python3
"""Append a compact overnight status report for the KD GPU lanes."""

from __future__ import annotations

import csv
import json
import pathlib
import statistics
import subprocess
import time


ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as e:
        return e.output.strip()


def latest_metrics(run: str) -> str:
    p = RUNS / run / "metrics.jsonl"
    if not p.exists():
        return f"- `{run}`: no metrics yet"
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    trains = [r for r in rows if r.get("kind") == "train"]
    evals = [r for r in rows if r.get("kind") == "eval"]
    done = [r for r in rows if r.get("kind") == "done"]
    parts = [f"- `{run}`"]
    if trains:
        vals = [r["loss"] for r in trains[-200:] if "loss" in r]
        parts.append(
            "train: epoch {epoch}, step {step}, last loss {loss:.4g}, mean200 {mean:.4g}".format(
                epoch=trains[-1].get("epoch"),
                step=trains[-1].get("step"),
                loss=float(trains[-1].get("loss", 0.0)),
                mean=statistics.mean(vals) if vals else float("nan"),
            )
        )
    if evals:
        e = evals[-1]
        parts.append(
            "probe: epoch {epoch}, CER {cer:.4g}, line {line:.4g}, prefix {prefix:.4g}, gen CE {ce}".format(
                epoch=e.get("epoch"),
                cer=float(e.get("cer", 0.0)),
                line=float(e.get("line_exact", 0.0)),
                prefix=float(e.get("prefix_lines", 0.0)),
                ce=("none" if e.get("gen_ce") is None else f"{float(e.get('gen_ce')):.4g}"),
            )
        )
    if done:
        d = done[-1]
        parts.append(
            "done: peak allocated {alloc} GiB, reserved {res} GiB, minutes {mins}".format(
                alloc=d.get("vram_gb"),
                res=d.get("reserved_vram_gb"),
                mins=d.get("minutes"),
            )
        )
    return "; ".join(parts)


def full_eval(run: str) -> str | None:
    p = RUNS / run / "eval" / "recite.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text(encoding="utf-8"))
    return (
        f"- `{run}` full eval: CER {r.get('cer')}, line {r.get('line_exact')}, "
        f"prefix {r.get('prefix_lines')}, general CE {r.get('general', {}).get('mean_ce')}"
    )


def teacher_rag_eval(run: str) -> str | None:
    p = RUNS / "teacher_rag" / f"{run}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text(encoding="utf-8"))
    return (
        f"- `{run}` teacher+RAG: CER {r.get('cer')}, line {r.get('line_exact')}, "
        f"prefix {r.get('prefix_lines')}, n {r.get('n')}"
    )


def layer_tops(run: str) -> str | None:
    p = RUNS / run / "eval" / "lora_layer_deltas_by_epoch.csv"
    if not p.exists():
        return None
    by_epoch: dict[int, list[tuple[float, int]]] = {}
    with p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                ep = int(float(row["epoch"]))
                layer = int(float(row["layer"]))
                val = float(row["adapter_update_rms"])
            except (KeyError, ValueError):
                continue
            by_epoch.setdefault(ep, []).append((val, layer))
    if not by_epoch:
        return None
    ep = max(by_epoch)
    tops = sorted(by_epoch[ep], reverse=True)[:8]
    top_s = ", ".join(f"L{layer}:{val:.3g}" for val, layer in tops)
    return f"- `{run}` layer tops at epoch {ep}: {top_s}"


def download_status() -> str:
    lock = RUNS / ".download_locks" / "qwen36_27b_snapshot.lock"
    pid = lock.read_text(encoding="utf-8").strip() if lock.exists() else ""
    ps = sh(["bash", "-lc", f"ps -p {pid} -o pid=,stat=,etime=,%cpu=,rss=,cmd= 2>/dev/null || true"]) if pid else ""
    size = sh(["bash", "-lc", "du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B 2>/dev/null | awk '{print $1}' || true"])
    tail = sh(["bash", "-lc", "tail -n 5 runs/downloads/qwen36_27b_snapshot.log 2>/dev/null | tr '\\r' '\\n' | tail -n 3"])
    if ps:
        return f"- Qwen3.6 download: running `{ps}`; cache {size or 'unknown'}; log tail `{tail}`"
    return f"- Qwen3.6 download: not running; cache {size or 'missing'}; log tail `{tail}`"


def main() -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"\n## {now}\n")
    print("**GPU lanes**")
    print("```")
    print(sh(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits", "-i", "2,3"]))
    print("```")
    print("\n**Active processes**")
    print("```")
    print(sh(["bash", "-lc", "ps -eo pid,ppid,stat,etime,%cpu,cmd | rg 'train.py|evaluate.py|gpu_memory_sampler|snapshot_download|Qwen3.6|qwen36|qwen3_30ba3b|e60_v3_14b|gpu_scheduler' || true"]))
    print("```")
    print("\n**Recall / Forgetting**")
    for run in [
        "kd_lora_ce_hi_e40_v3_qwen3_30ba3b_inst2507_rag",
        "kd_lora_ce_hi_e60_v3_14b_rag",
        "kd_lora_ce_hi_e40_v3_qwen36_27b_rag",
    ]:
        print(latest_metrics(run))
    for run in [
        "kd_lora_ce_hi_qwen3_30ba3b_inst2507_rag",
        "kd_lora_ce_hi_e40_v3_qwen3_30ba3b_inst2507_rag",
        "kd_lora_ce_hi_e60_v3_14b_rag",
        "kd_lora_ce_hi_e40_v3_qwen36_27b_rag",
    ]:
        fe = full_eval(run)
        if fe:
            print(fe)
    for run in [
        "kd_lora_ce_hi_e40_v3_qwen3_30ba3b_inst2507_rag",
        "kd_lora_ce_hi_e60_v3_14b_rag",
        "kd_lora_ce_hi_e40_v3_qwen36_27b_rag",
    ]:
        tr = teacher_rag_eval(run)
        if tr:
            print(tr)
    print("\n**Layer Modification**")
    for run in [
        "kd_lora_ce_hi_e40_v3_qwen3_30ba3b_inst2507_rag",
        "kd_lora_ce_hi_e60_v3_14b_rag",
        "kd_lora_ce_hi_e40_v3_qwen36_27b_rag",
    ]:
        lt = layer_tops(run)
        if lt:
            print(lt)
    print("\n**Downloads**")
    print(download_status())
    print("\n**Memory Summary**")
    mem = RUNS / "gpu_memory_summary.md"
    print(mem.read_text(encoding="utf-8").strip() if mem.exists() else "- no sampler summary yet")


if __name__ == "__main__":
    main()
