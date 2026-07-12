"""Build the didactic implementation note ``runs/v5_training.pdf``.

This script is deliberately local and read-only: it reads the active v5
configs/cache indexes and renders a source-backed PDF.  It does not launch or
alter training.
"""

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from transformers import AutoConfig


ROOT = Path(__file__).resolve().parent.parent
V = 151_936
ACTIVE = [
    ("0.6B", "Qwen/Qwen3-0.6B", "caches/Qwen3-0.6B-rag_system-remove-1a1d43a35c20770e",
     "configs/experiments/v5rs/base_0p6b.yaml", 8, 8, False),
    ("1.7B", "Qwen/Qwen3-1.7B", "caches/Qwen3-1.7B-rag_system-remove-5aa1e7a3eddaf5e9",
     "configs/experiments/v5rs/base_1p7b.yaml", 4, 16, True),
]


def mib(n):
    return n / 2**20


def gib(n):
    return n / 2**30


def stats(values):
    values = sorted(values)
    n = len(values)
    at = lambda q: values[min(n - 1, int(q * (n - 1)))]
    return {"n": n, "min": values[0], "p50": at(.50), "p90": at(.90),
            "p99": at(.99), "max": values[-1], "sum": sum(values)}


def read_model(label, name, cache, cfg, micro, accum, offload):
    c = AutoConfig.from_pretrained(name)
    index = json.loads((ROOT / cache / "index.json").read_text())
    spans = list(index["examples"].values())
    aligned = [x["A"] for x in spans]
    # A covers mid + generated answer.  The readout begins after mid.
    readout = [x["A"] - x["mid_len"] for x in spans]
    return {
        "label": label, "name": name, "cache": cache, "config": cfg,
        "layers": int(c.num_hidden_layers), "hidden": int(c.hidden_size),
        "vocab": int(c.vocab_size), "micro": micro, "accum": accum,
        "offload": offload, "aligned": stats(aligned), "readout": stats(readout),
        "cache_bytes_token": int(c.num_hidden_layers) * int(c.hidden_size) * 2,
        "cache_bytes_total": sum(aligned) * int(c.num_hidden_layers) * int(c.hidden_size) * 2,
    }


def fig(title, subtitle=""):
    f = plt.figure(figsize=(8.27, 11.69))
    f.text(.07, .955, title, fontsize=19, weight="bold", va="top")
    if subtitle:
        f.text(.07, .925, subtitle, fontsize=8.5, color="#555555", va="top")
    return f


def prose(f, y, heading, body, width=102):
    f.text(.07, y, heading, fontsize=11, weight="bold", va="top")
    y -= .024
    wrapped = "\n".join(textwrap.wrap(body, width=width))
    f.text(.07, y, wrapped, fontsize=9.2, va="top", linespacing=1.35)
    return y - .025 * (wrapped.count("\n") + 1) - .026


def codebox(f, y, ref, code, fontsize=7.2):
    lines = code.strip("\n").splitlines()
    h = .027 * len(lines) + .046
    box = FancyBboxPatch((.065, y - h), .87, h, boxstyle="round,pad=0.008",
                         facecolor="#eeeeee", edgecolor="#b8b8b8", linewidth=.7,
                         transform=f.transFigure)
    f.patches.append(box)
    f.text(.082, y - .012, ref, fontsize=7.2, color="#555555", va="top",
           family="monospace", weight="bold")
    f.text(.082, y - .031, "\n".join(lines), fontsize=fontsize, va="top",
           family="monospace", linespacing=1.2)
    return y - h - .023


def save(pdf, f):
    pdf.savefig(f, dpi=180)
    plt.close(f)


def flow_page(pdf):
    f = fig("The v5 training", "A source-backed map of cache construction and layerwise training")
    f.text(.07, .87, "What this note establishes", fontsize=12, weight="bold")
    f.text(.07, .84, "The v5 pipeline separates teacher computation (once, into a disk cache) from student training\n"
           "(many local backward steps).  The cache is cheap in GPU terms because it remains on disk/host until a batch\n"
           "needs it.  The current 1.7B OOM is instead consistent with transient full-vocabulary divergences.",
           fontsize=9.4, va="top", linespacing=1.4)
    ax = f.add_axes([.08, .48, .84, .25]); ax.axis("off")
    labels = [("v5 records", "empty answer\n+ RAG context"),
              ("build_teacher_cache", "generate, teacher\nforward, slice h¹…hᴸ"),
              ("safetensors cache", "fp16 [A,H] per layer\n+ JSON span index"),
              ("train.py", "local block step\n+ backward")]
    xs = [.02, .28, .54, .80]
    for x, (a, b) in zip(xs, labels):
        ax.add_patch(FancyBboxPatch((x, .35), .17, .35, boxstyle="round,pad=.02",
                                    facecolor="#e8f0f7", edgecolor="#557a95"))
        ax.text(x+.085, .57, a, ha="center", va="center", fontsize=9, weight="bold")
        ax.text(x+.085, .42, b, ha="center", va="center", fontsize=7.4)
    for x in [.20, .46, .72]:
        ax.add_patch(FancyArrowPatch((x, .525), (x+.07, .525), arrowstyle="->",
                                     mutation_scale=15, color="#557a95"))
    y = .42
    y = prose(f, y, "Scope", "This is a code-reading document, not a claim that the present loss implementation is optimal. It reports what the active v5 configuration and source do today, then identifies which tensor shapes make the current full-vocabulary objective memory-sensitive.")
    y = codebox(f, y, "scripts/train.py:20–36", "cfg = load_config(args.config, args.experiment)\n...\nrun_dir = train_layerwise(cfg)")
    y = prose(f, y, "Current active setting", "The queued 1.7B jacobian-lens-KL runs use summed scheduling, connected-window width 1, bucketed batches of four, 16-example gradient accumulation, a frozen teacher copy, and offloaded Adam moments. The 0.6B base uses batches of eight and eight-example accumulation.")
    save(pdf, f)


def cache_page(pdf, models):
    f = fig("1. Building the frozen-teacher cache", "The teacher is run once per record; its aligned hidden states become a model-specific artifact.")
    y = .89
    y = prose(f, y, "Open-answer v5 records", "The dataset has empty answer fields. For each record, the teacher first greedily generates an answer with the RAG context present. Those generated token ids are cache content, rather than reference-text training targets. The answer is then included in the teacher-forced forward whose hidden states are saved.")
    y = codebox(f, y, "scripts/build_teacher_cache.py:164–174", "if v5:\n    answer_ids, hard_cut = generate_answer(model, masker, ex, stop_id, budget)\n    pair = masker.build(ex, answer_ids=answer_ids)\n    extra = {\"answer_ids\": answer_ids, \"hard_cut\": hard_cut}")
    y = prose(f, y, "Teacher forward and span restriction", "The builder requests all transformer hidden states, but writes only the aligned span. For an open-answer record that span is the shared middle plus the teacher-generated answer, not the privileged RAG evidence itself.")
    y = codebox(f, y, "scripts/build_teacher_cache.py:186–203", "out = model(t_ids, output_hidden_states=True, use_cache=False)\nspan = pair.t_aligned\nhidden = {\n    L: out.hidden_states[L][0, span.start:span.stop]\n    for L in range(1, n_layers + 1)\n}\nwriter.add(ex.example_id, hidden, span={..., \"A\": pair.aligned_len, ...})")
    y = prose(f, y, "Storage contract", "Each layer is detached, converted to the configured storage dtype (the active caches are float16), made contiguous, moved to CPU, and written in safetensors shards. The index carries alignment and generated-answer metadata. Training subsequently opens tensors lazily by example and layer.")
    y = codebox(f, y, "src/selfupdate/teacher/cache.py:111–127, 158–167", "stored = h.detach().to(self.hidden_dtype).contiguous().cpu()\nself._buffer[f\"{example_id}/h{L:02d}\"] = stored\n...\nsave_file(self._buffer, str(self.root / f\"shard-{self._shard_no:05d}.safetensors\"))\n...\nreturn self._handle(example_id).get_tensor(f\"{example_id}/h{layer:02d}\")")
    save(pdf, f)


def numbers_page(pdf, models):
    f = fig("2. What the cache costs", "Exact sizes from the active cache indexes and model configurations; fp16 hidden states only.")
    ax = f.add_axes([.07, .58, .86, .25]); ax.axis("off")
    cols = ["model", "layers × hidden", "fp16 cache / aligned token", "examples", "aligned tokens", "cache payload"]
    rows = []
    for m in models:
        rows.append([m["label"], f'{m["layers"]} × {m["hidden"]:,}',
                     f'{mib(m["cache_bytes_token"]):.1f} MiB?'.replace('.0 MiB?', ' KiB').replace(' MiB?', ' KiB'),
                     str(m["aligned"]["n"]), f'{m["aligned"]["sum"]:,}', f'{gib(m["cache_bytes_total"]):.2f} GiB'])
    # The display trick above is not appropriate for 56/112 KiB; construct explicitly.
    for row, m in zip(rows, models):
        row[2] = f'{m["cache_bytes_token"] / 1024:.0f} KiB'
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if r == 0: cell.set_facecolor("#dbe8f2"); cell.set_text_props(weight="bold")
        elif r % 2: cell.set_facecolor("#f5f7f8")
    y = .52
    y = prose(f, y, "Formula", "For L layers, hidden width H, aligned length A, and fp16 storage, the hidden-state payload is L × H × A × 2 bytes. This is disk/host storage; it is not a simultaneous GPU allocation during normal cached training.")
    y = codebox(f, y, "src/selfupdate/teacher/cache.py:8–18", "Per example, restricted to the aligned span (length A):\n- ``h{L}``   [A, H] float16")
    y = prose(f, y, "Active 1.7B cache", f'The active 1.7B remove-compaction cache contains {models[1]["aligned"]["sum"]:,} aligned tokens. At 112 KiB per aligned token, its raw hidden payload is {gib(models[1]["cache_bytes_total"]):.2f} GiB before safetensors/index overhead. The 0.6B cache is 56 KiB/token and {gib(models[0]["cache_bytes_total"]):.2f} GiB raw.')
    save(pdf, f)


def spans_page(pdf, models):
    f = fig("3. Current span distribution", "Long spans drive transient vocabulary-loss memory; the values below are read from each cache index.")
    ax = f.add_axes([.07, .58, .86, .23]); ax.axis("off")
    cols = ["model", "span", "min", "p50", "p90", "p99", "max"]
    rows = []
    for m in models:
        for label, d in [("aligned (hidden loss)", m["aligned"]), ("answer (readout)", m["readout"])]:
            rows.append([m["label"], label, *[str(d[k]) for k in ("min", "p50", "p90", "p99", "max")]])
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.3); tbl.scale(1, 1.7)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if r == 0: cell.set_facecolor("#dbe8f2"); cell.set_text_props(weight="bold")
        elif r % 2: cell.set_facecolor("#f5f7f8")
    y = .50
    y = prose(f, y, "Why there are two lengths", "The hidden loss is evaluated on the full aligned span A: shared middle plus answer. The readout teacher-KL starts at the answer and therefore has A − mid_len rows. Both are variable-length; bucketed batching reduces padding but deliberately retains a real batched forward/backward.")
    y = codebox(f, y, "src/selfupdate/data/dataset.py:215–250", "Amax = max(it.A for it in items)\nans_lens = [it.s0 + it.A - it.ans0 for it in items]\nRmax = max(ans_lens) if ans_lens else 0\n...\nreadout_index[i, :rlen] = torch.arange(it.ans0 - 1, it.s0 + it.A - 1)")
    y = prose(f, y, "Important interpretation", "A long example contributes many tokenwise terms to the same example-average KL. It does not create one optimizer update per token. It does, however, cause a larger vocabulary-shaped intermediate when all positions are decoded together.")
    save(pdf, f)


def schedule_page(pdf):
    f = fig("4. What train.py does for the active slide-1 runs", "The current schedule is local in depth; it does not retain graphs for all 28 layers until epoch end.")
    ax = f.add_axes([.08, .64, .84, .15]); ax.axis("off")
    for i in range(5):
        x = .02 + i*.19
        label = "embed" if i == 0 else ("block 28 +\nreadout" if i == 4 else f"block {i}")
        ax.add_patch(FancyBboxPatch((x, .30), .13, .35, boxstyle="round,pad=.015", facecolor="#e8f0f7", edgecolor="#557a95"))
        ax.text(x+.065, .475, label, ha="center", va="center", fontsize=8, weight="bold")
        if i:
            ax.annotate("", (x, .475), (x-.055, .475), arrowprops=dict(arrowstyle="->", color="#557a95"))
    ax.text(.42, .05, "detach before each local step → backward → retain only detached output", ha="center", fontsize=8.5, color="#355b73")
    y = .58
    y = prose(f, y, "Slide width one", "With conn_window = 1, the summed loop calls one local block step per layer. The local helper runs that block, forms the layer loss, calls backward, and returns detached tensors. The final one-block window combines final hidden loss with the sanctioned teacher-KL readout and then calls backward once.")
    y = codebox(f, y, "src/selfupdate/train/layerwise.py:454–466", "loss_vals, h = local_block_step_batch(\n    stack, L, h.detach(), pos_emb, target,\n    batch, loss_fn, previous_target=targets.get(L - 1),\n)\nlayer_losses.append(loss_vals)\nL += 1")
    y = codebox(f, y, "src/selfupdate/train/steps.py:168–175", "h_out = stack.run_block(L, h_in, pos_emb)\nlosses = _layer_loss_per_example(...)\ntotal = losses.sum()\n...\ntotal.backward()\nreturn losses.detach(), h_out.detach()")
    y = prose(f, y, "What is accumulated", "Parameter gradients accumulate across examples until the configured gradient-accumulation boundary. The Python loss values kept for logging are detached; they do not keep layer graphs alive. A wider connected window intentionally retains a graph inside that window, but the active arms use width one.")
    save(pdf, f)


def vocab_page(pdf, models):
    m = models[1]
    f = fig("5. The vocabulary-shaped local-loss peak", "This is an implementation property of lens-KL objectives, not the disk-cache footprint.")
    y = .89
    y = prose(f, y, "The operation", "jacobian_lens_kl transports a hidden state through a frozen Jacobian and then uses the lens-KL fallback. The lens decodes student and teacher hidden rows through the frozen LM head, converts both logit tensors to fp32 for log-softmax, and evaluates KL over all 151,936 vocabulary entries.")
    y = codebox(f, y, "src/selfupdate/train/losses.py:390–410", "s_logits = self.lm_head(student_h)\nwith torch.no_grad():\n    t_logits = self.lm_head(teacher_h)\nreturn F.kl_div(\n    F.log_softmax(s_logits.float(), dim=-1),\n    F.log_softmax(t_logits.float(), dim=-1),\n    log_target=True, reduction=\"batchmean\",\n)")
    y = prose(f, y, "Per-example loop", "The padded-batch helper slices each example to its real prefix, but appends all per-example loss graphs before summing and calling backward. Therefore the full-vocabulary intermediates for up to the micro-batch size coexist during one local layer. They are released after that local backward, not retained across all depth layers.")
    y = codebox(f, y, "src/selfupdate/train/steps.py:75–89, 152–175", "for i, k in enumerate(lens):\n    losses.append(loss_fn(student_h[i, :k], teacher_h[i, :k], ...))\nreturn torch.stack(losses)\n...\ntotal = losses.sum()\ntotal.backward()")
    ax = f.add_axes([.09, .22, .82, .16]); ax.axis("off")
    for x, label, color in [(0.02, "hidden rows\n[A, H]", "#e8f0f7"), (.27, "LM head\nH → V", "#dbe8f2"), (.52, "fp32 log-softmax\n[A, 151,936]", "#f7e7d3"), (.77, "KL +\nbackward", "#f4dddd")]:
        ax.add_patch(FancyBboxPatch((x,.28), .16,.40, boxstyle="round,pad=.02", facecolor=color, edgecolor="#7a7a7a"))
        ax.text(x+.08,.48,label,ha="center",va="center",fontsize=8,weight="bold")
    for x in [.19,.44,.69]: ax.annotate("",(x+.06,.48),(x,.48),arrowprops=dict(arrowstyle="->"))
    save(pdf, f)


def peak_page(pdf, models):
    m = models[1]
    f = fig("6. Concrete 1.7B peak arithmetic", "Lower-bound tensor sizes; kernels and autograd can require additional workspace/saved tensors.")
    spans = [m["aligned"]["p50"], m["aligned"]["p99"], m["aligned"]["max"]]
    ax = f.add_axes([.07, .61, .86, .22]); ax.axis("off")
    rows = []
    for a in spans:
        fp32 = a * m["vocab"] * 4
        bf16 = a * m["vocab"] * 2
        rows.append([str(a), f"{mib(bf16):.0f} MiB", f"{mib(fp32):.0f} MiB", f"{mib(4*fp32):.0f} MiB"])
    tbl = ax.table(cellText=rows, colLabels=["aligned rows A", "one bf16 logit table", "one fp32 vocab table", "four fp32 tables"], cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.6); tbl.scale(1, 1.9)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if r == 0: cell.set_facecolor("#dbe8f2"); cell.set_text_props(weight="bold")
        elif r % 2: cell.set_facecolor("#f5f7f8")
    y = .53
    y = prose(f, y, "How to read the table", "One fp32 vocabulary table is A × 151,936 × 4 bytes. At the maximum 962-row aligned span it is 557 MiB. The actual KL path has more than one such conceptual table: student/teacher logits, float conversions, log-softmax outputs, KL work, autograd state, block activations, model weights, gradients, and the frozen teacher copy. The table is therefore not a complete VRAM estimate; it explains why the shape is hazardous.")
    y = prose(f, y, "Observed failure versus diagnosis", "The scheduler log reported an OOM while allocating 252 MiB in `log_softmax`/`kl_div`, after epoch-zero evaluation and before the first epoch completed. 252 MiB corresponds to about 435 rows of one fp32 [rows, vocab] table. This rules out a multi-epoch accumulation as the immediate trigger; it does not by itself prove that every transient is optimally released.")
    y = codebox(f, y, "src/selfupdate/train/steps.py:137–149", "for i, k in enumerate(lens):\n    losses.append(F.kl_div(\n        F.log_softmax(student_logits[i, :k].float(), dim=-1),\n        F.log_softmax(teacher_logits[i, :k].float(), dim=-1),\n        log_target=True, reduction=\"batchmean\",\n    ))")
    save(pdf, f)


def configuration_page(pdf, models):
    f = fig("7. Active v5 configurations and consequences", "Configuration values were read from the current base YAML files.")
    ax = f.add_axes([.07, .62, .86, .20]); ax.axis("off")
    rows = []
    for m in models:
        rows.append([m["label"], str(m["micro"]), str(m["accum"]), str(m["micro"] * m["accum"]), "yes" if m["offload"] else "no", "1"])
    tbl = ax.table(cellText=rows, colLabels=["model", "micro-batch", "grad accum", "examples / optimizer step", "Adam offload", "window"], cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.6); tbl.scale(1, 1.9)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if r == 0: cell.set_facecolor("#dbe8f2"); cell.set_text_props(weight="bold")
        elif r % 2: cell.set_facecolor("#f5f7f8")
    y = .54
    y = prose(f, y, "Gradient scale", "A tokenwise KL is reduced by `batchmean` inside each example, and example losses are summed before backward. Thus span length changes the number of token terms and memory/compute cost, but not the intended per-example loss scale. The batch path combines several examples in one backward; it does not make an optimizer update per token.")
    y = codebox(f, y, "configs/experiments/v5rs/base_1p7b.yaml:14–32", "micro_batch: 4\ngrad_accum: 16\nbatching: bucketed\nhidden_loss: huber        # arm overlay selects jacobian_lens_kl\nconn_window: 1\nreadout_window_blocks: 1\nreadout_source: teacher_kl\nfrozen_teacher_copy: true\noffload_adam: true")
    y = prose(f, y, "What the cache does not decide", "The cache fixes teacher targets and saves teacher-forward compute. It does not force whole-span vectorization, full-vocabulary KL, a particular position chunk size, or a particular gradient estimator. Those are training-loss implementation choices.")
    save(pdf, f)


def conclusion_page(pdf):
    f = fig("8. Findings and safe next measurements", "A concise separation of confirmed facts, inference, and changes that would need numerical re-checking.")
    y = .89
    y = prose(f, y, "Confirmed from code and current artifacts", "(1) v5 caches hold fp16 teacher hidden states per aligned token and layer on disk. (2) the active slide-1 trainer backpropagates locally per layer; it does not retain all 28 layer graphs. (3) jacobian_lens_kl and teacher-KL decode full vocabulary distributions in fp32. (4) the per-example helpers construct all rows in a micro-batch before the local backward. (5) the failed 1.7B attempts did not complete epoch one.")
    y = prose(f, y, "Strong implementation inference", "The dominant OOM pressure is a local full-vocabulary peak for long spans, on top of resident student, frozen teacher, gradients, and optimizer-paging state. The 252 MiB failed request matches a single roughly 435-row fp32 vocabulary table. Allocator fragmentation was small in the error report, so it is not the primary explanation.")
    y = prose(f, y, "Do not silently change the science", "Position chunking with correctly weighted partial losses aims to preserve the full-span objective, but it changes operation order and should be measured with `scripts/train_certify.py`. Position sampling or top-k/approximate KL changes the estimator/objective and needs an explicit experimental arm, not a memory-only patch.")
    y = codebox(f, y, "AGENTS.md — Training Runtime & Certification", "python scripts/train_certify.py --all --out-dir /tmp/$USER/certify_head\n# apply intended numerics-preserving change\npython scripts/train_certify.py --all --reference-dir /tmp/$USER/certify_head")
    y = prose(f, y, "Suggested instrumentation before a code change", "Record peak allocated/reserved memory at the start and end of each local block step, with aligned/readout lengths and layer number. This distinguishes a long-span peak from a true retained-allocation path without guessing from a single OOM trace.")
    save(pdf, f)


def recovered_pre_v5_page(pdf):
    f = fig("Appendix A. Recovered pre-v5 comparison", "Temporary detached worktree: /tmp/selfupdate_lw_pre_v5 at 3fb5305 (2026-07-12 00:17:52 +0200).")
    y = .89
    y = prose(f, y, "A necessary correction", "The recovered snapshot is not literally cache-free: it already contains a frozen hidden-state TeacherCacheWriter and writes h01…hL safetensors. The material v5 change is that question-only records have no answer in the dataset, so the teacher must generate an answer before the cache forward; answer ids and hard-cut status are then stored in the cache index.")
    y = codebox(f, y, "/tmp/selfupdate_lw_pre_v5/scripts/build_teacher_cache.py:65–84", "for ex in tqdm(examples, desc=\"teacher forward\"):\n    pair = masker.build(ex)\n    t_ids = torch.tensor([pair.teacher_ids], device=model.device)\n    with torch.no_grad():\n        out = model(t_ids, output_hidden_states=True, use_cache=False)\n    hidden = {L: out.hidden_states[L][0, span.start:span.stop]\n              for L in range(1, n_layers + 1)}\n    writer.add(ex.example_id, hidden, span={..., \"A\": pair.aligned_len, ...})")
    y = prose(f, y, "Current v5 difference", "The current builder detects the all-empty-answer dataset, greedily generates an answer under the RAG-bearing teacher prompt, rebuilds the pair with those ids, and persists them as `extra`. This is what makes a cache meaningful for the v5 question-only dataset: it supplies the teacher-generated answer sequence as well as teacher hidden targets.")
    y = codebox(f, y, "scripts/build_teacher_cache.py:164–203", "if v5:\n    answer_ids, hard_cut = generate_answer(model, masker, ex, stop_id, budget)\n    pair = masker.build(ex, answer_ids=answer_ids)\n    extra = {\"answer_ids\": answer_ids, \"hard_cut\": hard_cut}\n...\nwriter.add(ex.example_id, hidden, span={..., \"A\": pair.aligned_len, ...}, extra=extra)")
    y = prose(f, y, "Why this distinction matters", "Comparisons should not describe v5 as simply adding a hidden cache. The older implementation already amortized teacher hidden-state forwards. V5 changes target provenance and sequence construction: the teacher's generated, RAG-conditioned answer is a cache artifact, while the student receives a compacted version of that sequence.")
    save(pdf, f)


def recovered_delta_page(pdf):
    f = fig("Appendix B. Before/after: what actually changed", "Source comparison between recovered 3fb5305 and the current tree.")
    ax = f.add_axes([.06, .60, .88, .23]); ax.axis("off")
    rows = [
        ["Teacher hidden-state cache", "present: hL safetensors", "present: same core mechanism"],
        ["Answer source", "record supplies answer to masker", "teacher greedy generation supplies answer ids"],
        ["Answer metadata in cache index", "absent", "answer_ids + hard_cut stored as extra"],
        ["Question-only v5 records", "not handled by cache builder", "explicitly detected and supported"],
        ["Memory implication", "cache avoids teacher hidden forwards", "same, but generated spans can be longer/variable"],
    ]
    tbl = ax.table(cellText=rows, colLabels=["aspect", "recovered pre-generation snapshot", "current v5"], cellLoc="left", loc="center", colWidths=[.28,.31,.31])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.9)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if r == 0: cell.set_facecolor("#dbe8f2"); cell.set_text_props(weight="bold")
        elif r % 2: cell.set_facecolor("#f5f7f8")
    y = .53
    y = prose(f, y, "What did not change", "Both versions slice all model layers at the aligned span and write detached fp16 tensors to CPU safetensors. Therefore the basic L × H × A cache-storage arithmetic is not a novel v5 cost. The change to v5 primarily makes A depend on the teacher's generated answer, which creates the variable-span distribution shown earlier in this document.")
    y = codebox(f, y, "/tmp/selfupdate_lw_pre_v5/src/selfupdate/teacher/cache.py:99–128", "for L, h in hidden.items():\n    stored = h.detach().to(self.hidden_dtype).contiguous().cpu()\n    self._buffer[f\"{example_id}/h{L:02d}\"] = stored\n...\nsave_file(self._buffer, str(self.root / f\"shard-{self._shard_no:05d}.safetensors\"))")
    y = prose(f, y, "What this cannot establish", "The recovered snapshot is a source baseline, not a matched numerical baseline for the active v5 campaign. It should be used to understand provenance and control flow, not to infer a fair performance or memory delta without re-running matched configurations.")
    save(pdf, f)


def criticism_page(pdf):
    f = fig("Appendix C. Criticism of the current v5 method", "The cache is scientifically useful, but it is not free—and the costs should be stated plainly.")
    y = .89
    y = prose(f, y, "1. A new pre-generation stage", "Before v5, the cache builder teacher-forced the answer already present in a record. In v5 question-only data, it must first autoregressively generate an answer for every record, then run a second teacher-forced forward to obtain all hidden states. In the current queue this added roughly an hour-scale GPU preprocessing phase per model/compaction cache, before a student can begin training. It is an operational cost and a new failure surface: hard cuts, poor RAG attention, and cache identity all need validation.")
    y = codebox(f, y, "scripts/build_teacher_cache.py:164–188", "if v5:\n    budget = _generation_budget(...)\n    answer_ids, hard_cut = generate_answer(model, masker, ex, stop_id, budget)\n    pair = masker.build(ex, answer_ids=answer_ids)\n...\nout = model(t_ids, output_hidden_states=True, use_cache=False)")
    y = prose(f, y, "2. It trades teacher compute for large stored targets", "Caching avoids rerunning the teacher's full hidden-state trajectory during student training. But it stores L × H × A fp16 values and encourages full-span objectives. For 1.7B, each aligned token costs 112 KiB on disk; the active cache is several GiB. The longest generated spans also feed directly into the current full-vocabulary local losses, increasing transient VRAM risk.")
    y = codebox(f, y, "src/selfupdate/teacher/cache.py:111–119; src/selfupdate/train/losses.py:390–410", "stored = h.detach().to(self.hidden_dtype).contiguous().cpu()\n...\ns_logits = self.lm_head(student_h)\nt_logits = self.lm_head(teacher_h)\nF.log_softmax(..., dim=-1)  # full vocabulary")
    y = prose(f, y, "3. Cache generation fixes one teacher trajectory", "The student is trained on the teacher-generated answer that was fixed at cache-build time. This is stable and replayable, but it cannot ask what the teacher would do after a student deviation. The method therefore optimizes imitation on a fixed teacher path, not an on-policy student-generated conversation.")
    y = codebox(f, y, "src/selfupdate/data/dataset.py:104–121", "if not ex.answer:\n    answer_ids = cache.answer_ids(ex.example_id)\n    if answer_ids is None:\n        raise ValueError(... \"rebuild the cache\")\npair = masker.build(ex, answer_ids=answer_ids)")
    y = prose(f, y, "Bottom line", "The v5 cache is a deliberate compute/replay trade-off, not a free acceleration. It should be compared against online alternatives on wall time, peak VRAM, teacher-call cost, trajectory bias, and resulting recall/damage—not only against the cost of a single student epoch.")
    save(pdf, f)


def review_correction_page(pdf):
    f = fig("Appendix F. 2026-07-12 execution review correction", "The first v5 results are legacy execution, not the intended disk-cache method.")
    y = .89
    y = prose(f, y, "Resident-teacher defect", "frozen_teacher_copy was enabled only because anchor-KL needs one-time frozen-base logits. Teacher presence was also used as the online-target switch, so every batch retained a second model and recomputed teacher targets instead of using disk targets. That directly explains the abnormal VRAM pressure and much of the epoch slowdown.")
    y = codebox(f, y, "Old control-flow error — layerwise.py", "anchor = _make_anchor(cfg, tok, teacher)\nonline = teacher is not None\ntargets = teacher.aligned_targets_batch(batch, device) if online else batch.hidden")
    y = prose(f, y, "Repair", "The target-source switch is now cfg.train.online_teacher. Once the anchor bank is materialized, an offline summed run drops its frozen copy and releases CUDA cache. Cached targets are then fp16 disk tensors; online LoRA and teacher-stream schedules remain resident-teacher designs by explicit configuration.")
    y = codebox(f, y, "Current repair — layerwise.py; runtime.py", "online = cfg.train.online_teacher\nif not online and release_teacher is not None:\n    teacher = None\n    release_teacher()\ntargets = teacher.aligned_targets_batch(...) if online else batch.hidden")
    y = prose(f, y, "Cache/gate contract", "Generated answer ids are payload, not a decoding preference. Cache identity now includes generation_extra_tokens. A post-build cache_generation_gate reads the exact generation_report.json and blocks an arm when its own cached answers have too many hard cuts; the earlier RAG gate separately establishes retrieval use.")
    y = codebox(f, y, "cache.py; cache_generation_gate.py", '"generation_extra_tokens": int(cfg.cache.generation_extra_tokens)\nfraction = summary["hard_cut_fraction"]\npass_ = fraction <= max_hard_cut_fraction')
    y = prose(f, y, "Interpretation", "Workers started before the repair may finish as diagnostics, but cannot be pooled with repaired cached runs: their target source, dtype, compute, peak memory and numerical trajectory differ.")
    save(pdf, f)


def online_design_page(pdf):
    f = fig("Appendix D. Proposed online, token-at-a-time alternative", "Design sketch only. It would need a dedicated experiment/config and certification before use.")
    y = .89
    y = prose(f, y, "Objective", "Avoid pre-generating and storing the whole teacher trajectory. At token t, obtain a teacher distribution from the RAG-bearing teacher context and a student distribution from the compact context, compute a teacher-sourced KL for that one next-token row, and immediately backpropagate the local layer loss. This changes the memory shape from [answer length, vocabulary] to [1, vocabulary].")
    y = codebox(f, y, "Proposed pseudocode — not present in the repository", "prefix = initial_prompt\nfor t in range(max_new_tokens):\n    p_teacher, teacher_kv = teacher.next_distribution(rag_prompt, prefix, teacher_kv)\n    p_student, student_kv = student.next_distribution(compact_prompt, prefix, student_kv)\n    local_teacher_kl(p_student, p_teacher).backward()\n    token = sample_or_greedy(p_student.detach())\n    prefix.append(token)")
    y = prose(f, y, "Teacher source and alignment", "For a genuinely on-policy variant, append the student-selected token to both histories. The teacher is then queried on the student's actual generated prefix, still with its RAG evidence; this keeps the behavioral target teacher-sourced even after a deviation. Gradients do not pass through the discrete token selection. A teacher cache cannot answer these dynamic-prefix queries, so the teacher must be resident, paged, or recomputed online.")
    y = codebox(f, y, "Current online-teacher interface (analogy): src/selfupdate/train/teacher_source.py:75–81", "t_ids = batch.teacher_ids.to(device)\nt_pos = torch.arange(t_ids.shape[1], device=device)[None].expand(\n    t_ids.shape[0], -1\n)")
    y = prose(f, y, "The crucial KV-cache constraint", "If `optimizer.step()` occurs after every token, it changes every trainable student block. All student KV entries created under the previous weights are then stale. An exact next-token forward must re-encode the compact prefix after every update; reusing stale KVs would train a different, internally inconsistent model. This is the main compute cost that replaces the current cache-build cost.")
    y = codebox(f, y, "Proposed exact update rule — not present in the repository", "loss_t.backward()\noptimizer.step(); optimizer.zero_grad(set_to_none=True)\n# weights changed: do NOT reuse student_kv\nstudent_kv = student.encode_prefix(compact_prompt + prefix)\n# teacher_kv remains valid only because teacher is frozen")
    y = prose(f, y, "Practical compromise", "Accumulate token losses/backward calls for a short chunk while retaining a valid student KV cache, then update once and replay the prefix. This preserves the token-sized vocabulary peak but makes the update cadence explicit. It is not numerically equivalent to current fixed-trajectory layerwise training and must be introduced as a distinct online experiment.")
    save(pdf, f)


def kv_directions_page(pdf):
    f = fig("Appendix E. KV reuse and teacher-vector directions", "Status of three future ideas: distinguish existing code from proposed semantics.")
    ax = f.add_axes([.06, .61, .88, .21]); ax.axis("off")
    rows = [
        ["(a) teacher KV minus censored rows", "not implemented", "Need a defined treatment for later KV rows that already absorbed censored context."],
        ["(b) stale student KV after update", "not implemented", "Avoids replay, but forward uses keys/values made by older weights."],
        ["(c) teacher hidden-vector input", "implemented: teacher_censored", "Teacher runs full context; student block sees non-privileged teacher hidden rows."],
    ]
    tbl = ax.table(cellText=rows, colLabels=["direction", "status", "meaning"], cellLoc="left", loc="center", colWidths=[.31,.22,.39])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 2.05)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if r == 0: cell.set_facecolor("#dbe8f2"); cell.set_text_props(weight="bold")
        elif r % 2: cell.set_facecolor("#f5f7f8")
    y = .52
    y = prose(f, y, "Existing implementation: teacher_censored", "The online frozen teacher runs a full sequence and returns raw outputs h0…hn. For each trained block L, the schedule forms the student-visible row index by deleting privileged rows, takes the teacher h[L−1] vectors at those rows as the block input, and targets teacher h[L]. This retains teacher-context information inside the vectors while preventing the student block from directly attending to the deleted token rows.")
    y = codebox(f, y, "src/selfupdate/train/teacher_source.py:46–58; src/selfupdate/train/layerwise.py:247–253, 289–331", "states = [h]\nfor L in range(1, self.stack.n_layers + 1):\n    h = self.stack.run_block(L, h, pos_emb)\n    states.append(h)\n...\nrows = censored_rows(it.s0, tA0, it.A, it.t_priv, device)\n# block L consumes censored rows of teacher h[L-1]")
    y = prose(f, y, "Why this is not teacher-KV reuse", "No training path passes `past_key_values` into the model. The only DynamicCache usage is evaluation-time greedy decoding. A cached teacher key/value from a later row can already contain information mixed from privileged tokens, so simply deleting the privileged KV rows is not equivalent to rerunning a censored teacher attention computation. That ambiguity must be resolved before (a) is a controlled experiment.")
    y = codebox(f, y, "src/selfupdate/eval/recite.py:79–93", "cache = DynamicCache()\nout = model(input_ids=input_ids, ..., past_key_values=cache, use_cache=True)\n...\nout = model(input_ids=torch.tensor([[next_tok]], device=device),\n            ..., past_key_values=cache, use_cache=True)")
    y = prose(f, y, "Experimental discipline", "(a) and (b) should be named as approximation arms, with a stated cache-validity rule and matched controls. They cannot be presented as invisible memory optimizations. The existing teacher_censored schedule is the clean reference for the teacher-vector idea, but it is a different objective/input stream from the active v5 summed cache arms.")
    save(pdf, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/v5_training.pdf")
    args = ap.parse_args()
    models = [read_model(*x) for x in ACTIVE]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        flow_page(pdf)
        cache_page(pdf, models)
        numbers_page(pdf, models)
        spans_page(pdf, models)
        schedule_page(pdf)
        vocab_page(pdf, models)
        peak_page(pdf, models)
        configuration_page(pdf, models)
        conclusion_page(pdf)
        recovered_pre_v5_page(pdf)
        recovered_delta_page(pdf)
        criticism_page(pdf)
        review_correction_page(pdf)
        online_design_page(pdf)
        kv_directions_page(pdf)
    print(out)


if __name__ == "__main__":
    main()
