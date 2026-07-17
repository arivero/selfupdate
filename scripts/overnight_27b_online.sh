#!/usr/bin/env bash
# Overnight chain (2026-07-17): once the vLLM exact-ids generation finishes,
# build the index-only cache and launch the infinite 27B PPP4 online run.
# Every step gates on verified artifacts, not exit codes.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=/tmp/$USER/selfupdate-venv/bin/python
RESP=runs/vllm_h100/qwen36_27b_full_exactids/responses_bs256.jsonl
BASE=configs/experiments/h100_smoke/base_qwen36_27b_v4_full.yaml
EXP=configs/experiments/h100_smoke/qwen36_27b_v4_ppp4_einf.yaml
log() { echo "$(date '+%H:%M:%S') $*"; }

log "waiting for vLLM responses at $RESP"
#until [ -s "$RESP" ]; do
#  if rg -q 'Traceback' runs/vllm_27b_exactids_gen.log 2>/dev/null; then
#    log "ABORT: vLLM generation crashed"; exit 1
#  fi
#  sleep 60
#done
#sleep 30  # let the writer finish the file
log "responses present; verifying token ids"
"$PY" - <<'PYEOF' || exit 1
import json
rows = [json.loads(l) for l in open(
    "runs/vllm_h100/qwen36_27b_full_exactids/responses_bs256.jsonl")]
assert len(rows) == 2071, f"expected 2071 rows, got {len(rows)}"
bad = [r["example_id"] for r in rows
       if not isinstance(r.get("token_ids"), list) or not r["token_ids"]]
assert not bad, f"{len(bad)} rows without token_ids, first {bad[:3]}"
print(f"verified: {len(rows)} rows with exact token ids")
PYEOF

log "building index-only cache"
TQDM_DISABLE=1 "$PY" scripts/build_teacher_cache.py --coordinated-node-cache \
  --config "$BASE" --experiment "$EXP" --index-only \
  > runs/h100_27b_index_build.log 2>&1
INDEX=$("$PY" - "$BASE" "$EXP" <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, "src")
from selfupdate.config import load_config
from selfupdate.teacher.cache import resolve_cache_dir
from selfupdate.teacher.node_epoch0 import ready_manifest, runtime_identity

cfg = load_config(sys.argv[1], sys.argv[2])
root, cache_hash = resolve_cache_dir(cfg)
if ready_manifest(root, cache_hash, compatibility=runtime_identity()) is None:
    raise SystemExit(f"node-epoch0 index was not atomically published: {root}")
print(root / "index.json")
PYEOF
) || {
  log "ABORT: index-only build did not publish a ready node cache"
  tail -5 runs/h100_27b_index_build.log
  exit 1
}
log "index at $INDEX"

log "launching einf PPP4 online"
scripts/launch_v4_stages.sh "$BASE" "$EXP" || { log "ABORT: launch refused"; exit 1; }
sleep 420
for k in 0 1 2 3; do
  E=$(rg -c '"kind": "v4_epoch"' "runs/h100_27b_v4_ppp4_einf/stage$k/metrics.jsonl" 2>/dev/null || echo 0)
  log "stage$k epoch rows after 7 min: $E"
done
log "overnight chain armed; stop with SIGTERM to the stage pids (lease file)"
