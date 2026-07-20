#!/usr/bin/env bash
# M1 sequencer (task #6): waits for the resident leg (m1a), runs
# store -> store+adam -> store+rotate (m1b/c/d) on the
# same devices, then writes the numerics verdict. Detached launcher in the
# gpu_scheduler mold — the agent still reviews the verdict personally.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
PY=/tmp/$USER/selfupdate-venv/bin/python
BASE=configs/experiments/h100_smoke/base_qwen3_0p6b_v4_lora.yaml
OUT=runs/m1_verdict.txt

wait_leg() {  # wait until no train.py of this leg remains (bracketed grep)
  local tag="$1"
  while ps auxww | grep "train.py" | grep -v grep | grep -q "$tag"; do
    sleep 20
  done
}

run_leg() {
  local leg="$1"
  compressed/launch_v4_stages.sh "$BASE" \
    "configs/experiments/h100_smoke/${leg}.yaml" \
    >> "runs/${leg}_launch.log" 2>&1
  sleep 30
  wait_leg "$leg"
}

echo "M1 sequencer start $(date -Is)" > "$OUT"
wait_leg m1a_0p6b
echo "resident leg (m1a) done $(date -Is)" >> "$OUT"
for leg in m1b_0p6b_ppp2_store_e2 m1c_0p6b_ppp2_store_adam_e2 \
           m1d_0p6b_ppp2_store_adam_rotate_e2; do
  run_leg "$leg"
  echo "$leg done $(date -Is)" >> "$OUT"
done

echo "== resident vs store (m1a vs m1b; store relay; expect small bf16 boundary noise) ==" >> "$OUT"
"$PY" compressed/compare_v4_shard_numerics.py \
  runs/h100_m1a_0p6b_ppp2_resident_e2 runs/h100_m1b_0p6b_ppp2_store_e2 \
  --rtol 2e-3 >> "$OUT" 2>&1
echo "== store+adam vs store+rotate (m1c vs m1d; rotation; must be BIT-identical) ==" >> "$OUT"
"$PY" compressed/compare_v4_shard_numerics.py \
  runs/h100_m1c_0p6b_ppp2_store_adam_e2 \
  runs/h100_m1d_0p6b_ppp2_store_adam_rotate_e2 >> "$OUT" 2>&1

echo "== store+adam vs store+rotate Adam moments (m1c vs m1d; bitwise) ==" >> "$OUT"
"$PY" - >> "$OUT" 2>&1 <<'PYEOF'
import glob, torch
ok = True
c_files = sorted(glob.glob(
    "runs/h100_m1c_0p6b_ppp2_store_adam_e2/stage*/checkpoint/adam_moments.pt"))
if not c_files:
    print("NO adam_moments.pt found in the store+adam leg (m1c) — persistence path did not fire")
    ok = False
for cf in c_files:
    df = cf.replace("m1c_0p6b_ppp2_store_adam_e2",
                    "m1d_0p6b_ppp2_store_adam_rotate_e2")
    c, d = torch.load(cf, map_location="cpu"), torch.load(df, map_location="cpu")
    if sorted(c) != sorted(d):
        print(f"layer sets differ: {cf}")
        ok = False
        continue
    for layer in c:
        cs, ds = c[layer]["state"], d[layer]["state"]
        for idx in cs:
            for key in ("exp_avg", "exp_avg_sq"):
                a, b = cs[idx].get(key), ds[idx].get(key)
                if a is None or b is None or not torch.equal(a, b):
                    print(f"moment mismatch stage={cf} layer={layer} "
                          f"param={idx} {key}")
                    ok = False
print("ADAM MOMENTS BIT-IDENTICAL" if ok else "ADAM MOMENT MISMATCH")
PYEOF
echo "M1 sequencer end $(date -Is)" >> "$OUT"
