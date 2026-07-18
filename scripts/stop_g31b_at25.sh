#!/usr/bin/env bash
# Owner instruction 2026-07-18: pause the 31B memorization test at epoch 25
# for a results check (graceful TERM at the boundary; checkpoints publish).
set -u
cd /fs/agustina/arivero/supercomplex/selfup_teacher || exit 1
M=runs/h100_g31b_v4_ppp4/stage0/metrics.jsonl
until [ "$(grep -c '"kind": "v4_epoch"' "$M" 2>/dev/null)" -ge 25 ]; do
  sleep 30
done
echo "epoch 25 reached $(date -Is), stopping" >> runs/g31b_stop25.log
for p in $(ps auxww | grep '[t]rain.py' | grep -v grep | awk '{print $2}'); do
  kill -TERM "$p"
done
sleep 60
for p in $(ps auxww | grep '[t]rain.py' | grep -v grep | awk '{print $2}'); do
  kill -TERM "$p"
done
echo "stop signals sent $(date -Is)" >> runs/g31b_stop25.log
