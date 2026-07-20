#!/usr/bin/env bash
# Sequential frontier-model downloads for the agpuh02 lanes (2026-07-06).
# --max-workers 2 (owner directive); one at a time so Lustre and the proxy
# are never saturated. Each completed download touches a sentinel that
# queue_agpuh02.tsv uses as an `after` dependency — queued jobs stay
# parked instead of crash-looping on missing weights.
#
# Launch: nohup setsid bash compressed/download_mlt2.sh >> runs/hf_download_mlt2.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
export SSL_CERT_FILE=/fs/agustina/arivero/supercomplex/.local/lib/python3.11/site-packages/certifi/cacert.pem
# The runtime venv lives in node-local /tmp (repo law: no Lustre .venv). Point
# the CLI there; override with HF_CLI=... if a node uses a different path.
HF_CLI="${HF_CLI:-/tmp/$USER/selfupdate-venv/bin/hf}"

dl() {  # dl <repo> <sentinel>
    local repo="$1" sentinel="$2"
    if [ -e "$sentinel" ]; then echo "[skip] $repo"; return 0; fi
    echo "[$(date '+%F %T')] downloading $repo"
    if "$HF_CLI" download "$repo" --max-workers 2; then
        touch "$sentinel"
        echo "[$(date '+%F %T')] done $repo"
    else
        echo "[$(date '+%F %T')] FAILED $repo (no sentinel written)"
    fi
}

# Envelope target: the only model not yet on Lustre (owner decision 2026-07-17,
# FP8 variant, ~406 GB). Everything below it is already complete in the cache.
dl Qwen/Qwen3.5-397B-A17B-FP8        runs/.dl_qwen35_397b_fp8.done
dl Qwen/Qwen3.5-122B-A10B            runs/.dl_qwen35_122b.done
dl Qwen/Qwen3.5-122B-A10B-GPTQ-Int4  runs/.dl_qwen35_122b_gptq.done
dl mistralai/Mistral-Medium-3.5-128B runs/.dl_mistralmed35.done
dl deepseek-ai/DeepSeek-V4-Flash     runs/.dl_dsv4flash.done
dl zai-org/GLM-5.2                   runs/.dl_glm52.done
echo "[$(date '+%F %T')] all downloads processed"
