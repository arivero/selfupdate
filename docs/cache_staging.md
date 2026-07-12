# Node-local Hugging Face cache staging

The account cache at `$HOME/.cache/huggingface` is the durable source of
model snapshots. On a GPU node, stage only the models needed for a campaign
to node-local `/tmp` before launching work:

```bash
scripts/stage_hf_cache.sh                         # Qwen3 0.6B, 1.7B, 4B
scripts/stage_hf_cache.sh Qwen3-4B                # one model
SELFUPDATE_HF_STAGE=/tmp/$USER/my-hf scripts/stage_hf_cache.sh
```

The script preserves the account cache, resumes interrupted transfers, and
writes `.selfupdate-hf-stage-ready` only after a complete stage. The container
launcher automatically prefers that ready stage; until then it binds the
account cache. Set `SELFUPDATE_HF_CACHE_HOST` only to deliberately override
both choices.

Do not stage an entire account cache indiscriminately: large 14B/32B/MoE
snapshots consume local NVMe quickly. Check `df -h /tmp` and stage the exact
models needed. On `agpul02` at 2026-07-12, `/tmp` had 391 GB free and the
0.6B+1.7B+4B set occupied about 13 GB.
