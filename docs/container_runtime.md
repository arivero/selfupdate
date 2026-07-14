# Container runtime contract

The in-repo teacher/cache and training commands run through
`scripts/container_exec.sh`. The expected layout is:

- `containers/pytorch-2.11.0-cu128-cudnn9-runtime.sif`: immutable base image;
  Python 3.12, Torch 2.11.0+cu128, CUDA 12.8, and cuDNN.
- `containers/selfupdate-python-deps-cu128.sqsh`: immutable Python overlay;
  Transformers 5.12.1, Accelerate 1.14.0, PEFT, and project dependencies.
- `/tmp/$USER/selfupdate-dev-python`: writable node-local development layer,
  mounted as `/dev-python`. It must be recreated on a new node for packages
  installed during the session, currently including `kernels==0.12.0`.
- The checkout is mounted at `/work`; commands passed to the container must
  use `/work/...` or paths relative to `/work`, never host paths such as
  `/fs/agustina/...`.

The launcher binds the Hugging Face cache as `/hf-cache`, sets `/work` as the
container working directory, and puts Singularity, TorchInductor, Triton,
temporary files, and the container home under `/tmp/$USER`. `/tmp` is
node-local and is disposable. Cache artifacts that must survive node changes
belong under `runs/` (usually only timings and reports; large transient
hidden-state shards should be removed after timing).

The external vLLM benchmark is intentionally outside this container:
`../venvs/vllm025/bin/python` runs vLLM 0.25.0 with its own Torch
2.11.0+cu129 environment. It must not be updated or modified. Consequently,
the two runtimes can report different CUDA minor versions while sharing the
same model files and exact response JSONL. Always record which runtime owns a
timing.

On a fresh node, verify four things before a run:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
scripts/container_exec.sh python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())'
scripts/container_pip.sh install --no-deps 'kernels==0.12.0'
../venvs/vllm025/bin/python -c 'import vllm, torch; print(vllm.__version__, torch.__version__)'
```

Do not copy a venv into the repo or install another Torch stack. If `/tmp` is
cleared, reinstall the pinned dev-layer packages and recreate transient
cache directories; no source change is implied by that cleanup.
