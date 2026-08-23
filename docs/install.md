# Install Notes

`DVLA_ROOT` is the source checkout. `DVLA_DATA_ROOT` is the runtime asset root:

```bash
export DVLA_ROOT="$(pwd -P)"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
cd "${DVLA_ROOT}"
```

Install and activate the environment:

```bash
bash scripts/install_env.sh
conda activate dreamervla
```

For the Docker-independent, lockfile-backed uv environment, see
[`uv_environment.md`](uv_environment.md). The maintained CUDA profiles are
`cu124-train` (the official H100-compatible pins) and `cu130-train` (CUDA 13.0 /
Blackwell with PyTorch SDPA).

To create a separate environment without changing the repository default, pass
the Hydra-owned environment name to the installer and repeat it for direct step
scripts:

```bash
bash scripts/install_env.sh env.CONDA_ENV_NAME=dreamervla-2
conda activate dreamervla-2
CONDA_ENV_NAME=dreamervla-2 bash scripts/install/60_verify.sh
```

The direct `scripts/install/*.sh` entrypoints read shell variables rather than
Hydra overrides, which is why the verification command uses
`CONDA_ENV_NAME=dreamervla-2` before `bash`.

Run one install step when debugging:

```bash
bash scripts/install_env.sh only=[20_torch] force=true
```

## Versions

| Component | Default |
| --- | --- |
| Python | 3.11 |
| PyTorch | 2.5.1 |
| CUDA wheel index | cu124 |
| flash-attn | 2.7.1.post1 |

## Assets

Download the pinned LeRobot LIBERO data used by the π0.5 loader:

```bash
bash scripts/download_assets.sh 'only=[20_libero_dataset]'
```

The step downloads `physical-intelligence/libero` at revision
`a4336d589d589045d1c56423ffdf3b88a0e19b1f` into
`${DVLA_DATA_ROOT}/datasets/lerobot/physical-intelligence/libero`. It uses the
Hugging Face mirror and clears inherited proxy/VPN variables. OpenVLA one-trajectory
checkpoints are still opt-in:

```bash
bash scripts/download_assets.sh download.openvla_one_traj=true only=[10_openvla_oft_one_trajectory]
```

Optional CALVIN downloads:

```bash
bash scripts/download_assets.sh download.libero=false download.calvin=true \
  env.HF_ENDPOINT=https://hf-mirror.com env.CALVIN_DOWNLOAD_METHOD=hf_shards
bash scripts/download_assets.sh download.libero=false download.calvin=true \
  env.HF_ENDPOINT=https://hf-mirror.com env.CALVIN_DOWNLOAD_METHOD=hf_subsets
bash scripts/download_assets.sh download.libero=false download.calvin=true \
  env.CALVIN_DOWNLOAD_METHOD=opendatalab
```

## Optional π0.5 / OpenPI

The system-Docker/JFS-uv setup is defined by
[`environments/rlinf-libero-pi05`](../environments/rlinf-libero-pi05/README.md).
Its `embodied`, `libero`, and `pi05` extras install RLinf's published OpenPI
runtime into `/runtime/venvs/openpi-libero`; DreamerVLA runs the locally migrated
RLinf loader/transforms/FSDP SFT step. `OPENPI_ROOT`/`third_party/openpi` remains
an optional development fallback when changing upstream OpenPI itself.

The released JAX `pi05_libero` weights can be converted with OpenPI's official
converter. The normal DreamerVLA recipe instead consumes the published PyTorch
`lerobot/pi05_base` checkpoint directly:

```bash
export OPENPI_ROOT=/path/to/openpi
cd "${OPENPI_ROOT}"
python examples/convert_jax_model_to_pytorch.py \
  --config-name pi05_libero \
  --checkpoint-dir /path/to/pi05_libero \
  --output-path /path/to/pi05_libero_pytorch
cd "${DVLA_ROOT}"
```

The registered download step above is the canonical dataset command. The
equivalent direct command is shown only for diagnosis:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u ALL_PROXY -u all_proxy HF_ENDPOINT=https://hf-mirror.com \
  hf download physical-intelligence/libero --repo-type dataset \
  --revision a4336d589d589045d1c56423ffdf3b88a0e19b1f \
  --local-dir "${DVLA_DATA_ROOT}/datasets/lerobot/physical-intelligence/libero"

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u ALL_PROXY -u all_proxy HF_ENDPOINT=https://hf-mirror.com \
  hf download lerobot/pi05_base \
  --revision b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba \
  --local-dir /jfs/oss-import/simate_pretrain_checkpoints/lerobot/pi05_base

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u ALL_PROXY -u all_proxy HF_ENDPOINT=https://hf-mirror.com \
  hf download RLinf/RLinf-Pi05-LIBERO-SFT \
  --revision 45ccfcc4e28634f1576ebf78cab0fbe2fd82432d \
  --local-dir \
    /jfs/oss-import/simate_pretrain_checkpoints/RLinf/RLinf-Pi05-LIBERO-SFT

torchrun --standalone --nproc-per-node=8 -m dreamervla.train \
  experiment=pi05_libero_sft
```

The SFT process uses `torchrun` plus FSDP and does not start Ray. The port retains
RLinf's official OpenPI loader, LIBERO transforms, SFT loss, VLM-freeze,
gradient-accumulation, optimizer, and warmup semantics locally.

Training reads the completed local LeRobot root through the Hydra-selected
`LeRobotLIBERODataLoaderFactory`. The default is a completed local path; an
explicit Hugging Face repo id retains RLinf/OpenPI's normal download behavior.
The factory validates the official repo identity and pinned revision. SFT initializes model
parameters from `task.pi05.base_ckpt_path` and
loads `physical-intelligence/libero/norm_stats.json` from
`task.pi05.assets_path`. The default assets path is the official RLinf SFT repo,
so both Hugging Face snapshots remain unmodified.

The converted directory must retain `model.safetensors` and the checkpoint's
normalization statistics. Evaluate a DreamerVLA SFT checkpoint with
`python -m dreamervla.train experiment=eval_pi05_libero eval.ckpt_path=/path/to/latest.ckpt`.

Run the isolated runtime verifier before π0.5 training or evaluation:

```bash
python -m dreamervla.diagnostics.verify_pi05_runtime
```

The one-episode-per-task recipe retains flat milestone checkpoints at steps
2k, 5k, 10k, 20k, 40k, and 80k. Evaluate those checkpoints with
`experiment=eval_pi05_libero`; select a policy by LIBERO success rate rather
than by the final SFT loss.

## Verify

```bash
bash scripts/install/60_verify.sh
python -m pytest tests/unit_tests -q
ruff check dreamervla tests
```

The fully executed native reproduction record for the separate environment is
in [`native_environment_reproduction.md`](native_environment_reproduction.md).
