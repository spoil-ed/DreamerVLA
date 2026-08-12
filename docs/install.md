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

Download the current LIBERO cotrain assets:

```bash
bash scripts/download_assets.sh download.openvla_one_traj=true only=[10_openvla_oft_one_trajectory]
bash scripts/download_assets.sh only=[20_libero_dataset] env.LIBERO_SUITES=libero_goal
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

The π0.5 SFT/evaluation route uses the official OpenPI source and transforms. Clone
OpenPI at `third_party/openpi` or export `OPENPI_ROOT` to another checkout, then
install its dependencies into the active DreamerVLA environment. The released
`pi05_libero` weights are JAX; convert them with OpenPI's official converter:

```bash
export OPENPI_ROOT=/path/to/openpi
cd "${OPENPI_ROOT}"
python examples/convert_jax_model_to_pytorch.py \
  --config-name pi05_libero \
  --checkpoint-dir /path/to/pi05_libero \
  --output-path /path/to/pi05_libero_pytorch
cd "${DVLA_ROOT}"
```

For official π0.5 SFT, download `physical-intelligence/libero` without inheriting
proxy variables. Direct Hugging Face is preferred; when direct access is unavailable,
use an explicit mirror endpoint:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u ALL_PROXY -u all_proxy HF_ENDPOINT=https://hf-mirror.com \
  hf download physical-intelligence/libero --repo-type dataset \
  --local-dir data/datasets/lerobot/physical-intelligence/libero

torchrun --standalone --nproc-per-node=8 -m dreamervla.train \
  experiment=pi05_libero_sft \
  task.pi05.ckpt_path=/path/to/RLinf-Pi05-LIBERO-SFT
```

The SFT process is native PyTorch DDP. It does not start Ray, use FSDP, or import
the sibling RLinf repository at runtime; the port uses the installed OpenPI model
and retains RLinf's SFT loss, VLM-freeze, optimizer, and warmup semantics locally.

Training reads the completed local LeRobot root, so it performs no implicit dataset
download. The checkpoint directory must include
`physical-intelligence/libero/norm_stats.json`.

The converted directory must retain `model.safetensors` and the checkpoint's
normalization statistics. Evaluate a DreamerVLA SFT checkpoint with
`python -m dreamervla.train experiment=eval_pi05_libero eval.ckpt_path=/path/to/latest.ckpt`.

## Verify

```bash
bash scripts/install/60_verify.sh
python -m pytest tests/unit_tests -q
ruff check dreamervla tests
```

The fully executed native reproduction record for the separate environment is
in [`native_environment_reproduction.md`](native_environment_reproduction.md).
