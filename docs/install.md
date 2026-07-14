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

## Verify

```bash
bash scripts/install/60_verify.sh
python -m pytest tests/unit_tests -q
ruff check dreamervla tests
```
