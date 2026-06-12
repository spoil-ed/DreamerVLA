# Install Notes

`DVLA_ROOT` points at the source checkout. `DVLA_DATA_ROOT` points at runtime
assets and may be anywhere with enough disk space:

```bash
export DVLA_ROOT=/path/to/DreamerVLA
export DVLA_DATA_ROOT=/path/to/dvla_data
cd "${DVLA_ROOT}"
```

Use the resumable one-command installer for a new machine:

```bash
bash scripts/install_env.sh
conda activate dreamervla
```

The installer runs `scripts/install/*.sh` in order and records completed steps
under `${DVLA_DATA_ROOT:-data}/install_state/`. Re-run a failed install with
the same command; completed steps are skipped. To force a step, use:

```bash
INSTALL_FORCE=1 INSTALL_ONLY=20_python_deps bash scripts/install_env.sh
```

Run install steps one at a time when debugging:

```bash
bash scripts/install/00_apt_tools.sh
bash scripts/install/10_conda_env.sh
bash scripts/install/20_python_deps.sh
bash scripts/install/30_third_party.sh
bash scripts/install/40_verify.sh
```

## Key Versions

| Component | Default |
| --- | --- |
| Python | 3.11 |
| PyTorch | 2.5.1 |
| CUDA wheel index | cu124 |
| flash-attn | 2.7.1.post1 |

## LIBERO

`scripts/install/30_third_party.sh` clones LIBERO and installs it in editable
mode after adding packaging metadata compatible with recent `pip` /
`setuptools`.

Launch scripts write `${DVLA_DATA_ROOT}/.libero/config.yaml` and point LIBERO
datasets to:

```text
${DVLA_DATA_ROOT}/datasets/libero
```

Download default assets in one command:

```bash
bash scripts/download_assets.sh
```

Download assets one step at a time:

```bash
bash scripts/download/10_worldvla.sh
bash scripts/download/20_lumina.sh
LIBERO_SUITES=libero_goal bash scripts/download/30_rynnvla.sh
LIBERO_SUITES=libero_goal bash scripts/download/40_libero_dataset.sh
bash scripts/download/50_calvin_dataset.sh
```

Verify the environment:

```bash
bash scripts/install/40_verify.sh
python -m pytest tests/unit_tests -q
```
