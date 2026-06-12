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
INSTALL_FORCE=1 INSTALL_ONLY=20_torch bash scripts/install_env.sh
```

Run install steps one at a time when debugging:

```bash
bash scripts/install/00_apt_tools.sh
bash scripts/install/10_conda_env.sh
bash scripts/install/20_torch.sh
bash scripts/install/30_python_deps.sh
bash scripts/install/40_third_party.sh
bash scripts/install/50_special_packages.sh
bash scripts/install/60_verify.sh
```

| Step | Scope | How to extend |
| --- | --- | --- |
| `00_apt_tools.sh` | system packages | Add only apt-level dependencies needed before Python packages. |
| `10_conda_env.sh` | conda environment | Change `CONDA_ENV_NAME` or `PYTHON_VERSION` through environment variables. |
| `20_torch.sh` | PyTorch CUDA wheels | Override `CUDA_INDEX_URL` if using a different CUDA wheel index. |
| `30_python_deps.sh` | DreamerVLA editable package and pip requirements | Add normal Python runtime packages to `requirements.txt`. |
| `40_third_party.sh` | LIBERO, robosuite-family packages, OpenSora, OpenVLA-OFT helpers | Add vendored upstream packages under `third_party/` and install them here. |
| `50_special_packages.sh` | flash-attn, egl_probe, optional apex / TensorNVMe | Add fragile wheels or host-specific GPU extensions here. |
| `60_verify.sh` | import and CUDA checks | Add lightweight import checks for newly required packages. |

## Key Versions

| Component | Default |
| --- | --- |
| Python | 3.11 |
| PyTorch | 2.5.1 |
| CUDA wheel index | cu124 |
| flash-attn | 2.7.1.post1 |

## LIBERO

`scripts/install/40_third_party.sh` clones LIBERO and the pinned robosuite
family, then installs vendored OpenSora and OpenVLA-OFT components in the same
style as the related WMPO installer. `scripts/install/50_special_packages.sh`
handles flash-attn, egl_probe, and optional apex / TensorNVMe.

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
LIBERO_SUITES=libero_goal bash scripts/download/10_rynnvla.sh
OPENVLA_OFT_REPOS="owner/repo:libero_goal_hdf5_latest_6650" bash scripts/download/20_openvla_oft.sh
bash scripts/download/30_openvla_oft_one_trajectory.sh
LIBERO_SUITES=libero_goal bash scripts/download/40_libero_dataset.sh
bash scripts/download/50_calvin_dataset.sh
```

Verify the environment:

```bash
bash scripts/install/60_verify.sh
python -m pytest tests/unit_tests -q
```
