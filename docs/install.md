# Install Notes

Use the resumable installer for a new machine:

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

Verify the environment:

```bash
bash scripts/install/40_verify.sh
python -m pytest tests/unit_tests -q
```
