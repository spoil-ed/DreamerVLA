# Install Notes

## PyTorch

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

## LIBERO Editable Install Fix

### Problem

`LIBERO` upstream has an editable-install issue tracked in:

- `Issue #31`: `No module named 'libero' when trying to run main.py`
- `PR #84`: migrate packaging to `pyproject.toml`

The root cause is that the old `setup.py`-only packaging can fail under recent `pip` / `setuptools` when using:

```bash
pip install -e .
```

In that broken state:

- `pip show libero` may still look correct
- but `import libero` fails outside the repository root
- scripts such as `python scripts/check_dataset_integrity.py` raise `ModuleNotFoundError: No module named 'libero'`

### Fix Applied In This Repo

This repository uses the same fix direction as upstream `PR #84`:

- add `pyproject.toml`
- keep `setup.py` as a thin shim
- reinstall `LIBERO` in editable mode

### Install / Reinstall

```bash
cd "$DVLA_ROOT/third_party/LIBERO"
python -m pip install --no-build-isolation -e .
```

If you are using another checkout of `LIBERO`, apply the same packaging fix there and reinstall from that checkout instead.

### Verify The Fix

Check that `libero` is importable even outside the repo root:

```bash
cd /tmp
python -c "import libero; print(libero.__path__)"
```

Then verify the dataset integrity script can run directly:

```bash
cd "$DVLA_ROOT/third_party/LIBERO"
python scripts/check_dataset_integrity.py
```

If this command runs without `ModuleNotFoundError`, the packaging issue is fixed.

## LIBERO Dataset Notes

`LIBERO` does not always read datasets from the current repo. It uses the global config file:

```bash
$HOME/.libero/config.yaml
```

Check the active dataset path with:

```bash
grep '^datasets:' "$HOME/.libero/config.yaml"
```

The built-in checks mean:

- `check_libero_dataset(...)`: verifies expected `.hdf5` file counts
- `scripts/check_dataset_integrity.py`: verifies each `.hdf5` contains 50 demo trajectories and checks `tag == "libero-v1"`

For the default in-repo setup, the dataset path is:

```bash
$DVLA_ROOT/third_party/LIBERO/libero/datasets
```
