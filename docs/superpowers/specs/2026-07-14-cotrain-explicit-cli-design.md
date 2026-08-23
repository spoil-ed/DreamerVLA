# Cotrain Explicit CLI Design

## Goal

Expose the mainline cotrain launch through one readable command:

```bash
bash scripts/experiments/cotrain/train.sh \
  --config openvla_libero \
  --wm_ckpt /path/to/wm.ckpt \
  --cls_ckpt /path/to/cls.ckpt
```

`openvla_libero` is the real Hydra experiment name and corresponds directly to
`configs/experiment/openvla_libero.yaml`; it is not a launcher-only alias.

## Launcher Contract

`dreamervla.launchers.cotrain` parses the three public options and translates them
at the Python boundary:

- `--config NAME` -> `experiment=NAME`
- `--wm_ckpt PATH` -> `init.world_model_state_ckpt=PATH`
- `--cls_ckpt PATH` -> `init.classifier_state_ckpt=PATH`

Both `--option value` and `--option=value` forms are accepted. Remaining arguments
must be Hydra `key=value` overrides and are forwarded unchanged. The shell script
remains a one-command launcher with no defaults or custom parsing.

The two checkpoint options form a pair: both supplied loads WM and classifier;
both omitted preserves random initialization; exactly one supplied is an error.
Paths are expanded, resolved, and checked before Hydra composition. Passing a
public option together with its equivalent Hydra override is rejected as ambiguous.
The existing checkpoint environment variables remain a compatibility fallback,
but public documentation uses the explicit flags.

## Experiment Rename

Rename `configs/experiment/dreamervla_wmcls_cotrain.yaml` to
`configs/experiment/openvla_libero.yaml`. Update active source documentation,
launcher defaults, and tests to use `openvla_libero`. Historical design and plan
documents remain unchanged because they describe prior repository states.

## Error Handling and Tests

Unit tests cover option translation, both accepted option syntaxes, paired
checkpoint validation, missing paths, duplicate public/Hydra inputs, trailing Hydra
overrides, and selection of the `openvla_libero` experiment. Existing Hydra-only
launches remain valid for internal callers. A dry-run launcher check verifies the
fully resolved command without starting Ray or allocating GPUs.
