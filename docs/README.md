# Documentation Index

Current usage and implementation references. Architecture contracts live in
[`architecture/`](architecture/); change history is available through `git log`.

## Root References

- [`PARAMETERS.md`](PARAMETERS.md): parameter reference.
- [`install.md`](install.md): installation and environment setup.
- [`uv_environment.md`](uv_environment.md): native uv environments and CUDA profiles.
- [`docker_reproduction.md`](docker_reproduction.md): public 8xH100 Docker pull,
  asset preparation, WM/CLS warmup, and frozen Dreamer reproduction.
- [`repository_structure.md`](repository_structure.md): repository layout.
- [`data_layout.md`](data_layout.md): runtime data and artifact layout.
- [`pi05_prefix_input_latent.md`](pi05_prefix_input_latent.md): native
  pre-PaliGemma prefix-input WM and action-inference contract.
- [`vjepa2_ac_initialization.md`](vjepa2_ac_initialization.md): V-JEPA2-AC transfer,
  latent supervision, and independent decoder evaluation.
- [`reference/routes.md`](reference/routes.md): experiment/runner route matrix.
- [`reference/classifier_reward_balance.md`](reference/classifier_reward_balance.md):
  WMPO classifier labels and balanced sampling.
- [`../configs/environments/rlinf-libero-pi05/`](../configs/environments/rlinf-libero-pi05/README.md):
  isolated RLinf/OpenPI Python environment and container variables.

## Directories

- [`architecture/`](architecture/): architecture map for overview, complete loop, and Ray
  implementation.
- [`reference/`](reference/): structured references such as metrics and model/dataset
  notes.
- [`tutorials/experiments/`](tutorials/experiments/): runnable experiment recipes.
- [`papers/`](papers/): LaTeX paper workspaces, kept as self-contained projects.
- [`licenses/`](licenses/): retained upstream license notices; the project license
  remains at [`../LICENSE`](../LICENSE).
