# DreamerVLA Setup

All commands run from the repository root. `DVLA_ROOT` is the checkout.
`DVLA_DATA_ROOT` is the runtime asset root and can be any writable path. If
unset, release scripts use relative `data`. See `docs/data_layout.md`.

A fresh checkout contains source code, configs, scripts, tests, and docs only.
`third_party/` and everything under `data/` are local install/download/generated
state. Build the tree in order: install third-party code, download raw
datasets/checkpoints, then generate `processed_data/<task>/*`.

```bash
cd DreamerVLA
export DVLA_ROOT="$(pwd -P)"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
```

Quick experiment tutorials:

- [RynnVLA_LIBERO](docs/experiment_tutorials/RynnVLA_LIBERO.md)
- [OpenVLA_Onetraj_LIBERO](docs/experiment_tutorials/OpenVLA_Onetraj_LIBERO.md)

## 1. Install

One-command install:

```bash
bash scripts/install_env.sh
conda activate dreamervla
```

The installer runs these resumable steps and writes
`${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/install_state/*.done`:

```text
scripts/install/00_apt_tools.sh
scripts/install/10_conda_env.sh
scripts/install/20_torch.sh
scripts/install/30_python_deps.sh
scripts/install/40_third_party.sh
scripts/install/50_special_packages.sh
scripts/install/60_verify.sh
```

Run a single step when needed:

```bash
bash scripts/install_env.sh only=[20_torch]
bash scripts/install_env.sh only=[20_torch] force=true
```

Install step map:

| Step | Scope |
| --- | --- |
| `00_apt_tools.sh` | apt packages such as git-lfs, ffmpeg, OpenGL / OSMesa, cmake, ninja |
| `10_conda_env.sh` | conda environment and Python version |
| `20_torch.sh` | PyTorch / torchvision / torchaudio CUDA wheels |
| `30_python_deps.sh` | DreamerVLA editable install and curated pip requirements |
| `40_third_party.sh` | LIBERO, robosuite-family packages, OpenSora, OpenVLA-OFT helpers |
| `50_special_packages.sh` | flash-attn, egl_probe, optional apex / TensorNVMe |
| `60_verify.sh` | import and CUDA visibility checks + OpenVLA-OFT transformers-fork assertion |

### OpenVLA-OFT requires the custom transformers fork

OpenVLA-OFT runs on **moojink's transformers fork**
(`git+https://github.com/moojink/transformers-openvla-oft.git`), which patches the
Llama attention to be **bidirectional** (`is_causal=False`) for OFT's parallel
action-chunk decoding. **Vanilla** `transformers` produces **0% / garbage OFT
actions** — and it is silent, because the fork and vanilla **both report
`transformers.__version__ == "4.40.1"`**, so every version check passes.

`30_python_deps.sh` deliberately does **not** pin a `transformers` version;
`40_third_party.sh` installs the fork as the single authoritative `transformers`
with `--force-reinstall` (overriding anything `peft`/`diffusers` pulled in
transitively), and `60_verify.sh` **fails the install** if the active
`transformers` is not the fork. Verify manually with:

```bash
python -c "import transformers,os; p=os.path.dirname(transformers.__file__)+'/models/llama/modeling_llama.py'; \
print(transformers.__version__, sum(1 for _ in open(p)), 'lines', '-> FORK' if 'is_causal=False' in open(p).read() else '-> VANILLA(0% OFT)')"
# fork: 4.40.1 / 1620 lines / FORK   ;  vanilla: 4.40.1 / 1566 lines / VANILLA(0% OFT)
```

**Offline / air-gapped machines** (the GitHub fetch above needs internet) — use
either:

```bash
# (a) stage the fork source/wheel locally, then point the installer at it:
TRANSFORMERS_OFT_FORK_SRC=/path/to/transformers-openvla-oft \
  bash scripts/install_env.sh only=[40_third_party]

# (b) the fork is pure Python (no compiled ext) — copy its package dir + dist-info
#     straight into the env's site-packages (e.g. extracted from the rlinf docker
#     image at /opt/venv/openvla-oft/lib/python3.11/site-packages):
DST=$(python -c "import site; print(site.getsitepackages()[0])")
mv "$DST/transformers" "$DST/transformers.vanilla.bak"
cp -r /path/to/fork/transformers           "$DST/transformers"
cp -r /path/to/fork/transformers-4.40.1.dist-info "$DST/transformers-4.40.1.dist-info"
```

> Note: the fork makes **all** Llama attention bidirectional, which is correct for
> OFT but wrong for any standard causal-LM use in the same env. This project is
> OFT/world-model centric (the WM is not a Llama), so the fork is the intended
> env-wide transformers.

## 2. Download Assets

One-command download:

```bash
bash scripts/download_assets.sh
```

This wrapper runs:

```bash
python -m dreamervla.launchers.workflow --config-name download "$@"
```

The shell file only sets `DVLA_ROOT`, `DVLA_DATA_ROOT`, and `PYTHONPATH`, then
passes your Hydra overrides to `configs/scripts/download.yaml`. The Python
launcher reads that YAML and executes the numbered shell steps. In practice,
you still run simple commands such as `bash scripts/download_assets.sh
only=[10_rynnvla]`.

Single asset steps:

```bash
bash scripts/download_assets.sh only=[10_rynnvla]
bash scripts/download_assets.sh download.openvla_oft=true only=[20_openvla_oft] \
  env.OPENVLA_OFT_REPOS=owner/repo:libero_goal_hdf5_latest_6650
bash scripts/download_assets.sh download.openvla_one_traj=true only=[30_openvla_oft_one_trajectory]
bash scripts/download_assets.sh download.rynnvla=false download.libero=true only=[40_libero_dataset]
bash scripts/download_assets.sh download.rynnvla=false download.libero=false download.calvin=true
```

Useful variants:

```bash
bash scripts/download_assets.sh env.LIBERO_SUITES='"libero_goal libero_object"'
bash scripts/download_assets.sh download.rynnvla=false download.libero=true env.LIBERO_SUITES=libero_spatial
bash scripts/download_assets.sh download.rynnvla=false download.libero=false download.calvin=true env.CALVIN_TASKS=task_ABCD_D
bash scripts/download_assets.sh only=[10_rynnvla]
bash scripts/download_assets.sh download.openvla_one_traj=true only=[30_openvla_oft_one_trajectory]
```

CALVIN domestic / mirror-friendly options:

```bash
# Hugging Face mirror, sharded 30 GB multi-part zip files for task_ABCD_D
bash scripts/download_assets.sh download.rynnvla=false download.libero=false download.calvin=true \
  env.HF_ENDPOINT=https://hf-mirror.com env.CALVIN_DOWNLOAD_METHOD=hf_shards

# Hugging Face mirror, structured subset zips for task_ABCD_D
bash scripts/download_assets.sh download.rynnvla=false download.libero=false download.calvin=true \
  env.HF_ENDPOINT=https://hf-mirror.com env.CALVIN_DOWNLOAD_METHOD=hf_subsets

# OpenDataLab domestic platform; requires login through openxlab
pip install -U openxlab
openxlab login
bash scripts/download_assets.sh download.rynnvla=false download.libero=false download.calvin=true \
  env.CALVIN_DOWNLOAD_METHOD=opendatalab
```

OpenVLA-OFT one-trajectory checkpoints support both Hugging Face download
methods:

```bash
# Method 1: git clone with Git LFS
bash scripts/download_assets.sh download.openvla_one_traj=true only=[30_openvla_oft_one_trajectory] \
  env.OPENVLA_ONE_TRAJ_DOWNLOAD_METHOD=git

# Method 2: huggingface-hub; set HF_ENDPOINT=https://hf-mirror.com if needed
bash scripts/download_assets.sh download.openvla_one_traj=true only=[30_openvla_oft_one_trajectory] \
  env.OPENVLA_ONE_TRAJ_DOWNLOAD_METHOD=hf
```

Canonical dataset roots:

```text
${DVLA_DATA_ROOT}/datasets/libero/libero_goal/
${DVLA_DATA_ROOT}/datasets/libero/libero_object/
${DVLA_DATA_ROOT}/datasets/libero/libero_spatial/
${DVLA_DATA_ROOT}/datasets/libero/libero_10/
${DVLA_DATA_ROOT}/datasets/calvin/
```

## 3. Preprocess LIBERO

Preprocess has two phases. Run the base data phase first, train/evaluate the
VLA checkpoint, then generate action-hidden sidecars from that trained VLA.
Do not generate action-hidden before VLA SFT when the experiment is meant to
use a trained policy head.

Base preprocessing for all four standard LIBERO suites:

```bash
bash scripts/preprocess_libero.sh
```

By default this runs `libero_goal libero_object libero_spatial libero_10` and
delegates each suite to `scripts/preprocess/prepare_libero_data.sh`. To process
a subset:

```bash
bash scripts/preprocess_libero.sh tasks='"libero_goal libero_object"'
```

For a single suite or a resumable rerun of one suite:

```bash
bash scripts/preprocess/prepare_libero_data.sh task=libero_goal
```

The preprocessing path is split into numbered child scripts, matching the
download and install flow. To rerun only one part:

```bash
bash scripts/preprocess/prepare_libero_data.sh task=libero_goal only=[20_pretokenize_dataset] gpus=0 num_procs=8
```

The base path writes fixed no-op-filtered artifacts before reward and hidden
sidecar generation:

```text
input: downloaded raw LIBERO HDF5 files
  ${DVLA_DATA_ROOT}/datasets/libero/${TASK}/*.hdf5

stage 1: replay and mark no-ops
  python -m dreamervla.preprocess.libero_utils.regenerate_libero_dataset_filter_no_op --keep-noops
  writes ${DVLA_DATA_ROOT}/processed_data/${TASK}/marked_t_256 with data/demo_*/noop_mask

stage 2: filter marked no-ops
  python -m dreamervla.preprocess.filter_marked_libero_hdf5 --filter-noops
  writes ${DVLA_DATA_ROOT}/processed_data/${TASK}/no_noops_t_256
```

Base outputs:

```text
${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/processed_data/${TASK}/marked_t_256
${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/processed_data/${TASK}/no_noops_t_256
${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/processed_data/${TASK}/no_noops_t_256_remaining_reward
${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/configs/${TASK}/his_1_third_view_wrist_w_state_1_256_pretokenize*.yaml
```

Useful variants:

```bash
bash scripts/preprocess/prepare_libero_data.sh task=libero_10 gpus=0,1,2,3 ngpu=4
bash scripts/preprocess/prepare_libero_data.sh task=libero_goal only=[10_hdf5_reward]
bash scripts/preprocess/prepare_libero_data.sh task=libero_goal only=[20_pretokenize_dataset] gpus=0 num_procs=8
VLA_CKPT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/outputs/vla/<run>/checkpoints/latest_hf" \
bash scripts/preprocess/prepare_libero_data.sh task=libero_goal only=[30_action_hidden]
bash scripts/preprocess/prepare_libero_data.sh task=libero_goal only=[40_validate]
```

## 4. Train

VLA SFT:

```bash
bash scripts/train_vla.sh experiment=vla_rynnvla_action_head task=libero_goal \
  gpus=0,1,2,3 ngpu=4 batch_size=20
```

RynnVLA SFT writes the legacy runner checkpoint and a Hugging Face sidecar
directory for direct inference. For example:

```text
${DVLA_DATA_ROOT}/outputs/vla/rynnvla_action_head/<run>/checkpoints/latest.ckpt
${DVLA_DATA_ROOT}/outputs/vla/rynnvla_action_head/<run>/checkpoints/latest_hf/
${DVLA_DATA_ROOT}/outputs/vla/rynnvla_action_head/<run>/checkpoints/epoch=..._hf/
```

The pipeline is serial: train or download the VLA first, then generate
action-hidden sidecars from that exact VLA checkpoint, then train the world
model/classifier/DreamerVLA routes. Use the `*_hf/` directory for VLA eval and
for action-hidden extraction. Use `latest.ckpt` only when you need
optimizer/epoch state for legacy resume.
To resume from a HF directory as weights-only initialization:

```bash
bash scripts/train_vla.sh experiment=vla_rynnvla_action_head task=libero_goal \
  training.resume=true training.resume_dir="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/outputs/vla/<run>/checkpoints/latest_hf"
```

One-trajectory VLA:

```bash
bash scripts/train_vla.sh experiment=vla_sft_one_trajectory task=libero_goal
bash scripts/train_vla.sh experiment=vla_sft_one_trajectory task=libero_object dataset.trajectory_offset=2
bash scripts/train_vla.sh experiment=openvla_oft_hdf5_one_trajectory task=libero_10
bash scripts/train_vla.sh experiment=openvla_oft_hdf5_one_trajectory_l1 task=libero_goal
```

`openvla_oft_hdf5_one_trajectory` trains the discrete LM-head action-token
variant; `openvla_oft_hdf5_one_trajectory_l1` keeps the standard OFT L1
regression head, whose component checkpoints feed the action-hidden sidecar
and world-model chain.

After VLA SFT, generate RynnVLA action-hidden from the trained HF checkpoint:

```bash
TASK=libero_goal \
VLA_CKPT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/outputs/vla/<run>/checkpoints/latest_hf" \
GPUS=0 ACTION_HIDDEN_GPUS=1 \
bash scripts/preprocess/30_action_hidden.sh
```

`scripts/preprocess/30_action_hidden.sh` also accepts a legacy runner
checkpoint file via `VLA_CKPT=${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/outputs/vla/<run>/checkpoints/latest.ckpt`; in that case it uses
the base VLA under `BASE_VLA_CKPT` for assets and loads the trained encoder
state from the `.ckpt` file. Prefer the HF sidecar when it exists because the
same path can be passed directly to eval.

OpenVLA-OFT action-hidden sidecar (feeds the OFT WM/classifier/DreamerVLA
routes; both checkpoint formats extract the same backbone layer):

```bash
# Component-wise L1 checkpoint (auto-detected):
TASK=libero_goal \
OFT_CKPT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/checkpoints/OpenVLA-OFT/libero_goal_hdf5_latest_6650" \
OFT_POLICY_MODE=auto \
OFT_LATENT_SCHEME=action_hidden \
bash scripts/preprocess/35_oft_action_hidden.sh

# Downloaded discrete one-trajectory weights (single view, no history/proprio):
TASK=libero_goal \
OFT_CKPT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1" \
OFT_POLICY_MODE=discrete \
OFT_HISTORY=1 \
OFT_IMAGE_KEYS=agentview_rgb \
bash scripts/preprocess/35_oft_action_hidden.sh
```

Scheme B input-token sidecars feed frame-level DINO-WM routes.  RynnVLA uses
current-frame Chameleon VQ input-token embeddings; OFT uses current-frame
projected vision patch tokens:

```bash
# RynnVLA Scheme B from a trained HF checkpoint:
TASK=libero_goal \
VLA_CKPT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/outputs/vla/<run>/checkpoints/latest_hf" \
GPUS=0 ACTION_HIDDEN_GPUS=1 \
bash scripts/preprocess/32_input_token_hidden.sh

# OpenVLA-OFT Scheme B:
TASK=libero_goal OFT_LATENT_SCHEME=input_tokens \
OFT_CKPT=data/checkpoints/OpenVLA-OFT/libero_goal_hdf5_latest_6650 \
bash scripts/preprocess/35_oft_action_hidden.sh
```

First run on a new checkpoint: smoke-test by invoking the module directly with
`--max-files 1 --max-demos-per-file 2` before committing GPU hours.
Per model-on-dataset details (checkpoint formats, hidden shapes, expected
attrs): [docs/model_datasets/openvla_oft_libero_goal.md](docs/model_datasets/openvla_oft_libero_goal.md)
and [docs/model_datasets/rynnvla_libero_goal.md](docs/model_datasets/rynnvla_libero_goal.md).

Scheme A is the RynnVLA-002 contract: action-query/action-slot hidden states
are the WM observation tokens.  RynnVLA-002 uses 35 tokens (`5 × 7`) of dim
1024; OFT Scheme A uses the same downstream contract with 56 tokens (`8 × 7`)
of dim 4096.  Scheme B is intentionally different: input-side image tokens
are frame-level visual tokens, so they enter the WM as per-frame observations
and actions remain separate controls.

Discrete sidecars are written with `action_head_type=oft_discrete_token`; when
training the WM on them, point the route at the sidecar and align the expected
attrs, e.g.:

```bash
bash scripts/train_wm.sh experiment=oft_world_model_dinowm_chunk task=libero_goal \
  task.openvla_oft.ckpt_path="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1" \
  task.openvla_oft.action_hidden_dir="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/processed_data/<task>/<sidecar_dir>" \
  task.openvla_oft.expected_action_head_type=oft_discrete_token \
  task.openvla_oft.expected_history=1 \
  task.openvla_oft.expected_include_state=false
```

World model:

```bash
bash scripts/train_wm.sh experiment=world_model_dinowm_chunk task=libero_goal \
  gpus=0,1,2,3 ngpu=4 batch_size=16 \
  task.vla_ckpt_path="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/outputs/vla/<run>/checkpoints/latest_hf"

# Scheme B frame-token WM:
bash scripts/train_wm.sh experiment=world_model_dinowm_chunk_input_tokens task=libero_goal \
  task.vla_ckpt_path="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/outputs/vla/<run>/checkpoints/latest_hf"
bash scripts/train_wm.sh experiment=oft_world_model_dinowm_chunk_input_tokens task=libero_goal
```

Classifier:

```bash
bash scripts/train_wm.sh experiment=latent_classifier_libero_goal_chunk
bash scripts/train_wm.sh experiment=latent_classifier_libero_goal_chunk_input_tokens
bash scripts/train_wm.sh experiment=oft_latent_classifier_chunk_input_tokens task=libero_goal
```

DreamerVLA:

```bash
bash scripts/train_dreamervla.sh \
  experiment=dreamervla_rynn_dino_wm_wmpo_outcome \
  task=libero_goal \
  gpus=0,1,2,3 ngpu=4 batch_size=4 \
  task.vla_ckpt_path="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/outputs/vla/<run>/checkpoints/latest_hf" \
  init.world_model_state_ckpt="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/outputs/worldmodel/<run>/checkpoints/latest.ckpt" \
  init.classifier_state_ckpt="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/outputs/classifier/<run>/checkpoints/latest.ckpt"

# Scheme B uses a bridge actor from frame tokens to action slots:
bash scripts/train_dreamervla.sh \
  experiment=dreamervla_rynn_dino_wm_wmpo_outcome_input_tokens \
  task=libero_goal \
  task.vla_ckpt_path="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/outputs/vla/<run>/checkpoints/latest_hf" \
  init.world_model_state_ckpt="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/outputs/worldmodel/<run>/checkpoints/latest.ckpt" \
  init.classifier_state_ckpt="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/outputs/classifier/<run>/checkpoints/latest.ckpt"
```

Training launchers are Hydra wrappers: `experiment=...` selects a script-level
config group under `configs/experiment/`, while overrides such as
`training.max_steps=1`, `dataset.trajectory_offset=3`, or
`task.hdf5_dir=${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/processed_data/<task>/no_noops_t_256` are passed through to the real training route.

## 5. Evaluate

VLA checkpoint:

```bash
bash scripts/eval_libero_vla.sh gpus=0 \
  eval.ckpt_kind=auto \
  eval.ckpt_path="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/outputs/vla/<run>/checkpoints/latest_hf" \
  eval.task_suite_name=libero_goal \
  eval.num_episodes_per_task=10 \
  training.device=cuda:0
```

`eval.ckpt_path` accepts both trained RynnVLA HF directories and legacy
DreamerVLA `.ckpt` payloads. For a downloaded base VLA with no SFT checkpoint,
omit `eval.ckpt_path` and set `init.vla_ckpt_path=${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/checkpoints/<hf_dir>`.

Dreamer checkpoint:

```bash
bash scripts/eval_libero_vla.sh gpus=0 \
  eval.ckpt_kind=dreamer \
  eval.ckpt_path="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/outputs/dreamervla/<run>/checkpoints/latest.ckpt" \
  eval.dreamer_policy_source=ckpt \
  eval.dreamer_actor_input_source=rssm \
  eval.task_suite_name=libero_goal \
  eval.num_episodes_per_task=10 \
  training.device=cuda:0
```

OpenVLA-OFT one-trajectory checkpoint:

```bash
CKPT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1" \
SUITE=libero_goal \
GPU_ID=0 \
bash scripts/eval/launch_openvla_oft_official_libero_eval.sh
```

## 6. Verify

```bash
python -m pytest tests/unit_tests -q
```

Path checks:

```bash
test -d "${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/checkpoints/VLA_model_256/libero_goal"
test -d "${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/datasets/libero/libero_goal"
test -d "${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}/processed_data/libero_goal/no_noops_t_256"
```

Smoke train:

```bash
bash scripts/train_wm.sh experiment=world_model_dinowm_chunk task=libero_goal \
  out_dir=/tmp/dvla_wm_smoke max_steps=1 num_workers=0
```
