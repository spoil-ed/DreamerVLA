# Data Shard Rotation + Dual HF/Torch Checkpoints — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** (A) Slice cold-start rollout data into reasonably-sized HDF5 shards via rotation, and (B) save/load every cotrain component (world model, policy, critic, classifier) in BOTH HF (`save_pretrained`-style) and torch (`.ckpt`) formats, interchangeably.

**Architecture:** Part A adds a `demos_per_shard` knob to `RolloutDumpWorker` that closes the current `RolloutDumpWriter` and opens the next (`ray_shard_000.hdf5` → `_001` …) once a shard fills; the read side (`offline_seed`) already loads all `*.hdf5`. Part B adds one generic `PreTrainedModel` wrapper (`dreamervla/utils/hf_module.py`) that wraps any `nn.Module` + its Hydra init-args as a HF dir (`config.json` + `model.safetensors`), a `training.checkpoint_format` flag (`torch`/`hf`/`both`, default `both`), and dual save/load at the three checkpoint sites.

**Tech Stack:** Python, PyTorch, h5py, OmegaConf/Hydra, `transformers` (`PreTrainedModel`/`PretrainedConfig`), `safetensors`, pytest.

**Spec/roadmap:** the "Data & checkpoint format conventions" section of `docs/experiment_tutorials/OpenVLA_Onetraj_LIBERO_coldstart_warmup_cotrain.md` (commit e8cab57).

**Commit rules:** every commit uses `git commit --signoff`; commit *descriptions* must not contain `===` or `/`; ruff runs on changed Python.

---

## File Structure

- **Modify** `dreamervla/workers/rollout/dump_worker.py` — add `demos_per_shard` + rotation (close/open writer, per-shard local demo index, numbered shard names). The only writer-lifecycle owner.
- **Modify** `dreamervla/runners/cold_start_ray_collect_runner.py` — thread `collect.demos_per_shard` into the dump `WorkerGroup` (both the synthetic and OFT plan paths).
- **Modify** `scripts/collect_parallel.sh` — merge step copies *all* `*.hdf5` per job (multi-shard), not just `ray_shard_000.hdf5`.
- **Create** `dreamervla/utils/hf_module.py` — generic `HFModuleConfig(PretrainedConfig)` + `HFModuleWrapper(PreTrainedModel)` + `save_module_pretrained(...)` / `load_module_pretrained(...)`. One reusable HF wrapper for all components.
- **Modify** `dreamervla/runners/online_cotrain_pipeline_runner.py` — dual save in `_save_wm_warmup`/`_save_cls_warmup`; HF-aware load in the resume paths; read `training.checkpoint_format`.
- **Modify** `dreamervla/runners/online_cotrain_runner.py` — dual save in `_save_cotrain_ckpt`; HF-aware load for `init.world_model_state_ckpt` / `init.classifier_state_ckpt`.
- **Modify** `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml` — add `training.checkpoint_format: both` + `collect.demos_per_shard` default.
- **Create** `tests/unit_tests/test_hf_module.py`, `tests/unit_tests/test_dump_shard_rotation.py`.

---

# PART A — Data shard rotation

## Task 1: `demos_per_shard` rotation in RolloutDumpWorker

**Files:**
- Modify: `dreamervla/workers/rollout/dump_worker.py`
- Test: `tests/unit_tests/test_dump_shard_rotation.py`

Current `RolloutDumpWorker` (verbatim): `__init__(self, reward_dir, hidden_dir, shard_name="ray_shard_000.hdf5", preprocess_config=None, data_attrs=None)`; `init()` sets `self.writer = RolloutDumpWriter(Path(reward_dir), Path(hidden_dir), self.shard_name)`; `add_episode` calls `self._writer().write_demo(index=int(self.num_episodes), ..., preprocess_config=self.preprocess_config if index==0 else None, data_attrs=... if index==0 else None, ...)` then `self.num_episodes += 1`; `size()` returns `self.num_episodes`. Rotation must NOT change behavior when `demos_per_shard == 0` (default).

- [ ] **Step 1: Write the failing test** (rotation logic, no real HDF5 — fake writer)

```python
# tests/unit_tests/test_dump_shard_rotation.py
from unittest import mock

from dreamervla.workers.rollout import dump_worker as dw


class _FakeWriter:
    created: list[str] = []
    def __init__(self, reward_dir, hidden_dir, shard_name):
        self.shard_name = str(shard_name)
        self.demos: list[int] = []
        _FakeWriter.created.append(self.shard_name)
    def write_demo(self, index, steps, preprocess_config=None, data_attrs=None, **kw):
        self.demos.append(int(index))
    def close(self):
        pass


def _episode():
    return [{"task_id": 0, "episode_id": 0, "task_description": "t", "success": True}]


def test_no_rotation_when_disabled():
    _FakeWriter.created = []
    with mock.patch.object(dw, "RolloutDumpWriter", _FakeWriter):
        w = dw.RolloutDumpWorker("r", "h", demos_per_shard=0)
        w.init()
        for _ in range(5):
            w.add_episode(_episode())
        assert _FakeWriter.created == ["ray_shard_000.hdf5"]   # single shard
        assert w.size() == 5


def test_rotates_every_n_demos():
    _FakeWriter.created = []
    with mock.patch.object(dw, "RolloutDumpWriter", _FakeWriter):
        w = dw.RolloutDumpWorker("r", "h", demos_per_shard=2)
        w.init()
        for _ in range(5):
            w.add_episode(_episode())
        # 5 demos / 2 per shard -> shards 000(2), 001(2), 002(1)
        assert _FakeWriter.created == [
            "ray_shard_000.hdf5", "ray_shard_001.hdf5", "ray_shard_002.hdf5"
        ]
        assert w.size() == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit_tests/test_dump_shard_rotation.py -v`
Expected: FAIL — `RolloutDumpWorker.__init__() got an unexpected keyword argument 'demos_per_shard'`.

- [ ] **Step 3: Implement rotation**

In `dreamervla/workers/rollout/dump_worker.py`, add `import re` at the top with the other imports. Change `__init__` to accept the knob + rotation state:

```python
    def __init__(
        self,
        reward_dir: str,
        hidden_dir: str,
        shard_name: str = "ray_shard_000.hdf5",
        preprocess_config: dict[str, Any] | None = None,
        data_attrs: dict[str, Any] | None = None,
        demos_per_shard: int = 0,
    ) -> None:
        self.reward_dir = reward_dir
        self.hidden_dir = hidden_dir
        self.shard_name = str(shard_name)
        self.preprocess_config = preprocess_config
        self.data_attrs = data_attrs
        self.demos_per_shard = int(demos_per_shard)
        self.writer: RolloutDumpWriter | None = None
        self.num_episodes = 0
        self._shard_idx = 0
        self._shard_demos = 0
```
(Keep whatever other attributes the current `__init__` set; only add the last three + the `demos_per_shard` line.)

Add a shard-name helper:
```python
    def _shard_name(self, idx: int) -> str:
        stem = self.shard_name[:-5] if self.shard_name.endswith(".hdf5") else self.shard_name
        base = re.sub(r"_\d+$", "", stem)
        return f"{base}_{idx:03d}.hdf5"
```

Change `init()` to open the first shard (original name when rotation is off, for byte-for-byte back-compat):
```python
    def init(self) -> None:
        first = self.shard_name if self.demos_per_shard <= 0 else self._shard_name(0)
        self.writer = RolloutDumpWriter(
            Path(self.reward_dir), Path(self.hidden_dir), first
        )
```

Change `add_episode` to rotate + use a per-shard local index when rotation is on:
```python
    def add_episode(self, episode: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not episode:
            return None
        if self.demos_per_shard > 0 and self._shard_demos >= self.demos_per_shard:
            self._writer().close()
            self._shard_idx += 1
            self._shard_demos = 0
            self.writer = RolloutDumpWriter(
                Path(self.reward_dir), Path(self.hidden_dir), self._shard_name(self._shard_idx)
            )
        first = episode[0]
        index = self._shard_demos if self.demos_per_shard > 0 else int(self.num_episodes)
        self._writer().write_demo(
            index=index,
            steps=episode,
            preprocess_config=self.preprocess_config if index == 0 else None,
            data_attrs=self.data_attrs if index == 0 else None,
            task_id=_optional_int(first.get("task_id")),
            episode_id=_optional_int(first.get("episode_id")),
            task_description=first.get("task_description"),
            episode_success=bool(episode[-1].get("success", False)),
            episode_horizon=len(episode),
        )
        self._shard_demos += 1
        self.num_episodes += 1
        return {"episode_index": int(self.num_episodes - 1), "length": len(episode)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit_tests/test_dump_shard_rotation.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add dreamervla/workers/rollout/dump_worker.py tests/unit_tests/test_dump_shard_rotation.py
git commit --signoff -m "feat(collect): shard rotation in RolloutDumpWorker via demos_per_shard"
```

## Task 2: thread `collect.demos_per_shard` through the collector

**Files:**
- Modify: `dreamervla/runners/cold_start_ray_collect_runner.py`
- Modify: `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml` (add the default)

- [ ] **Step 1: Add the config default**

In `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml`, under the `collect:` block, add:
```yaml
  demos_per_shard: 0   # 0 = one shard per job; >0 = roll a new shard every N episodes
```
(0 preserves current behavior; set e.g. 25 for reasonable slicing.)

- [ ] **Step 2: Pass it into both dump WorkerGroups**

In `cold_start_ray_collect_runner.py`, the OFT plan path hard-codes the dump cfg (`build_oft_worker_plan`, ~line 178-188) and `_build_oft_components` launches the dump group (~line 202-209). Read `demos_per_shard` from config and pass it as the trailing `RolloutDumpWorker` arg. Add near where `num_envs` is read in `_build_oft_components`:
```python
        demos_per_shard = self._int_from(("collect.demos_per_shard", "demos_per_shard"), 0)
```
and extend the OFT dump `WorkerGroup(...)` call:
```python
        dump_group = WorkerGroup(
            RolloutDumpWorker,
            str(dump_cfg["reward_dir"]),
            str(dump_cfg["hidden_dir"]),
            str(dump_cfg.get("shard_name", "ray_shard_000.hdf5")),
            dump_cfg["preprocess_config"],
            dump_cfg["data_attrs"],
            demos_per_shard,
        ).launch(cluster, NodePlacementStrategy(1))
```
Do the same for the synthetic/default dump `WorkerGroup` (~line 76-83): read the same knob and append it as the final positional arg. (`RolloutDumpWorker`'s new param is positional-after-`data_attrs`, matching Task 1.)

- [ ] **Step 3: Verify**

Run: `python -m py_compile dreamervla/runners/cold_start_ray_collect_runner.py` → exit 0.
Run: `python -m pytest tests/unit_tests/test_dump_shard_rotation.py -q` → still green.
Confirm by reading: both `WorkerGroup(RolloutDumpWorker, ...)` calls now pass `demos_per_shard` as the last arg.

- [ ] **Step 4: Commit**

```bash
git add dreamervla/runners/cold_start_ray_collect_runner.py configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml
git commit --signoff -m "feat(collect): wire collect.demos_per_shard into the dump workers"
```

## Task 3: parallel-collect merge handles multiple shards per job

**Files:**
- Modify: `scripts/collect_parallel.sh`

The merge currently copies only `coldstart_g{g}/reward/ray_shard_000.hdf5`. With rotation a job can emit `ray_shard_000.hdf5`, `_001`, … — all must be merged with unique, reward/hidden-matching names.

- [ ] **Step 1: Replace the merge loop**

Find the merge block (the `for g in "${GPUS_USED[@]}"` loop that `cp`s `ray_shard_000.hdf5`) and replace it with a glob over every shard in the job's reward dir:
```bash
for g in "${GPUS_USED[@]}"; do
  rwdir="$run_root/coldstart_g${g}/reward"
  [[ -d "$rwdir" ]] || { echo "[collect_parallel] WARN: no reward dir for GPU $g — skipping" >&2; continue; }
  shopt -s nullglob
  found=0
  for src_rw in "$rwdir"/*.hdf5; do
    base="$(basename "$src_rw")"; src_hd="$run_root/coldstart_g${g}/hidden/$base"
    [[ -f "$src_hd" ]] || { echo "[collect_parallel] WARN: missing hidden shard $base for GPU $g" >&2; continue; }
    # unique name keyed by GPU + original shard name, identical in reward/ and hidden/
    dst="g${g}_${base}"
    cp -f "$src_rw" "$run_root/coldstart/reward/$dst"
    cp -f "$src_hd" "$run_root/coldstart/hidden/$dst"
    found=1
  done
  shopt -u nullglob
  [[ "$found" == "0" ]] && echo "[collect_parallel] WARN: GPU $g produced no shards" >&2
done
```
(The `preprocess_config.json` copy line below it stays unchanged.)

- [ ] **Step 2: Verify**

Run: `bash -n scripts/collect_parallel.sh` → no syntax error.
Run a dry-run sanity (no real collection): create `mkdir -p /tmp/cpm/coldstart_g0/{reward,hidden} /tmp/cpm/coldstart/{reward,hidden}` and `touch /tmp/cpm/coldstart_g0/reward/ray_shard_000.hdf5 /tmp/cpm/coldstart_g0/hidden/ray_shard_000.hdf5`, then paste just the merge loop with `GPUS_USED=(0)` and `run_root=/tmp/cpm` into a shell; confirm `ls /tmp/cpm/coldstart/reward` shows `g0_ray_shard_000.hdf5`. (Or trust the read + bash -n.)

- [ ] **Step 3: Commit**

```bash
git add scripts/collect_parallel.sh
git commit --signoff -m "feat(collect): merge all per-job shards in collect_parallel"
```

---

# PART B — Dual HF + torch checkpoints

## Task 4: generic HF module wrapper

**Files:**
- Create: `dreamervla/utils/hf_module.py`
- Test: `tests/unit_tests/test_hf_module.py`

One reusable wrapper saves any `nn.Module` + its Hydra init-args as a HF dir (`config.json` + `model.safetensors`) and rebuilds it via `hydra.utils.instantiate`.

- [ ] **Step 1: Write the failing test** (round-trip with a real Hydra-instantiable module)

```python
# tests/unit_tests/test_hf_module.py
import torch

from dreamervla.utils.hf_module import save_module_pretrained, load_module_pretrained


def test_save_load_roundtrip(tmp_path):
    m = torch.nn.Linear(4, 3)
    with torch.no_grad():
        m.weight.fill_(0.5); m.bias.fill_(-0.25)
    d = tmp_path / "wm"
    save_module_pretrained(
        m, str(d), target="torch.nn.Linear", init_args={"in_features": 4, "out_features": 3}
    )
    assert (d / "config.json").is_file()
    assert (d / "model.safetensors").is_file()
    loaded = load_module_pretrained(str(d))
    assert isinstance(loaded, torch.nn.Linear)
    for k, v in m.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit_tests/test_hf_module.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dreamervla.utils.hf_module'`.

- [ ] **Step 3: Implement the wrapper**

```python
# dreamervla/utils/hf_module.py
"""Save/load any nn.Module as a HF-style dir (config.json + model.safetensors).

Wraps a plain module + its Hydra init-args in a PreTrainedModel so HF's
save_pretrained/from_pretrained machinery produces a portable checkpoint;
load rebuilds the inner module via hydra.utils.instantiate and loads weights.
"""

from __future__ import annotations

from typing import Any

import hydra
import torch
from transformers import PretrainedConfig, PreTrainedModel


class HFModuleConfig(PretrainedConfig):
    model_type = "dreamervla_module"

    def __init__(self, target: str = "", init_args: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self.target = target
        self.init_args = init_args or {}
        super().__init__(**kwargs)


class HFModuleWrapper(PreTrainedModel):
    config_class = HFModuleConfig

    def __init__(self, config: HFModuleConfig, module: torch.nn.Module | None = None) -> None:
        super().__init__(config)
        if module is None:
            module = hydra.utils.instantiate({"_target_": config.target, **config.init_args})
        self.module = module

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - not used for save/load
        return self.module(*args, **kwargs)


def save_module_pretrained(
    module: torch.nn.Module, save_dir: str, *, target: str, init_args: dict[str, Any]
) -> None:
    cfg = HFModuleConfig(target=target, init_args=dict(init_args))
    wrapper = HFModuleWrapper(cfg, module=module)
    wrapper.save_pretrained(save_dir, safe_serialization=True)


def load_module_pretrained(save_dir: str, *, map_location: str = "cpu") -> torch.nn.Module:
    wrapper = HFModuleWrapper.from_pretrained(save_dir)
    return wrapper.module.to(map_location)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit_tests/test_hf_module.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dreamervla/utils/hf_module.py tests/unit_tests/test_hf_module.py
git commit --signoff -m "feat(checkpoint): generic HF save_pretrained wrapper for nn.Modules"
```

## Task 5: `training.checkpoint_format` flag + a base helper

**Files:**
- Modify: `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml`
- Modify: `dreamervla/runners/base_runner.py`
- Test: `tests/unit_tests/test_hf_module.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit_tests/test_hf_module.py
import types
from omegaconf import OmegaConf
from dreamervla.runners.base_runner import BaseRunner


def _runner(fmt):
    obj = types.SimpleNamespace()
    obj.cfg = OmegaConf.create({"training": {"checkpoint_format": fmt}})
    obj.checkpoint_save_torch = types.MethodType(BaseRunner.checkpoint_save_torch, obj)
    obj.checkpoint_save_hf = types.MethodType(BaseRunner.checkpoint_save_hf, obj)
    return obj


def test_checkpoint_format_flags():
    assert _runner("both").checkpoint_save_torch() and _runner("both").checkpoint_save_hf()
    assert _runner("torch").checkpoint_save_torch() and not _runner("torch").checkpoint_save_hf()
    assert _runner("hf").checkpoint_save_hf() and not _runner("hf").checkpoint_save_torch()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit_tests/test_hf_module.py::test_checkpoint_format_flags -v`
Expected: FAIL — `AttributeError: ... 'checkpoint_save_torch'`.

- [ ] **Step 3: Implement the flag + helpers**

Add the config default in `online_cotrain_pipeline_libero_goal.yaml` under `training:`:
```yaml
  checkpoint_format: both   # torch | hf | both
```
Add to `BaseRunner` (after `print_config`):
```python
    def _checkpoint_format(self) -> str:
        return str(OmegaConf.select(self.cfg, "training.checkpoint_format", default="both")).lower()

    def checkpoint_save_torch(self) -> bool:
        return self._checkpoint_format() in ("torch", "both")

    def checkpoint_save_hf(self) -> bool:
        return self._checkpoint_format() in ("hf", "both")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit_tests/test_hf_module.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml dreamervla/runners/base_runner.py tests/unit_tests/test_hf_module.py
git commit --signoff -m "feat(checkpoint): add training.checkpoint_format flag and base helpers"
```

## Task 6: dual save in the cotrain pipeline warmup checkpoints

**Files:**
- Modify: `dreamervla/runners/online_cotrain_pipeline_runner.py`

`_save_wm_warmup`/`_save_cls_warmup` currently torch-only. Add HF dirs alongside, gated by the flag. The component config blocks (`world_model`, `classifier`) are the HF `init_args`.

- [ ] **Step 1: Add HF saves**

Add an import at the top: `from dreamervla.utils.hf_module import save_module_pretrained`. Replace the two save methods:
```python
    def _wm_warmup_hf_dir(self) -> str:
        return os.path.join(self.output_dir, "ckpt", "wm_warmup_hf")

    def _cls_warmup_hf_dir(self) -> str:
        return os.path.join(self.output_dir, "ckpt", "classifier_warmup_hf")

    def _save_wm_warmup(self) -> None:
        if self.checkpoint_save_torch():
            torch.save({"global_step": int(self.global_step),
                        "world_model": _unwrap(self.world_model).state_dict()}, self._wm_warmup_ckpt())
        if self.checkpoint_save_hf():
            wm_cfg = OmegaConf.to_container(OmegaConf.select(self.cfg, "world_model"), resolve=True)
            target = wm_cfg.pop("_target_")
            save_module_pretrained(_unwrap(self.world_model), self._wm_warmup_hf_dir(),
                                   target=target, init_args=wm_cfg)

    def _save_cls_warmup(self) -> None:
        if self.checkpoint_save_torch():
            torch.save({"global_step": int(self.global_step),
                        "classifier": _unwrap(self.classifier).state_dict(),
                        "classifier_threshold": float(self.classifier_threshold)}, self._cls_warmup_ckpt())
        if self.checkpoint_save_hf():
            cls_cfg = OmegaConf.to_container(OmegaConf.select(self.cfg, "classifier"), resolve=True)
            target = cls_cfg.pop("_target_")
            save_module_pretrained(_unwrap(self.classifier), self._cls_warmup_hf_dir(),
                                   target=target, init_args=cls_cfg)
```
NOTE: if the classifier is built from a dataclass config (not a direct `_target_` block), use the same kwargs dict that `_build_trainable_classifier` passes to `LatentSuccessClassifierConfig` as `init_args` and set `target="dreamervla.models.reward.latent_success_classifier.LatentSuccessClassifier"`. Read `_build_trainable_classifier` and reuse its `cls_kwargs` to keep save/load symmetric.

- [ ] **Step 2: Verify**

Run: `python -m py_compile dreamervla/runners/online_cotrain_pipeline_runner.py` → exit 0.
Run: `python -m pytest tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py tests/unit_tests/test_hf_module.py -q` → green.

- [ ] **Step 3: Commit**

```bash
git add dreamervla/runners/online_cotrain_pipeline_runner.py
git commit --signoff -m "feat(checkpoint): dual torch and HF warmup checkpoints in cotrain pipeline"
```

## Task 7: dual save in the cotrain loop checkpoint

**Files:**
- Modify: `dreamervla/runners/online_cotrain_runner.py`

`_save_cotrain_ckpt` writes `latest.ckpt` with all four components. Add HF dirs per component, gated by the flag.

- [ ] **Step 1: Add HF saves**

Add `from dreamervla.utils.hf_module import save_module_pretrained` to the imports. Extend `_save_cotrain_ckpt`:
```python
    def _save_cotrain_ckpt(self) -> None:
        ckpt_dir = os.path.join(self.output_dir, "ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.checkpoint_save_torch():
            torch.save(
                {
                    "global_step": int(self.global_step),
                    "world_model": _unwrap(self.world_model).state_dict(),
                    "policy": _unwrap(self.policy).state_dict(),
                    "critic": _unwrap(self.critic).state_dict(),
                    "classifier": _unwrap(self.classifier).state_dict(),
                    "classifier_threshold": float(self.classifier_threshold),
                },
                os.path.join(ckpt_dir, "latest.ckpt"),
            )
        if self.checkpoint_save_hf():
            for name, module, cfg_key in (
                ("world_model", self.world_model, "world_model"),
                ("policy", self.policy, "policy"),
                ("critic", self.critic, "critic"),
            ):
                blk = OmegaConf.to_container(OmegaConf.select(self._cfg_for_hf(), cfg_key), resolve=True)
                target = blk.pop("_target_")
                save_module_pretrained(_unwrap(module), os.path.join(ckpt_dir, f"latest_hf_{name}"),
                                       target=target, init_args=blk)
        print(f"[online-cotrain] ckpt -> {ckpt_dir}", flush=True)
```
Add a small accessor returning the cfg used to build components (store `self._cfg` in `_build_components`, or reuse the cfg already kept on the runner — read `_build_components`/`run` to find the stored config and expose `_cfg_for_hf()` returning it). For the classifier, mirror Task 6's note (use the dataclass kwargs + explicit target). Keep the existing single `latest.ckpt` print semantics.

- [ ] **Step 2: Verify**

Run: `python -m py_compile dreamervla/runners/online_cotrain_runner.py` → exit 0.
Run: `python -m pytest tests/unit_tests/test_hf_module.py tests/unit_tests/test_base_runner_console.py -q` → green.

- [ ] **Step 3: Commit**

```bash
git add dreamervla/runners/online_cotrain_runner.py
git commit --signoff -m "feat(checkpoint): dual torch and HF cotrain checkpoint per component"
```

## Task 8: HF-aware loading (resume + init ckpts)

**Files:**
- Modify: `dreamervla/runners/online_cotrain_pipeline_runner.py`, `dreamervla/runners/online_cotrain_runner.py`
- Reuse: `dreamervla/utils/hf_checkpoint.py` (`is_hf_checkpoint`)

Loads must accept EITHER a torch `.ckpt` OR an HF dir, so the two formats are interchangeable.

- [ ] **Step 1: HF-aware warmup resume**

In `online_cotrain_pipeline_runner.py`, the resume branches (`need_wm`/`need_cls`) currently `torch.load(self._wm_warmup_ckpt())`. Make resume prefer the torch ckpt, else fall back to the HF dir:
```python
        from dreamervla.utils.hf_module import load_module_pretrained
        # WM warmup resume:
        if os.path.exists(self._wm_warmup_ckpt()):
            payload = torch.load(self._wm_warmup_ckpt(), map_location="cpu", weights_only=False)
            _unwrap(self.world_model).load_state_dict(payload["world_model"])
        elif os.path.isdir(self._wm_warmup_hf_dir()):
            src = load_module_pretrained(self._wm_warmup_hf_dir())
            _unwrap(self.world_model).load_state_dict(src.state_dict())
```
Update the `need_wm = not (resume and os.path.exists(self._wm_warmup_ckpt()))` guard to also treat the HF dir as a valid resume source:
```python
        need_wm = not (resume and (os.path.exists(self._wm_warmup_ckpt()) or os.path.isdir(self._wm_warmup_hf_dir())))
        need_cls = not (resume and (os.path.exists(self._cls_warmup_ckpt()) or os.path.isdir(self._cls_warmup_hf_dir())))
```
Apply the analogous HF fallback to the classifier resume branch (load `src.state_dict()` into `_unwrap(self.classifier)`; keep `classifier_threshold` from the torch ckpt when present, else leave default).

- [ ] **Step 2: HF-aware init-ckpt loading**

In `online_cotrain_runner.py`, `_load_world_model_init_ckpt` and the `init.classifier_state_ckpt` warm-start currently `torch.load`. Guard with `is_hf_checkpoint`:
```python
        from dreamervla.utils.hf_checkpoint import is_hf_checkpoint
        from dreamervla.utils.hf_module import load_module_pretrained
        # world_model init:
        if is_hf_checkpoint(path):
            src = load_module_pretrained(path)
            self._unwrapped_world_model.load_state_dict(src.state_dict())
        else:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            ...existing extraction...
```
Mirror for the classifier `init.classifier_state_ckpt` (HF dir → `load_module_pretrained(...).state_dict()`; else the existing `payload.get("model", ...)` path). Confirm `is_hf_checkpoint`'s exact signature in `dreamervla/utils/hf_checkpoint.py` first and match it.

- [ ] **Step 3: Verify**

Run: `python -m py_compile dreamervla/runners/online_cotrain_pipeline_runner.py dreamervla/runners/online_cotrain_runner.py` → exit 0.
Run: `python -m pytest tests/unit_tests/test_hf_module.py tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py -q` → green.
Live (GPU) round-trip smoke (optional but recommended): run a `training.debug=true` warmup with `training.checkpoint_format=hf`, confirm `ckpt/wm_warmup_hf/{config.json,model.safetensors}` are written and a second `training.resume=true` run loads them without error. (Use the smoke command from the cotrain tutorial + `training.checkpoint_format=hf training.resume=true`.)

- [ ] **Step 4: Commit**

```bash
git add dreamervla/runners/online_cotrain_pipeline_runner.py dreamervla/runners/online_cotrain_runner.py
git commit --signoff -m "feat(checkpoint): load HF or torch checkpoints interchangeably"
```

---

## Self-Review

- **Spec coverage:**
  - "Data sharding — slice into reasonable shards": Task 1 (rotation) + Task 2 (knob) + Task 3 (parallel merge). ✓
  - "Checkpoints — dual HF + torch via save_pretrained, interchangeable": Task 4 (wrapper) + Task 5 (flag) + Task 6/7 (dual save) + Task 8 (dual load). ✓
  - All four components (WM/policy/critic/classifier): Task 6 (WM, classifier) + Task 7 (WM, policy, critic, classifier). ✓
- **Placeholder scan:** Task 6/7 leave one read-time detail — the classifier's HF `init_args`/`target` (it is built from `LatentSuccessClassifierConfig`, not a plain `_target_` block) and the runner's stored-cfg accessor (`_cfg_for_hf`). Both are explicitly flagged with how to resolve (reuse `_build_trainable_classifier`'s `cls_kwargs`; read where the runner stores its cfg). All other steps carry literal code. This is the one place the exact line depends on reading the classifier-build + cfg-storage code, which the implementer does at task time.
- **Type/name consistency:** `save_module_pretrained(module, save_dir, *, target, init_args)` / `load_module_pretrained(save_dir)` / `checkpoint_save_torch()` / `checkpoint_save_hf()` / `demos_per_shard` / `_shard_name` — used identically across tasks. ✓
- **Scope:** Part A and Part B are independent; each is committable and testable on its own. Default values (`demos_per_shard=0`, `checkpoint_format=both`) keep current behavior unless changed (HF saving is additive; torch still written under `both`).
