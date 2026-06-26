# Cotrain P1 Task Conditioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add rollbackable explicit `task_ids` conditioning for multi-task cotrain while leaving default single-task training behavior unchanged.

**Architecture:** Keep `task_conditioning.enabled=false` by default. When enabled, `OnlineReplay` already returns `task_ids`; runner validation requires selected WM and classifier implementations to declare `supports_task_conditioning`; WM/classifier forward paths consume `task_ids` without importing concrete sibling implementations.

**Tech Stack:** Python 3.11, PyTorch `nn.Embedding`, Hydra config interpolation, pytest.

---

## File Structure

- Modify: `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml:124`
  - Keep top-level task-conditioning config and pass it into WM/classifier component configs.
- Modify: `dreamervla/runners/online_cotrain_runner.py:136`
  - Keep capability validation and ensure it runs after components are built.
- Modify: `dreamervla/models/reward/latent_success_classifier.py:16`
  - Add optional classifier task embedding and `task_ids` forward argument.
- Modify: `dreamervla/models/world_model/dino_wm_chunk.py:155`
  - Add optional task embedding and consume `task_ids` in train/observe paths.
- Modify: `dreamervla/algorithms/dreamervla.py:112`
  - Already forwards `task_ids`; verify tests cover this.
- Modify: `tests/unit_tests/test_online_cotrain_pipeline.py:426`
  - Extend classifier task-id test to real classifier.
- Create: `tests/unit_tests/test_task_conditioning.py`
  - Focused tests for default-off behavior, enabled classifier behavior, and runner fail-fast.

## Task 1: Wire Task Conditioning Through Hydra Component Config

**Files:**
- Modify: `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml:124`
- Modify: `tests/unit_tests/test_online_cotrain_pipeline_config.py`

- [ ] **Step 1: Add a config composition test**

Add to `tests/unit_tests/test_online_cotrain_pipeline_config.py`:

```python
def test_task_conditioning_config_is_component_visible():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "task_conditioning": {
                "enabled": True,
                "num_tasks": 10,
                "embedding_dim": 64,
            },
            "world_model": {"task_conditioning": "${task_conditioning}"},
            "classifier": {"task_conditioning": "${task_conditioning}"},
        }
    )
    resolved = OmegaConf.to_container(cfg, resolve=True)

    assert resolved["world_model"]["task_conditioning"]["enabled"] is True
    assert resolved["classifier"]["task_conditioning"]["num_tasks"] == 10
```

- [ ] **Step 2: Run the config test**

Run:

```bash
pytest tests/unit_tests/test_online_cotrain_pipeline_config.py::test_task_conditioning_config_is_component_visible -q
```

Expected: PASS after config interpolation is added; FAIL if component configs cannot see the top-level block.

- [ ] **Step 3: Add component config interpolation**

In `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml`, keep the top-level block:

```yaml
task_conditioning:
  enabled: false
  num_tasks: 10
  embedding_dim: 64
```

Under `world_model`, add:

```yaml
  task_conditioning: ${task_conditioning}
```

Under `classifier`, add:

```yaml
  task_conditioning: ${task_conditioning}
```

- [ ] **Step 4: Verify the config test**

Run:

```bash
pytest tests/unit_tests/test_online_cotrain_pipeline_config.py::test_task_conditioning_config_is_component_visible -q
```

Expected: PASS.

## Task 2: Add Optional Task Conditioning to the Classifier

**Files:**
- Modify: `dreamervla/models/reward/latent_success_classifier.py:16`
- Create: `tests/unit_tests/test_task_conditioning.py`

- [ ] **Step 1: Write classifier tests**

Create `tests/unit_tests/test_task_conditioning.py`:

```python
from __future__ import annotations

import torch

from dreamervla.models.reward.latent_success_classifier import LatentSuccessClassifier


def test_latent_success_classifier_default_has_no_task_conditioning() -> None:
    model = LatentSuccessClassifier(
        latent_dim=4,
        window=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        head_type="linear",
    )

    logits = model(torch.zeros(3, 2, 4))

    assert logits.shape == (3, 2)
    assert model.supports_task_conditioning is False


def test_latent_success_classifier_uses_task_ids_when_enabled() -> None:
    model = LatentSuccessClassifier(
        latent_dim=4,
        window=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        head_type="linear",
        task_conditioning={
            "enabled": True,
            "num_tasks": 3,
            "embedding_dim": 4,
        },
    )
    windows = torch.zeros(2, 2, 4)

    logits_a = model(windows, task_ids=torch.tensor([0, 1]))
    logits_b = model(windows, task_ids=torch.tensor([1, 0]))

    assert model.supports_task_conditioning is True
    assert logits_a.shape == (2, 2)
    assert not torch.allclose(logits_a, logits_b)
```

- [ ] **Step 2: Run classifier tests**

Run:

```bash
pytest tests/unit_tests/test_task_conditioning.py::test_latent_success_classifier_default_has_no_task_conditioning tests/unit_tests/test_task_conditioning.py::test_latent_success_classifier_uses_task_ids_when_enabled -q
```

Expected before implementation: FAIL because `LatentSuccessClassifier.forward()` does not accept `task_ids`.

- [ ] **Step 3: Extend classifier config dataclass**

In `dreamervla/models/reward/latent_success_classifier.py`, add this field to `LatentSuccessClassifierConfig`:

```python
    task_conditioning: dict | None = None
```

- [ ] **Step 4: Add classifier embedding in `__init__`**

After `self.cfg = cfg`, add:

```python
        task_cfg = dict(getattr(cfg, "task_conditioning", None) or {})
        self.task_conditioning_enabled = bool(task_cfg.get("enabled", False))
        self.supports_task_conditioning = bool(self.task_conditioning_enabled)
        if self.task_conditioning_enabled:
            num_tasks = int(task_cfg.get("num_tasks", 0) or 0)
            embedding_dim = int(task_cfg.get("embedding_dim", 0) or 0)
            if num_tasks <= 0 or embedding_dim <= 0:
                raise ValueError(
                    "classifier.task_conditioning requires positive num_tasks and embedding_dim"
                )
            if embedding_dim != int(cfg.latent_dim):
                raise ValueError(
                    "LatentSuccessClassifier task_conditioning.embedding_dim must match "
                    f"latent_dim ({embedding_dim} != {int(cfg.latent_dim)})"
                )
            self.task_embedding = nn.Embedding(num_tasks, int(cfg.latent_dim))
        else:
            self.task_embedding = None
```

- [ ] **Step 5: Accept `task_ids` in classifier forward**

Change the signature:

```python
    def forward(
        self,
        latent_window: torch.Tensor,
        *,
        task_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
```

Before selecting the head type, add:

```python
        if self.task_conditioning_enabled:
            if task_ids is None:
                raise ValueError("task_ids are required when classifier task conditioning is enabled")
            task_emb = self.task_embedding(task_ids.to(latent_window.device).long())
            latent_window = latent_window + task_emb[:, None, :].to(latent_window.dtype)
```

- [ ] **Step 6: Pass task IDs in `predict_success` only when caller supplies them**

Keep `predict_success()` default unchanged for current LUMOS imagination. Add optional argument:

```python
        task_ids: torch.Tensor | None = None,
```

Inside the scan loop, call:

```python
            logits = self(window, task_ids=task_ids)
```

- [ ] **Step 7: Verify classifier task conditioning**

Run:

```bash
pytest tests/unit_tests/test_task_conditioning.py tests/unit_tests/test_online_cotrain_pipeline.py::test_task_conditioned_classifier_receives_replay_task_ids -q
```

Expected: selected tests pass.

## Task 3: Add Optional Task Conditioning to Chunk-Aware WM

**Files:**
- Modify: `dreamervla/models/world_model/dino_wm_chunk.py:155`
- Create: `tests/unit_tests/test_task_conditioning.py`

- [ ] **Step 1: Write WM task-conditioning test**

Append to `tests/unit_tests/test_task_conditioning.py`:

```python
def test_chunk_wm_declares_task_conditioning_support_when_enabled() -> None:
    from dreamervla.models.world_model.dino_wm_chunk import ChunkAwareDinoWMWorldModel

    wm = ChunkAwareDinoWMWorldModel(
        chunk_size=2,
        obs_dim=8,
        action_dim=7,
        token_count=2,
        token_dim=4,
        action_emb_dim=2,
        num_action_repeat=1,
        model_dim=6,
        depth=1,
        heads=2,
        dim_head=2,
        mlp_dim=16,
        num_hist=2,
        chunk_rollout_chunks=1,
        task_conditioning={
            "enabled": True,
            "num_tasks": 3,
            "embedding_dim": 4,
        },
    )

    assert wm.supports_task_conditioning is True
    batch = {
        "obs_embedding": torch.zeros(2, 4, 8),
        "actions": torch.zeros(2, 4, 7),
        "rewards": torch.zeros(2, 4),
        "dones": torch.zeros(2, 4),
        "is_first": torch.zeros(2, 4, dtype=torch.bool),
        "task_ids": torch.tensor([0, 1]),
    }
    out = wm(batch)
    assert "loss" in out or "_loss" in out
```

- [ ] **Step 2: Run the WM test**

Run:

```bash
pytest tests/unit_tests/test_task_conditioning.py::test_chunk_wm_declares_task_conditioning_support_when_enabled -q
```

Expected before implementation: FAIL because `ChunkAwareDinoWMWorldModel` does not declare support or consume `task_ids`.

- [ ] **Step 3: Add WM constructor argument**

In `ChunkAwareDinoWMWorldModel.__init__()`, add:

```python
        task_conditioning: dict | None = None,
```

After `self.attn_impl = str(attn_impl)`, add:

```python
        task_cfg = dict(task_conditioning or {})
        self.task_conditioning_enabled = bool(task_cfg.get("enabled", False))
        self.supports_task_conditioning = bool(self.task_conditioning_enabled)
        if self.task_conditioning_enabled:
            num_tasks = int(task_cfg.get("num_tasks", 0) or 0)
            embedding_dim = int(task_cfg.get("embedding_dim", 0) or 0)
            if num_tasks <= 0 or embedding_dim <= 0:
                raise ValueError(
                    "world_model.task_conditioning requires positive num_tasks and embedding_dim"
                )
            if embedding_dim != int(self.token_dim):
                raise ValueError(
                    "ChunkAwareDinoWMWorldModel task_conditioning.embedding_dim must match "
                    f"token_dim ({embedding_dim} != {int(self.token_dim)})"
                )
            self.task_embedding = nn.Embedding(num_tasks, int(self.token_dim))
        else:
            self.task_embedding = None
```

- [ ] **Step 4: Add helper to condition token tensors**

Add a method inside `ChunkAwareDinoWMWorldModel`:

```python
    def _apply_task_conditioning(
        self,
        obs_tokens: torch.Tensor,
        task_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.task_conditioning_enabled:
            return obs_tokens
        if task_ids is None:
            raise ValueError("task_ids are required when world model task conditioning is enabled")
        task_emb = self.task_embedding(task_ids.to(obs_tokens.device).long())
        while task_emb.ndim < obs_tokens.ndim:
            task_emb = task_emb.unsqueeze(1)
        return obs_tokens + task_emb.to(obs_tokens.dtype)
```

In `chunk_loss()`, insert the helper call immediately after the existing token conversion:

```python
        obs_tokens = self.obs_to_tokens(obs)
        obs_tokens = self._apply_task_conditioning(obs_tokens, batch.get("task_ids"))
```

Do not add the task embedding inside `predict_next_chunk()`: that method receives a latent history that was already built from conditioned tokens during training or from the selected rollout path, and adding the embedding there would apply it once per imagined chunk.

- [ ] **Step 5: Verify WM task conditioning**

Run:

```bash
pytest tests/unit_tests/test_task_conditioning.py::test_chunk_wm_declares_task_conditioning_support_when_enabled -q
```

Expected: PASS.

## Task 4: Validate Runner Fail-Fast and Default-Off Behavior

**Files:**
- Modify: `tests/unit_tests/test_task_conditioning.py`
- Verify: `dreamervla/runners/online_cotrain_runner.py:136`

- [ ] **Step 1: Add runner validation tests**

Append:

```python
def test_validate_task_conditioning_rejects_missing_capability():
    from omegaconf import OmegaConf
    from dreamervla.runners.online_cotrain_runner import validate_task_conditioning_cfg

    cfg = OmegaConf.create(
        {
            "task_conditioning": {
                "enabled": True,
                "num_tasks": 2,
                "embedding_dim": 4,
            }
        }
    )

    class NoSupport:
        supports_task_conditioning = False

    try:
        validate_task_conditioning_cfg(cfg, world_model=NoSupport(), classifier=NoSupport())
    except ValueError as exc:
        assert "lack task-conditioning support" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_task_conditioning_accepts_default_off():
    from omegaconf import OmegaConf
    from dreamervla.runners.online_cotrain_runner import validate_task_conditioning_cfg

    cfg = OmegaConf.create({"task_conditioning": {"enabled": False}})

    class NoSupport:
        supports_task_conditioning = False

    validate_task_conditioning_cfg(cfg, world_model=NoSupport(), classifier=NoSupport())
```

- [ ] **Step 2: Run validation tests**

Run:

```bash
pytest tests/unit_tests/test_task_conditioning.py::test_validate_task_conditioning_rejects_missing_capability tests/unit_tests/test_task_conditioning.py::test_validate_task_conditioning_accepts_default_off -q
```

Expected: PASS.

- [ ] **Step 3: Run task-conditioning regression suite**

Run:

```bash
pytest tests/unit_tests/test_task_conditioning.py tests/unit_tests/test_online_cotrain_pipeline.py::test_task_conditioned_classifier_receives_replay_task_ids -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Commit**

```bash
git add configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml dreamervla/models/reward/latent_success_classifier.py dreamervla/models/world_model/dino_wm_chunk.py dreamervla/runners/online_cotrain_runner.py tests/unit_tests/test_task_conditioning.py tests/unit_tests/test_online_cotrain_pipeline.py tests/unit_tests/test_online_cotrain_pipeline_config.py
git commit -s -m "feat(cotrain): add optional task conditioning"
```
