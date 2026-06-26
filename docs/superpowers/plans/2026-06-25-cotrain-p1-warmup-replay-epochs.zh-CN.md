# Cotrain P1 Warmup Replay Epochs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make offline warmup steps derive from replay coverage in the correct units: WM sampleable sequence windows for WM, classifier candidate windows for classifier.

**Architecture:** Keep one public `training.warmup_replay_epochs` knob and `training.warmup_replay_max_steps` cap. Resolve WM steps from `OnlineReplay.sampleable_window_count()` and classifier steps from `OnlineReplay.classifier_window_count(window, chunk_size)`, then execute WM/classifier updates in one alternating replay-warmup loop.

**Tech Stack:** Python 3.11, pytest, Hydra/OmegaConf, PyTorch optimizers, DreamerVLA `OnlineCotrainPipelineRunner`.

---

## File Structure

- Modify: `dreamervla/runners/online_cotrain_pipeline_runner.py:122`
  - Add classifier replay-epoch step resolution.
  - Preserve fixed-step compatibility when `warmup_replay_epochs == 0`.
  - Keep alternating WM/classifier warmup.
- Modify: `dreamervla/config.py:407`
  - Validate warmup epoch and max-step knobs are non-negative.
- Modify: `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml:49`
  - Keep `warmup_replay_epochs` and cap declared in Hydra.
- Modify: `tests/unit_tests/test_online_cotrain_pipeline.py:387`
  - Add unit tests for classifier-window-derived warmup steps and alternating logs.
- Modify: `tests/unit_tests/test_online_cotrain_pipeline_config.py:4`
  - Keep validation tests for negative warmup knobs.

## Task 1: Resolve Classifier Warmup Steps from Candidate Windows

**Files:**
- Modify: `dreamervla/runners/online_cotrain_pipeline_runner.py:122`
- Modify: `tests/unit_tests/test_online_cotrain_pipeline.py:387`

- [ ] **Step 1: Write the failing classifier replay-epoch test**

Add this test to `tests/unit_tests/test_online_cotrain_pipeline.py` near the existing warmup replay epoch tests:

```python
def test_warmup_replay_epochs_use_classifier_window_count_for_classifier(tmp_path):
    from dreamervla.runners.online_cotrain_pipeline_runner import OnlineCotrainPipelineRunner

    replay = _seeded_replay(tmp_path, seq_len=4)

    assert replay.sampleable_window_count() == 12
    assert replay.classifier_window_count(window=2, chunk_size=2) == 2
    assert OnlineCotrainPipelineRunner._resolve_warmup_steps(
        replay,
        wm_steps=1200,
        cls_steps=1200,
        replay_epochs=2,
        replay_max_steps=0,
        wm_batch_size=6,
        cls_batch_size=2,
        cls_window=2,
        cls_chunk_size=2,
    ) == (4, 2)
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest tests/unit_tests/test_online_cotrain_pipeline.py::test_warmup_replay_epochs_use_classifier_window_count_for_classifier -q
```

Expected before implementation: FAIL because `_resolve_warmup_steps()` does not accept `cls_window` or `cls_chunk_size`.

- [ ] **Step 3: Add classifier step calculation**

In `dreamervla/runners/online_cotrain_pipeline_runner.py`, add:

```python
    @staticmethod
    def _steps_for_classifier_replay_epochs(
        replay,
        *,
        replay_epochs: int,
        batch_size: int,
        window: int,
        chunk_size: int,
    ) -> int:
        epochs = int(replay_epochs)
        if epochs <= 0:
            return 0
        windows = int(
            replay.classifier_window_count(
                window=int(window),
                chunk_size=int(chunk_size),
            )
        )
        if windows <= 0:
            return 0
        return epochs * max(1, (windows + int(batch_size) - 1) // int(batch_size))
```

Change `_resolve_warmup_steps()` signature to:

```python
    def _resolve_warmup_steps(
        cls,
        replay,
        *,
        wm_steps: int,
        cls_steps: int,
        replay_epochs: int,
        replay_max_steps: int,
        wm_batch_size: int,
        cls_batch_size: int,
        cls_window: int,
        cls_chunk_size: int,
    ) -> tuple[int, int]:
```

Inside it, replace classifier resolution with:

```python
        resolved_cls = cls._steps_for_classifier_replay_epochs(
            replay,
            replay_epochs=epoch_count,
            batch_size=int(cls_batch_size),
            window=int(cls_window),
            chunk_size=int(cls_chunk_size),
        )
```

- [ ] **Step 4: Pass classifier config from `run()`**

In `OnlineCotrainPipelineRunner.run()`, before `_resolve_warmup_steps()`, read:

```python
            cls_window = int(getattr(_unwrap(self.classifier).cfg, "window", self._cls_window))
            cls_chunk_size = int(
                getattr(_unwrap(self.classifier).cfg, "chunk_size", 1)
            )
```

Update the call:

```python
                cls_window=cls_window,
                cls_chunk_size=cls_chunk_size,
```

- [ ] **Step 5: Update existing warmup tests for the new signature**

In existing `_resolve_warmup_steps()` tests, pass:

```python
        cls_window=2,
        cls_chunk_size=2,
```

- [ ] **Step 6: Verify warmup step resolution**

Run:

```bash
pytest tests/unit_tests/test_online_cotrain_pipeline.py::test_warmup_replay_epochs_derive_steps_from_sampleable_windows tests/unit_tests/test_online_cotrain_pipeline.py::test_warmup_replay_epochs_cap_to_configured_budget tests/unit_tests/test_online_cotrain_pipeline.py::test_warmup_replay_epochs_use_classifier_window_count_for_classifier -q
```

Expected: all selected tests pass.

## Task 2: Keep Alternating WM/Classifier Warmup Observable

**Files:**
- Modify: `dreamervla/runners/online_cotrain_pipeline_runner.py:163`
- Modify: `tests/unit_tests/test_online_cotrain_pipeline.py:340`

- [ ] **Step 1: Write an alternating-order test**

Add this test to `tests/unit_tests/test_online_cotrain_pipeline.py` near `test_offline_warmup_steps_update_modules`:

```python
def test_offline_warmup_alternating_interleaves_wm_and_classifier(tmp_path, monkeypatch):
    import torch
    import dreamervla.runners.online_cotrain_pipeline_runner as mod

    replay = _seeded_replay(tmp_path)
    calls = []

    def fake_wm_step(**kw):
        assert kw["batch"] is not None
        calls.append("wm")
        return {"loss": float(len(calls))}

    def fake_cls_step(**kw):
        assert kw["replay"] is replay
        calls.append("cls")
        return {"loss": 0.2, "acc": 0.5, "f1": 0.25, "pos_frac": 0.5}

    monkeypatch.setattr(mod, "world_model_pretrain_step", fake_wm_step)
    monkeypatch.setattr(mod, "online_classifier_update_step", fake_cls_step)

    runner = mod.OnlineCotrainPipelineRunner.__new__(mod.OnlineCotrainPipelineRunner)
    runner.device = torch.device("cpu")
    runner._build_wm_pretrain_batch = lambda b: {
        "images": torch.zeros(1),
        "obs_embedding": torch.zeros(1),
        "actions": torch.zeros(1),
    }
    runner.world_model = torch.nn.Module()
    runner.world_model_optimizer = object()
    runner.policy = object()
    runner.classifier = torch.nn.Module()
    runner.classifier_optimizer = object()
    runner._log_replay_warmup_metrics = lambda metrics, step: None

    wm_last, cls_last = runner._offline_warmup_alternating(
        replay,
        wm_steps=2,
        cls_steps=3,
        wm_batch_size=2,
        cls_batch_size=2,
        optim_cfg=None,
        early_neg_stride=8,
        grad_clip=1.0,
    )

    assert calls == ["wm", "cls", "wm", "cls", "cls"]
    assert wm_last == 3.0
    assert cls_last == 0.5
```

- [ ] **Step 2: Run the alternating-order test**

Run:

```bash
pytest tests/unit_tests/test_online_cotrain_pipeline.py::test_offline_warmup_alternating_interleaves_wm_and_classifier -q
```

Expected: PASS if the alternating loop is already present; FAIL if warmup runs all WM then all classifier.

- [ ] **Step 3: Ensure alternating loop emits classifier detail metrics**

In `_offline_warmup_alternating()`, keep local variables for classifier metrics:

```python
            cls_loss = 0.0
            cls_f1 = 0.0
            cls_pos_frac = 0.0
```

After classifier update, set:

```python
                cls_loss = float(cls_metrics.get("loss", 0.0))
                cls_last = float(cls_metrics["acc"])
                cls_f1 = float(cls_metrics.get("f1", 0.0))
                cls_pos_frac = float(cls_metrics.get("pos_frac", 0.0))
```

Log:

```python
                    {
                        "train/wm_warmup_loss": wm_last,
                        "train/classifier_warmup_loss": cls_loss,
                        "train/classifier_warmup_acc": cls_last,
                        "train/classifier_warmup_f1": cls_f1,
                        "train/classifier_warmup_pos_frac": cls_pos_frac,
                    },
```

- [ ] **Step 4: Run warmup loop tests**

Run:

```bash
pytest tests/unit_tests/test_online_cotrain_pipeline.py::test_offline_warmup_steps_update_modules tests/unit_tests/test_online_cotrain_pipeline.py::test_offline_warmup_alternating_interleaves_wm_and_classifier -q
```

Expected: selected tests pass.

## Task 3: Validate Warmup Config Is Declared, Not Chosen in Code

**Files:**
- Modify: `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml:49`
- Modify: `dreamervla/config.py:407`
- Modify: `tests/unit_tests/test_online_cotrain_pipeline_config.py:4`

- [ ] **Step 1: Keep explicit Hydra warmup knobs**

Verify `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml` includes:

```yaml
training:
  wm_warmup_steps: 1200
  classifier_warmup_steps: 1200
  warmup_replay_epochs: 1
  warmup_replay_max_steps: 1200
```

- [ ] **Step 2: Keep validation for non-negative values**

In `dreamervla/config.py`, `_validate_online_cotrain_pipeline()` should loop over:

```python
    for key in (
        "wm_warmup_steps",
        "classifier_warmup_steps",
        "warmup_replay_epochs",
        "warmup_replay_max_steps",
    ):
        val = int(OmegaConf.select(cfg, f"training.{key}", default=0))
        if val < 0:
            raise ValueError(f"training.{key} must be >= 0, got {val}")
```

- [ ] **Step 3: Run config validation tests**

Run:

```bash
pytest tests/unit_tests/test_online_cotrain_pipeline_config.py::test_validate_cfg_warmup -q
```

Expected: PASS.

## Task 4: Run P1 Warmup Regression Suite

**Files:**
- Verify: all files modified in this plan.

- [ ] **Step 1: Run focused warmup tests**

Run:

```bash
pytest tests/unit_tests/test_online_cotrain_pipeline.py tests/unit_tests/test_online_cotrain_pipeline_config.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run a GPU-gated smoke only on a GPU/LIBERO host**

Run only when the OFT checkpoint, sidecar, LIBERO assets, and a free GPU are available:

```bash
python -m dreamervla.train experiment=online_cotrain_pipeline_oft_action_hidden_smoke task=openvla_onetraj_coldstart_libero
```

Expected: run reaches `[1/3] REPLAY WARMUP`, saves `ckpt/wm_warmup.ckpt` and `ckpt/classifier_warmup.ckpt`, then enters `[3/3] ONLINE COTRAIN`.

- [ ] **Step 3: Commit**

```bash
git add dreamervla/runners/online_cotrain_pipeline_runner.py dreamervla/config.py configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml tests/unit_tests/test_online_cotrain_pipeline.py tests/unit_tests/test_online_cotrain_pipeline_config.py
git commit -s -m "feat(cotrain): derive warmup from replay epochs"
```
