# World Model File Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split monolithic world-model implementations into snake_case files while keeping old imports working.

**Architecture:** Move shared world-model adapter types into `base_world_model.py`. Move each WorldModel class body into its own snake_case module under `dreamer_vla/models/world_model/`. Keep `dreamerv3_torch.py` and `tssm_torch.py` as compatibility aggregation modules for existing Hydra targets and scripts.

**Tech Stack:** Python, PyTorch, Hydra import targets, pytest.

---

### Task 1: Import Surface Test

**Files:**
- Create: `tests/test_world_model_file_split_imports.py`
- Modify: none

- [ ] **Step 1: Write the failing test**

```python
from dreamer_vla.models.world_model.base_world_model import BaseWorldModel, DreamerV3LatentState, DreamerV3Loss
from dreamer_vla.models.world_model.dreamer_v3_pixel_rynn_backbone_world_model import DreamerV3PixelRynnBackboneWorldModel
from dreamer_vla.models.world_model.dreamer_v3_pixel_world_model import DreamerV3PixelWorldModel
from dreamer_vla.models.world_model.dreamer_v3_token_from_pixel_world_model import DreamerV3TokenFromPixelWorldModel
from dreamer_vla.models.world_model.dreamer_v3_token_world_model import DreamerV3TokenWorldModel
from dreamer_vla.models.world_model.tssm_rynn_backbone_world_model import TSSMRynnBackboneWorldModel
from dreamer_vla.models.world_model.tssm_token_rynn_backbone_world_model import TSSMTokenRynnBackboneWorldModel
from dreamer_vla.models.world_model import dreamerv3_torch, tssm_torch


def test_split_world_model_modules_export_classes() -> None:
    assert issubclass(DreamerV3PixelWorldModel, BaseWorldModel)
    assert issubclass(DreamerV3TokenWorldModel, BaseWorldModel)
    assert issubclass(DreamerV3TokenFromPixelWorldModel, BaseWorldModel)
    assert issubclass(DreamerV3PixelRynnBackboneWorldModel, BaseWorldModel)
    assert issubclass(TSSMRynnBackboneWorldModel, BaseWorldModel)
    assert issubclass(TSSMTokenRynnBackboneWorldModel, BaseWorldModel)
    assert DreamerV3LatentState.__name__ == "DreamerV3LatentState"
    assert DreamerV3Loss.__name__ == "DreamerV3Loss"


def test_legacy_modules_reexport_split_world_model_classes() -> None:
    assert dreamerv3_torch.DreamerV3PixelWorldModel is DreamerV3PixelWorldModel
    assert dreamerv3_torch.DreamerV3TokenWorldModel is DreamerV3TokenWorldModel
    assert dreamerv3_torch.DreamerV3TokenFromPixelWorldModel is DreamerV3TokenFromPixelWorldModel
    assert dreamerv3_torch.DreamerV3PixelRynnBackboneWorldModel is DreamerV3PixelRynnBackboneWorldModel
    assert tssm_torch.TSSMRynnBackboneWorldModel is TSSMRynnBackboneWorldModel
    assert tssm_torch.TSSMTokenRynnBackboneWorldModel is TSSMTokenRynnBackboneWorldModel
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_world_model_file_split_imports.py -q`
Expected: FAIL with `ModuleNotFoundError` for the new split modules.

### Task 2: Split Production Modules

**Files:**
- Create: `dreamer_vla/models/world_model/base_world_model.py`
- Create: `dreamer_vla/models/world_model/dreamer_v3_pixel_world_model.py`
- Create: `dreamer_vla/models/world_model/dreamer_v3_token_world_model.py`
- Create: `dreamer_vla/models/world_model/dreamer_v3_token_from_pixel_world_model.py`
- Create: `dreamer_vla/models/world_model/dreamer_v3_pixel_rynn_backbone_world_model.py`
- Create: `dreamer_vla/models/world_model/tssm_rynn_backbone_world_model.py`
- Create: `dreamer_vla/models/world_model/tssm_token_rynn_backbone_world_model.py`
- Modify: `dreamer_vla/models/world_model/dreamerv3_torch.py`
- Modify: `dreamer_vla/models/world_model/tssm_torch.py`
- Modify: `dreamer_vla/models/world_model/__init__.py`

- [ ] **Step 1: Move shared adapter types**

Move `DreamerV3Loss`, `DreamerV3LatentState`, and `DreamerV3ActorAdapterMixin` to `base_world_model.py`; add `BaseWorldModel`.

- [ ] **Step 2: Move each WorldModel class**

Move each class body into its snake_case module. Leave non-WorldModel building blocks in family modules.

- [ ] **Step 3: Keep old import targets**

Import the split classes back into `dreamerv3_torch.py` and `tssm_torch.py` so old Hydra configs and scripts keep working.

### Task 3: Verification

**Files:**
- Test: `tests/test_world_model_file_split_imports.py`
- Test: existing world-model tests.

- [ ] **Step 1: Run split import test**

Run: `PYTHONPATH=. pytest tests/test_world_model_file_split_imports.py -q`
Expected: PASS.

- [ ] **Step 2: Run focused regression tests**

Run: `PYTHONPATH=. pytest tests/test_dreamerv3_online_observe.py tests/test_preprocess_rynn_pixel_hidden.py tests/test_reward_head.py tests/test_tssm_transdreamer_compat.py -q`
Expected: PASS.
