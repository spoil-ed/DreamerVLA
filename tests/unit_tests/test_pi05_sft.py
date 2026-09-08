from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamervla.config import validate_cfg
from dreamervla.models.embodiment.pi05.policy import (
    _freeze_unused_continuous_action_parameters,
)
from dreamervla.models.embodiment.pi05.sft_data import (
    OFFICIAL_PI05_LIBERO_REPO,
    LeRobotLIBERODataLoaderFactory,
    configure_openpi_pytorch_runtime,
    resolve_lerobot_source,
)
from dreamervla.runners.vla_sft_training_runner import (
    VLASFTTrainingRunner,
    _build_lr_scheduler,
    _strip_legacy_unused_lm_head_optimizer_state,
)
from dreamervla.train import _auto_apply_distributed
from dreamervla.utils.integrations.openpi_imports import configure_openpi_jax_runtime

_LOCAL_EXPERIMENT_NAMES = {
    "pi05_libero_sft_one_episode_per_task",
    "pi05_libero_sft_five_episodes_per_task",
}


def _experiment_overrides(experiment: str) -> list[str]:
    overrides = [f"experiment={experiment}"]
    if experiment not in _LOCAL_EXPERIMENT_NAMES:
        return overrides
    root = Path(__file__).resolve().parents[2]
    local_config_root = root / "experiments" / "configs"
    config_path = local_config_root / "experiment" / f"{experiment}.yaml"
    if not config_path.is_file():
        pytest.skip(f"local ignored experiment config is absent: {config_path}")
    return [f"hydra.searchpath=[file://{local_config_root}]", *overrides]


def test_pi05_pytorch_loader_keeps_jax_off_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    monkeypatch.delitem(sys.modules, "jax", raising=False)

    configure_openpi_pytorch_runtime()

    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_pi05_pytorch_loader_rejects_jax_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")

    with pytest.raises(RuntimeError, match="requires JAX_PLATFORMS=cpu"):
        configure_openpi_pytorch_runtime()


def test_openpi_runtime_rejects_jax_imported_before_backend_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setitem(sys.modules, "jax", object())

    with pytest.raises(RuntimeError, match="JAX was imported before"):
        configure_openpi_jax_runtime()


def test_pi05_sft_experiment_composes_migrated_rlinf_fsdp_recipe() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=pi05_libero_sft"])
    OmegaConf.resolve(cfg)
    assert cfg._target_ == "dreamervla.runners.VLASFTTrainingRunner"
    assert cfg.data.loader._target_ == "dreamervla.dataset.libero.LeRobotV3LIBERODataLoaderFactory"
    assert cfg.data.loader.dataset.dataset_dir == cfg.data.source
    assert cfg.data.source == "/jfs/public/prod/hf-datasets/datasets/lerobot/libero"
    assert cfg.data.repo_id == "lerobot/libero"
    assert cfg.data.format == "lerobot_v3"
    assert cfg.data.loader.normalization_asset_id == "physical-intelligence/libero"
    assert cfg.data.loader.action_horizon == 10
    assert cfg.data.loader.dataset.sequence_length == 10
    assert cfg.actor.micro_batch_size == 4
    assert cfg.actor.global_batch_size == 128
    assert cfg.actor.use_action_chunk_loss is False
    assert cfg.actor.policy_cfg.kwargs.add_value_head is False
    assert cfg.actor.policy_cfg.kwargs.batch_size == cfg.actor.global_batch_size
    assert cfg.actor.policy_cfg.kwargs.learning_rate == cfg.actor.optim.lr
    assert cfg.actor.policy_cfg.kwargs.total_training_steps == cfg.actor.optim.total_training_steps
    assert cfg.actor.policy_cfg.kwargs.model_path == cfg.task.pi05.base_ckpt_path
    assert cfg.actor.policy_cfg.kwargs.assets_path == cfg.task.pi05.ckpt_path
    assert cfg.task.pi05.base_ckpt_path.endswith("data/checkpoints/pi05_base")
    assert cfg.task.pi05.ckpt_path.endswith("data/checkpoints/RLinf-Pi05-LIBERO-SFT")
    assert cfg.task.pi05.base_ckpt_path != cfg.task.pi05.ckpt_path
    assert cfg.actor.distributed.strategy == "fsdp"
    assert cfg.training.distributed_strategy == "fsdp"
    assert cfg.actor.optim.lr_scheduler == "constant"
    assert cfg.actor.optim.lr == pytest.approx(2.5e-5)
    assert cfg.actor.optim.adam_beta1 == pytest.approx(0.9)
    assert cfg.actor.optim.adam_beta2 == pytest.approx(0.95)
    assert cfg.actor.optim.adam_eps == pytest.approx(1.0e-8)
    assert cfg.actor.optim.weight_decay == pytest.approx(1.0e-10)
    assert cfg.actor.optim.clip_grad == pytest.approx(1.0)
    assert cfg.actor.optim.lr_warmup_steps == 1000
    assert cfg.actor.optim.total_training_steps == 30000
    assert cfg.training.max_steps == 30000
    assert cfg.training.checkpoint_every == 2000
    validate_cfg(cfg, world_size=2)


@pytest.mark.parametrize(
    (
        "experiment",
        "run_name",
        "learning_rate",
        "micro_batch_size",
        "accumulation",
        "training_steps",
        "action_horizon",
        "world_size",
    ),
    [
        (
            "pi05_libero_sft_full",
            "pi05_libero_sft_full",
            2.5e-5,
            1,
            32,
            30000,
            10,
            8,
        ),
        (
            "pi05_libero_sft_one_episode_per_task",
            "pi05_libero_sft_one_episode_per_task",
            5.0e-6,
            32,
            1,
            30000,
            50,
            8,
        ),
        (
            "pi05_libero_sft_five_episodes_per_task",
            "pi05_libero_sft_five_episodes_per_task",
            5.0e-6,
            32,
            1,
            30000,
            50,
            8,
        ),
    ],
)
def test_pi05_sft_eight_gpu_recipes_use_global_batch_256(
    experiment: str,
    run_name: str,
    learning_rate: float,
    micro_batch_size: int,
    accumulation: int,
    training_steps: int,
    action_horizon: int,
    world_size: int,
) -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train", overrides=_experiment_overrides(experiment))
    OmegaConf.resolve(cfg)

    assert cfg.run.name == run_name
    assert cfg.data.loader.batch_size == micro_batch_size
    assert cfg.actor.micro_batch_size == micro_batch_size
    assert cfg.actor.global_batch_size == 256
    assert cfg.actor.global_batch_size // (micro_batch_size * world_size) == accumulation
    assert cfg.actor.optim.lr == pytest.approx(learning_rate)
    assert cfg.actor.optim.total_training_steps == training_steps
    assert cfg.training.max_steps == training_steps
    assert cfg.task.pi05.action_horizon == action_horizon
    assert cfg.task.action_horizon == action_horizon
    assert cfg.data.loader.action_horizon == action_horizon
    assert cfg.actor.policy_cfg.kwargs.action_chunk == action_horizon
    assert cfg.actor.distributed.strategy == "ddp"
    assert cfg.actor.distributed.backend == "nccl"
    assert cfg.actor.distributed.init_sync is False
    assert cfg.actor.policy_cfg.kwargs.batch_size == 256
    assert cfg.actor.distributed.find_unused_parameters is False
    assert cfg.training.distributed_strategy == "ddp"
    validate_cfg(cfg, world_size=world_size)


def test_pi05_five_episode_recipe_uses_balanced_subset_and_milestones() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=_experiment_overrides("pi05_libero_sft_five_episodes_per_task"),
        )
    OmegaConf.resolve(cfg)

    assert "pi05_libero_five_episodes_per_task" in cfg.data.source
    assert cfg.data.loader.dataset.dataset_dir == cfg.data.source
    assert list(cfg.training.milestone_steps) == [2000, 5000, 10000, 20000, 30000]


def test_pi05_one_trajectory_recipe_retains_eval_milestones() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=_experiment_overrides("pi05_libero_sft_one_episode_per_task"),
        )
    OmegaConf.resolve(cfg)
    runner = object.__new__(VLASFTTrainingRunner)
    runner.cfg = cfg
    runner.config = cfg
    runner._output_dir = None

    assert list(cfg.training.milestone_steps) == [
        2000,
        5000,
        10000,
        20000,
        30000,
    ]
    assert runner._milestone_checkpoint_paths(4000) == ()
    assert runner._milestone_checkpoint_paths(5000) == (
        Path(cfg.training.out_dir).resolve() / "checkpoints" / "step=5000.ckpt",
    )


def test_pi05_sft_rejects_invalid_milestone_steps() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=pi05_libero_sft"])
    cfg.training.milestone_steps = [2000, 2000]

    with pytest.raises(ValueError, match="sorted and unique"):
        validate_cfg(cfg, world_size=8)


def test_torchrun_auto_distribution_preserves_explicit_pi05_fsdp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=pi05_libero_sft"])
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")

    _auto_apply_distributed(cfg)

    assert cfg.training.distributed_strategy == "fsdp"


def test_pi05_sft_rejects_v2_format_for_local_v3_reader() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=pi05_libero_sft"])
    cfg.data.format = "lerobot_v2"

    with pytest.raises(ValueError, match="data.format=lerobot_v3"):
        validate_cfg(cfg, world_size=2)


def test_pi05_sft_rejects_loader_normalization_asset_mismatch() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=pi05_libero_sft"])
    cfg.data.loader.normalization_asset_id = "lerobot/libero"

    with pytest.raises(ValueError, match="normalization_asset_id must match"):
        validate_cfg(cfg, world_size=2)


def test_pi05_checkpoint_paths_accept_submit_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PI05_BASE_CKPT", "/runtime/models/pi05-base")
    monkeypatch.setenv("PI05_LIBERO_CKPT", "/runtime/models/pi05-libero-sft")
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=pi05_libero_sft"])
    OmegaConf.resolve(cfg)
    assert cfg.task.pi05.base_ckpt_path == "/runtime/models/pi05-base"
    assert cfg.task.pi05.ckpt_path == "/runtime/models/pi05-libero-sft"
    assert cfg.task.pi05.assets_path == "/runtime/models/pi05-libero-sft"


def test_resolve_pi05_lerobot_source_accepts_repo_and_complete_local_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert resolve_lerobot_source(OFFICIAL_PI05_LIBERO_REPO) == OFFICIAL_PI05_LIBERO_REPO
    root = tmp_path / "physical-intelligence" / "libero"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "image": {},
                    "wrist_image": {},
                    "state": {},
                    "actions": {},
                }
            }
        ),
        encoding="utf-8",
    )
    assert resolve_lerobot_source(root) == str(root.resolve())
    monkeypatch.delenv("HF_LEROBOT_HOME", raising=False)
    factory = LeRobotLIBERODataLoaderFactory(
        source=root,
        model_path="/models/pi05-base",
        assets_path="/models/pi05-libero",
    )
    assert factory.source == str(root.resolve())


def test_pi05_lerobot_factory_accepts_official_repo_id() -> None:
    factory = LeRobotLIBERODataLoaderFactory(
        source=OFFICIAL_PI05_LIBERO_REPO,
        model_path="/models/pi05-base",
        assets_path="/models/pi05-libero",
    )
    assert factory.source == OFFICIAL_PI05_LIBERO_REPO


def test_pi05_lerobot_factory_rejects_wrong_local_snapshot_revision(
    tmp_path: Path,
) -> None:
    root = tmp_path / "physical-intelligence" / "libero"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "image": {},
                    "wrist_image": {},
                    "state": {},
                    "actions": {},
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="revision"):
        LeRobotLIBERODataLoaderFactory(
            source=root,
            model_path="/models/pi05-base",
            assets_path="/models/pi05-libero",
            revision="wrong-commit",
        )


def test_resolve_pi05_lerobot_source_rejects_incomplete_local_root(tmp_path: Path) -> None:
    root = tmp_path / "libero"
    root.mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        resolve_lerobot_source(root)


def test_pi05_policy_freezes_unused_lm_head() -> None:
    import torch

    lm_head = torch.nn.Linear(4, 3)
    model = SimpleNamespace(
        paligemma_with_expert=SimpleNamespace(gemma_expert=SimpleNamespace(lm_head=lm_head))
    )

    frozen = _freeze_unused_continuous_action_parameters(model)

    assert frozen == sum(parameter.numel() for parameter in lm_head.parameters())
    assert not any(parameter.requires_grad for parameter in lm_head.parameters())


def test_pi05_sft_tracks_epochs_around_openpi_infinite_wrapper() -> None:
    import torch

    class Sampler:
        def __init__(self) -> None:
            self.epochs: list[int] = []

        def set_epoch(self, epoch: int) -> None:
            self.epochs.append(epoch)

    class InnerLoader:
        def __init__(self) -> None:
            self.generator = torch.Generator().manual_seed(7)
            self.sampler = Sampler()

        def __len__(self) -> int:
            return 2

    class OfficialLoader:
        def __init__(self, inner: InnerLoader) -> None:
            self._data_loader = SimpleNamespace(_data_loader=inner)

        def __iter__(self):
            return iter(("batch-0", "batch-1"))

    inner = InnerLoader()
    runner = object.__new__(VLASFTTrainingRunner)
    runner.data_loader = OfficialLoader(inner)
    runner.data_iterator = iter(("stale",))
    runner._data_epoch = 0
    runner._data_iter_offset = 2
    runner._data_generator_state = None
    runner.epoch = 0

    batch = runner._next_batch()

    assert batch == "batch-0"
    assert runner._data_epoch == 1
    assert runner._data_iter_offset == 1
    assert runner._data_generator_state is not None
    assert inner.sampler.epochs == [1]


def test_pi05_sft_synchronizes_only_the_first_loaded_batch() -> None:
    class DistributedStub:
        def __init__(self) -> None:
            self.barrier_calls = 0

        def object_barrier(self) -> None:
            self.barrier_calls += 1

    runner = object.__new__(VLASFTTrainingRunner)
    runner.distributed = DistributedStub()
    runner._first_batch_synchronized = False

    runner._synchronize_first_batch()
    runner._synchronize_first_batch()

    assert runner.distributed.barrier_calls == 1
    assert runner._first_batch_synchronized is True


def test_pi05_sft_checkpoint_uses_canonical_trainable_policy_keys(tmp_path: Path) -> None:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    class DistributedStub:
        is_main_process = True

        @staticmethod
        def unwrap_module(module: DDP) -> torch.nn.Module:
            return module.module

    if dist.is_initialized():
        pytest.skip("test requires ownership of the default process group")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'ddp_init'}",
        rank=0,
        world_size=1,
    )
    try:
        source = torch.nn.Linear(3, 2)
        source.bias.requires_grad_(False)
        wrapped_source = DDP(source)
        runner = object.__new__(VLASFTTrainingRunner)
        runner.distributed = DistributedStub()

        policy_state = runner._state_dict_for_checkpoint("policy", wrapped_source)

        assert policy_state is not None
        assert set(policy_state) == {"weight"}

        target = torch.nn.Linear(3, 2)
        target.bias.requires_grad_(False)
        target.weight.data.zero_()
        original_bias = target.bias.detach().clone()
        policy_state["model.paligemma_with_expert.gemma_expert.lm_head.weight"] = torch.zeros(1)
        original_policy_keys = set(policy_state)
        runner._load_state_dict_from_checkpoint(
            "policy",
            DDP(target),
            policy_state,
        )
        assert set(policy_state) == original_policy_keys
        assert torch.equal(target.weight, source.weight)
        assert torch.equal(target.bias, original_bias)
    finally:
        dist.destroy_process_group()


def test_pi05_sft_non_main_rank_accepts_empty_full_policy_state(monkeypatch) -> None:
    import torch

    class DistributedStub:
        is_main_process = False

        @staticmethod
        def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
            return module

    runner = object.__new__(VLASFTTrainingRunner)
    runner.distributed = DistributedStub()
    policy = torch.nn.Linear(3, 2)
    monkeypatch.setattr(
        "dreamervla.runners.vla_sft_training_runner.get_model_state_dict",
        lambda *_args, **_kwargs: {},
    )

    assert runner._state_dict_for_checkpoint("policy", policy) == {}


def test_pi05_sft_checkpoint_uses_canonical_optimizer_keys() -> None:
    import torch

    class DistributedStub:
        is_main_process = True

        @staticmethod
        def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
            return module

        @staticmethod
        def load_optimizer_state_dict(*_args, **_kwargs) -> None:
            raise AssertionError("canonical optimizer state must use the distributed API")

    source = torch.nn.Linear(3, 2)
    source.bias.requires_grad_(False)
    source_optimizer = torch.optim.AdamW((source.weight,), lr=1.0e-3)
    source(torch.ones(1, 3)).sum().backward()
    source_optimizer.step()
    runner = object.__new__(VLASFTTrainingRunner)
    runner.policy = source
    runner.distributed = DistributedStub()

    optimizer_state = runner._state_dict_for_checkpoint("policy_optimizer", source_optimizer)

    assert optimizer_state is not None
    assert set(optimizer_state["state"]) == {"weight"}
    original_state_keys = set(optimizer_state["state"])
    target = torch.nn.Linear(3, 2)
    target.bias.requires_grad_(False)
    target_optimizer = torch.optim.AdamW((target.weight,), lr=1.0e-3)
    runner.policy = target
    runner._load_state_dict_from_checkpoint(
        "policy_optimizer",
        target_optimizer,
        optimizer_state,
    )
    assert set(optimizer_state["state"]) == original_state_keys
    assert len(target_optimizer.state) == 1


def test_pi05_sft_checkpoint_loads_legacy_ddp_optimizer_ids() -> None:
    import torch

    class DistributedStub:
        is_main_process = True

        def __init__(self) -> None:
            self.used_legacy_loader = False

        @staticmethod
        def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
            return module

        def load_optimizer_state_dict(
            self,
            _module: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            state_dict: dict,
        ) -> None:
            self.used_legacy_loader = True
            optimizer.load_state_dict(state_dict)

    source = torch.nn.Linear(3, 2)
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=1.0e-3)
    source(torch.ones(1, 3)).sum().backward()
    source_optimizer.step()
    legacy_state = source_optimizer.state_dict()
    assert all(isinstance(parameter_id, int) for parameter_id in legacy_state["state"])
    target = torch.nn.Linear(3, 2)
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1.0e-3)
    runner = object.__new__(VLASFTTrainingRunner)
    runner.policy = target
    distributed = DistributedStub()
    runner.distributed = distributed

    runner._load_state_dict_from_checkpoint(
        "policy_optimizer",
        target_optimizer,
        legacy_state,
    )

    assert distributed.used_legacy_loader is True
    assert len(target_optimizer.state) == 2


def test_pi05_sft_strips_legacy_unused_lm_head_optimizer_names() -> None:
    legacy_name = "model.paligemma_with_expert.gemma_expert.lm_head.weight"
    action_name = "model.action_out_proj.weight"
    state = {
        "state": {legacy_name: {"step": 1}, action_name: {"step": 1}},
        "param_groups": [{"params": [legacy_name, action_name]}],
    }

    cleaned = _strip_legacy_unused_lm_head_optimizer_state(state)

    assert set(cleaned["state"]) == {action_name}
    assert cleaned["param_groups"][0]["params"] == [action_name]
    assert legacy_name in state["state"]


def test_pi05_sft_scheduler_matches_rlinf_effective_constant_warmup() -> None:
    import torch

    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=2.5e-5)
    cfg = OmegaConf.create(
        {"lr_scheduler": "constant", "lr_warmup_steps": 10, "total_training_steps": 30}
    )
    scheduler = _build_lr_scheduler(optimizer, cfg)
    assert optimizer.param_groups[0]["lr"] == 0.0
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-5)
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-5)
