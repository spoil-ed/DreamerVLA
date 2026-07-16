from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import IterableDataset, SequentialSampler

from dreamervla.algorithms.critic.latent_success_classifier import (
    LatentSuccessClassifier,
    LatentSuccessClassifierConfig,
)
from dreamervla.dataset.lumos_aligned_latent_dataset import (
    LumosAlignedLatentTrainDataset,
    LumosAlignedLatentValDataset,
    _DemoRecord,
    _partition_demo_pairs,
)
from dreamervla.runners.success_classifier_training_runner import (
    SuccessClassifierTrainingRunner,
    _classifier_loss_and_predictions,
    _success_probabilities_from_logits,
)
from dreamervla.runtime.classifier_metrics import sweep_threshold_metrics


def test_classifier_checkpoint_includes_all_resume_loop_state() -> None:
    assert {
        "global_step",
        "epoch",
        "best_window_f1",
        "best_episode_f1",
        "best_window_ckpt_path",
        "best_episode_ckpt_path",
    }.issubset(set(SuccessClassifierTrainingRunner.include_keys))


def test_classifier_strict_resume_rejects_missing_optimizer() -> None:
    import pytest

    runner = object.__new__(SuccessClassifierTrainingRunner)
    runner.model = torch.nn.Linear(2, 2)
    runner.optim = torch.optim.SGD(runner.model.parameters(), lr=0.1)
    runner.distributed = _FakeDistributed()

    with pytest.raises(RuntimeError, match="optim"):
        runner.load_payload(
            {
                "format_version": 2,
                "state_dicts": {"model": runner.model.state_dict()},
                "pickles": {},
                "rng_by_rank": [],
            },
            restore_rng=True,
        )

    with pytest.raises(RuntimeError, match="global_step/epoch"):
        runner.load_payload(
            {
                "format_version": 2,
                "state_dicts": {
                    "model": runner.model.state_dict(),
                    "optim": runner.optim.state_dict(),
                },
                "pickles": {},
                "rng_by_rank": [],
            },
            restore_rng=True,
        )

    with pytest.raises(ValueError, match="format_version=3"):
        runner.load_payload(
            {
                "format_version": 3,
                "state_dicts": {
                    "model": runner.model.state_dict(),
                    "optim": runner.optim.state_dict(),
                },
                "pickles": {"global_step": b"x", "epoch": b"x"},
                "rng_by_rank": [],
            },
            restore_rng=True,
        )


class _FixedLogitClassifier(torch.nn.Module):
    supports_language_conditioning = False
    supports_proprio_conditioning = False
    supports_task_conditioning = False

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "_logits",
            torch.tensor(
                [
                    [10.0, 9.0],
                    [0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, xs: torch.Tensor, **_: object) -> torch.Tensor:
        return self._logits[: xs.shape[0]].to(xs.device)


def test_window_eval_uses_softmax_positive_probability() -> None:
    runner = object.__new__(SuccessClassifierTrainingRunner)
    runner.model = _FixedLogitClassifier()
    runner.device = torch.device("cpu")
    runner.val_loader = [
        (
            torch.zeros(2, 1, 1),
            torch.tensor([0, 1], dtype=torch.long),
            {},
        )
    ]
    runner.cfg = OmegaConf.create(
        {"training": {"thresh_min": 0.5, "thresh_max": 0.5, "thresh_steps": 1}}
    )

    metrics = runner._evaluate_window_level()

    assert metrics["best_f1"] == 1.0
    assert metrics["best_thresh"] == 0.5


class _EpisodeDataset:
    K = 1
    chunk_pool = "last"

    def trajectories(self):
        return [
            (torch.zeros(1, 1).numpy(), False, 1, "negative"),
            (torch.zeros(1, 1).numpy(), True, 1, "positive"),
        ]


def test_episode_eval_uses_softmax_positive_probability() -> None:
    runner = object.__new__(SuccessClassifierTrainingRunner)
    runner.model = _FixedLogitClassifier()
    runner.device = torch.device("cpu")
    runner.val_ds = _EpisodeDataset()
    runner.cfg = OmegaConf.create(
        {
            "data": {"window": 1},
            "training": {
                "episode_eval_batch": 8,
                "episode_eval_min_steps": 0,
                "episode_eval_stride": 1,
                "thresh_min": 0.5,
                "thresh_max": 0.5,
                "thresh_steps": 1,
            },
        }
    )

    metrics = runner._evaluate_episode_level()

    assert metrics["best_f1"] == 1.0
    assert metrics["best_thresh"] == 0.5


def test_threshold_sweep_reports_confusion_counts() -> None:
    probs = np.asarray([0.9, 0.8, 0.4, 0.1], dtype=np.float32)
    ys = np.asarray([1, 0, 1, 0], dtype=np.int64)

    metrics = sweep_threshold_metrics(
        probs,
        ys,
        np.asarray([0.5], dtype=np.float32),
        tag="window",
    )

    row = metrics["per_thresh"]["th_0.50"]
    assert row["tp"] == 1
    assert row["tn"] == 1
    assert row["fp"] == 1
    assert row["fn"] == 1
    assert row["pred_pos"] == 2
    assert row["pred_neg"] == 2
    assert row["true_pos"] == 2
    assert row["true_neg"] == 2


def test_runner_dataset_summary_payload_handles_train_and_val() -> None:
    runner = object.__new__(SuccessClassifierTrainingRunner)

    class _Dataset:
        def summary(self) -> dict[str, int | str]:
            return {"num_demos": 3, "num_success_demos": 2, "num_failure_demos": 1}

    payload = runner._dataset_summary_payload("train", _Dataset())

    assert payload == {
        "event": "dataset_summary",
        "split": "train",
        "num_demos": 3,
        "num_success_demos": 2,
        "num_failure_demos": 1,
    }


def test_classifier_run_profiles_full_optimizer_update(tmp_path: Path) -> None:
    runner = object.__new__(SuccessClassifierTrainingRunner)
    runner.cfg = OmegaConf.create(
        {
            "training": {
                "num_epochs": 1,
                "steps_per_epoch": 1,
                "eval_every": 100,
                "checkpoint_every_epochs": 1,
                "log_every": 100,
                "label_smoothing": 0.0,
                "loss_type": "ce",
                "class_balanced": False,
                "precision": "fp32",
                "update_profile_steps": 1,
            }
        }
    )
    runner.config = runner.cfg
    runner._output_dir = str(tmp_path)
    runner.device = torch.device("cpu")
    runner.model = torch.nn.Linear(3, 2)
    runner.optim = torch.optim.SGD(runner.model.parameters(), lr=0.01)
    runner.train_loader = [
        (
            torch.ones(2, 3),
            torch.tensor([0, 1], dtype=torch.long),
        )
    ]
    runner.val_loader = []
    runner.train_ds = object()
    runner.distributed = _FakeDistributed()
    runner.epoch = 0
    runner.global_step = 0
    runner.best_window_f1 = -1.0
    runner.best_episode_f1 = -1.0
    runner.best_window_ckpt_path = None
    runner.best_episode_ckpt_path = None
    runner._log = lambda _payload: None
    runner.console_banner = lambda *_args, **_kwargs: None
    runner.console_progress = lambda *_args, **_kwargs: None
    runner.console_metrics = lambda *_args, **_kwargs: None
    runner.save_checkpoint = lambda *_args, **_kwargs: ""
    runner._finalize_validation_checkpoints = lambda: {}
    logged: list[dict[str, float]] = []
    runner.log_metrics = lambda metrics, **_kwargs: logged.append(dict(metrics))

    summary = runner.run()

    profile_keys = {key for metrics in logged for key in metrics}
    for stage in (
        "data_wait",
        "h2d",
        "forward",
        "backward",
        "grad_clip",
        "optimizer",
        "metrics",
        "total",
    ):
        assert f"time/classifier_update_{stage}_ms" in profile_keys
    assert "time/classifier_update_device_active_fraction" in profile_keys
    assert summary["total_steps"] == 1


def test_classifier_resume_precedes_first_log(tmp_path: Path) -> None:
    runner = object.__new__(SuccessClassifierTrainingRunner)
    runner.cfg = OmegaConf.create({"training": {"resume": True}})
    runner.config = runner.cfg
    runner._output_dir = str(tmp_path)
    runner._log_path = tmp_path / "logs" / "train_log.jsonl"
    runner._metric_logger = None
    runner._metric_resume_step = None
    runner.global_step = 0
    runner._pending_setup_logs = []
    events: list[str] = []

    def resume(_cfg) -> None:
        runner.global_step = 17
        events.append("resume")

    runner.resume = resume
    runner._log = lambda _payload: events.append(f"log:{runner.global_step}")

    runner._finish_setup_after_optimizer()

    assert events[:2] == ["resume", "log:17"]
    assert runner._metric_resume_step == 17


def test_classifier_resume_keeps_jsonl(tmp_path: Path) -> None:
    runner = object.__new__(SuccessClassifierTrainingRunner)
    runner._log_path = tmp_path / "logs" / "train_log.jsonl"
    runner._log_path.parent.mkdir(parents=True)
    runner._log_path.write_text('{"event":"before"}\n', encoding="utf-8")

    runner._prepare_train_log(resume=True)

    assert runner._log_path.read_text(encoding="utf-8") == '{"event":"before"}\n'


def test_classifier_final_save_writes_only_flat_latest(tmp_path: Path) -> None:
    runner = object.__new__(SuccessClassifierTrainingRunner)
    runner.cfg = OmegaConf.create({"training": {"topk_k": 0}, "classifier": {"latent_dim": 2}})
    runner.config = runner.cfg
    runner._output_dir = str(tmp_path)
    runner.model = torch.nn.Linear(2, 1)
    runner.optim = torch.optim.AdamW(runner.model.parameters(), lr=1.0e-3)
    runner.distributed = _FakeDistributed()
    runner.global_step = 4
    runner.epoch = 2
    runner.best_window_f1 = -1.0
    runner.best_episode_f1 = -1.0
    runner.best_window_ckpt_path = None
    runner.best_episode_ckpt_path = None

    runner._save_final_checkpoint()

    latest = tmp_path / "checkpoints" / "latest.ckpt"
    assert latest.is_file()
    assert not (tmp_path / "checkpoints" / "classifier_warmup.ckpt").exists()


@pytest.mark.parametrize(
    ("num_epochs", "cadence", "start_epoch", "expected_saves", "expected_loader_epochs"),
    [
        (2, 1, 0, [("latest", 1, 2), ("latest_final", 2, 4)], [0, 1]),
        (1, 0, 0, [("latest_final", 1, 2)], [0]),
        (4, 1, 3, [("latest_final", 4, 2)], [3]),
    ],
)
def test_classifier_latest_checkpoint_is_saved_only_after_epoch_boundary(
    tmp_path: Path,
    num_epochs: int,
    cadence: int,
    start_epoch: int,
    expected_saves: list[tuple[str, int, int]],
    expected_loader_epochs: list[int],
) -> None:
    runner = object.__new__(SuccessClassifierTrainingRunner)
    runner.cfg = OmegaConf.create(
        {
            "training": {
                "num_epochs": num_epochs,
                "steps_per_epoch": 2,
                "eval_every": 100,
                "checkpoint_every_epochs": cadence,
                "log_every": 100,
                "label_smoothing": 0.0,
                "loss_type": "ce",
                "class_balanced": False,
                "precision": "fp32",
            }
        }
    )
    runner.config = runner.cfg
    runner._output_dir = str(tmp_path)
    runner.device = torch.device("cpu")
    runner.model = torch.nn.Linear(3, 2)
    runner.optim = torch.optim.SGD(runner.model.parameters(), lr=0.01)
    runner.train_loader = [
        (torch.ones(2, 3), torch.tensor([0, 1], dtype=torch.long)),
        (torch.ones(2, 3), torch.tensor([0, 1], dtype=torch.long)),
    ]
    runner.val_loader = []
    runner.train_ds = object()
    runner.distributed = _FakeDistributed()
    runner.epoch = start_epoch
    runner.global_step = 0
    runner.best_window_f1 = -1.0
    runner.best_episode_f1 = -1.0
    runner.best_window_ckpt_path = None
    runner.best_episode_ckpt_path = None
    runner._log = lambda _payload: None
    runner.console_banner = lambda *_args, **_kwargs: None
    runner.console_progress = lambda *_args, **_kwargs: None
    runner.console_metrics = lambda *_args, **_kwargs: None
    runner._finalize_validation_checkpoints = lambda: {}
    runner.log_metrics = lambda *_args, **_kwargs: None
    loader_epochs: list[int] = []
    runner.set_dataloader_epoch = lambda _loader, epoch: loader_epochs.append(int(epoch))
    saves: list[tuple[str, int, int]] = []
    runner.save_checkpoint = lambda *, tag: (
        saves.append((str(tag), int(runner.epoch), int(runner.global_step))) or ""
    )
    runner._save_final_checkpoint = lambda: (
        saves.append(("latest_final", int(runner.epoch), int(runner.global_step))) or ""
    )

    runner.run()

    assert saves == expected_saves
    assert loader_epochs == expected_loader_epochs


def test_demo_pair_partition_is_deterministic_disjoint_and_complete() -> None:
    pairs = [
        (Path(f"raw_{index}.hdf5"), Path(f"hidden_{index}.hdf5"), "data/demo_0")
        for index in range(10)
    ]

    train = _partition_demo_pairs(
        pairs,
        split="train",
        val_fraction=0.2,
        split_seed=7,
    )
    val = _partition_demo_pairs(
        pairs,
        split="val",
        val_fraction=0.2,
        split_seed=7,
    )

    assert len(train) == 8
    assert len(val) == 2
    assert set(train).isdisjoint(set(val))
    assert set(train) | set(val) == set(pairs)
    assert train == _partition_demo_pairs(
        pairs,
        split="train",
        val_fraction=0.2,
        split_seed=7,
    )


class _TinyMapDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.tensor([float(idx)]), int(idx % 2)

    @staticmethod
    def collate_fn(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        xs = torch.stack([item[0] for item in batch])
        ys = torch.tensor([item[1] for item in batch])
        return xs, ys


class _TinyIterableDataset(IterableDataset):
    def __iter__(self):
        yield torch.tensor([1.0]), 1

    @staticmethod
    def collate_fn(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        xs = torch.stack([item[0] for item in batch])
        ys = torch.tensor([item[1] for item in batch])
        return xs, ys


class _FakeDistributed:
    rank = 1
    local_rank = 1
    world_size = 2
    is_distributed = True
    is_main_process = True
    requires_collective_checkpointing = False

    def __init__(self) -> None:
        self.sampler_calls: list[tuple[object, bool, bool]] = []

    def maybe_make_sampler(self, dataset: object, shuffle: bool, drop_last: bool):
        self.sampler_calls.append((dataset, shuffle, drop_last))
        return SequentialSampler(dataset)

    def unwrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
        return module.module if hasattr(module, "module") else module

    def clip_grad_norm(self, module: torch.nn.Module, max_norm: float) -> float:
        return float(torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm))

    def reduce_mean_dict(self, metrics: dict[str, float | int]) -> dict[str, float]:
        return {key: float(value) for key, value in metrics.items()}

    def barrier(self) -> None:
        return None

    def cleanup(self) -> None:
        return None


def test_classifier_loader_uses_distributed_sampler_for_map_style_dataset() -> None:
    runner = object.__new__(SuccessClassifierTrainingRunner)
    distributed = _FakeDistributed()
    runner.distributed = distributed

    dataset = _TinyMapDataset()
    loader = runner._make_classifier_loader(
        dataset,
        batch_size=2,
        num_workers=0,
        shuffle=True,
        drop_last=True,
        use_distributed_sampler=True,
    )

    assert distributed.sampler_calls == [(dataset, True, True)]
    assert isinstance(loader.sampler, SequentialSampler)


def test_classifier_loader_does_not_attach_sampler_to_iterable_dataset() -> None:
    runner = object.__new__(SuccessClassifierTrainingRunner)
    distributed = _FakeDistributed()
    runner.distributed = distributed

    loader = runner._make_classifier_loader(
        _TinyIterableDataset(),
        batch_size=2,
        num_workers=0,
        shuffle=False,
        drop_last=True,
        use_distributed_sampler=True,
    )

    assert distributed.sampler_calls == []
    assert loader.batch_size == 2


def test_classifier_runner_marks_iterable_train_dataset_for_rank_sharding() -> None:
    runner = object.__new__(SuccessClassifierTrainingRunner)
    distributed = _FakeDistributed()
    runner.distributed = distributed
    dataset = _TinyIterableDataset()

    runner._prepare_train_dataset_for_distributed(dataset)

    assert dataset.distributed_rank == 1
    assert dataset.distributed_world_size == 2


def test_named_classifier_checkpoint_saves_full_flat_payload_once(tmp_path: Path) -> None:
    inner = torch.nn.Linear(2, 1)
    wrapper = torch.nn.Module()
    wrapper.module = inner
    runner = object.__new__(SuccessClassifierTrainingRunner)
    runner._output_dir = str(tmp_path)
    runner.cfg = OmegaConf.create(
        {
            "classifier": {"latent_dim": 2},
            "checkpoint": {
                "topk": {
                    "monitor_key": "f1",
                    "metric_name": "f1",
                    "mode": "max",
                    "k": 1,
                }
            },
        }
    )
    runner.config = runner.cfg
    runner.model = wrapper
    runner.distributed = _FakeDistributed()
    runner.global_step = 12
    runner.epoch = 2
    runner.best_window_ckpt_path = None
    runner.best_episode_ckpt_path = None
    runner._log = lambda _payload: None

    runner._save_named("best_window_f10.5000_th0.50", extra={"val_window": {"best_f1": 0.5}})

    latest = tmp_path / "checkpoints" / "latest.ckpt"
    metric = tmp_path / "checkpoints" / "epoch=0002-f1=0.500000.ckpt"
    payload = torch.load(metric, map_location="cpu", weights_only=False)

    assert latest.is_file()
    assert metric.is_file()
    assert latest.read_bytes() == metric.read_bytes()
    assert sorted(payload["state_dicts"]["model"].keys()) == ["bias", "weight"]
    assert payload["classifier_threshold"] == 0.5


def test_named_classifier_checkpoint_is_rank_zero_only(tmp_path: Path) -> None:
    runner = object.__new__(SuccessClassifierTrainingRunner)
    runner._output_dir = str(tmp_path)
    runner.cfg = OmegaConf.create(
        {
            "classifier": {"latent_dim": 1},
            "checkpoint": {
                "topk": {
                    "monitor_key": "f1",
                    "metric_name": "f1",
                    "mode": "max",
                    "k": 1,
                }
            },
        }
    )
    runner.config = runner.cfg
    runner.model = torch.nn.Linear(1, 1)
    runner.global_step = 0
    runner.epoch = 1
    runner.best_window_ckpt_path = None
    runner.best_episode_ckpt_path = None
    runner._log = lambda _payload: None
    distributed = _FakeDistributed()
    distributed.is_main_process = False
    runner.distributed = distributed

    runner._save_named("not_rank_zero")

    assert not (tmp_path / "checkpoints" / "latest.ckpt").exists()
    assert not (tmp_path / "checkpoints" / "epoch=0001-f1=0.000000.ckpt").exists()


def test_configured_final_selection_materializes_window_best_checkpoint() -> None:
    runner = object.__new__(SuccessClassifierTrainingRunner)
    runner.cfg = OmegaConf.create(
        {
            "training": {
                "episode_eval_enabled": False,
                "final_selection_metric": "window_f1",
                "topk_k": 1,
            }
        }
    )
    runner.best_window_f1 = -1.0
    runner.best_episode_f1 = -1.0
    runner.best_window_ckpt_path = None
    runner.best_episode_ckpt_path = None
    runner._evaluate_window_level = lambda: {"best_f1": 0.4, "best_thresh": 0.3}
    saved: list[tuple[str, dict]] = []
    runner._maybe_save_named = lambda name, extra=None: saved.append((name, extra))

    metrics = runner._finalize_validation_checkpoints()

    assert metrics["window"]["best_f1"] == 0.4
    assert runner.best_window_f1 == 0.4
    assert saved[-1][0] == "best_window_f10.4000_th0.30"


def test_final_selection_always_offers_current_metric_to_topk() -> None:
    runner = object.__new__(SuccessClassifierTrainingRunner)
    runner.cfg = OmegaConf.create(
        {
            "training": {
                "episode_eval_enabled": False,
                "final_selection_metric": "window_f1",
                "topk_k": 1,
            }
        }
    )
    runner.best_window_f1 = 0.9
    runner.best_episode_f1 = -1.0
    runner._evaluate_window_level = lambda: {"best_f1": 0.4, "best_thresh": 0.3}
    saved: list[tuple[str, dict]] = []
    runner._maybe_save_named = lambda name, extra=None: saved.append((name, extra))

    runner._finalize_validation_checkpoints()

    assert runner.best_window_f1 == 0.9
    assert saved[-1][1]["val_window"]["best_f1"] == 0.4


class _StreamingEpisodeDataset:
    K = 1
    chunk_pool = "last"
    total = 3

    def __init__(self) -> None:
        self.yielded = 0

    def trajectories(self):
        for idx, complete in enumerate([False, True, False]):
            self.yielded += 1
            yield np.zeros((1, 1), dtype=np.float32), complete, 1, f"episode_{idx}", {}


class _StreamingGuardClassifier(torch.nn.Module):
    supports_language_conditioning = False
    supports_proprio_conditioning = False
    supports_task_conditioning = False

    def __init__(self, dataset: _StreamingEpisodeDataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.calls = 0

    def forward(self, xs: torch.Tensor, **_: object) -> torch.Tensor:
        if self.calls == 0:
            assert self.dataset.yielded < self.dataset.total, (
                "episode eval consumed all trajectories before first classifier forward"
            )
        self.calls += 1
        logits = torch.tensor([[10.0, 0.0]], dtype=torch.float32, device=xs.device)
        return logits.repeat(xs.shape[0], 1)


def test_episode_eval_streams_windows_before_consuming_all_trajectories() -> None:
    dataset = _StreamingEpisodeDataset()
    model = _StreamingGuardClassifier(dataset)
    runner = object.__new__(SuccessClassifierTrainingRunner)
    runner.model = model
    runner.device = torch.device("cpu")
    runner.val_ds = dataset
    runner.cfg = OmegaConf.create(
        {
            "data": {"window": 1},
            "training": {
                "episode_eval_batch": 1,
                "episode_eval_min_steps": 0,
                "episode_eval_stride": 1,
                "thresh_min": 0.5,
                "thresh_max": 0.5,
                "thresh_steps": 1,
            },
        }
    )

    metrics = runner._evaluate_episode_level()

    assert model.calls > 0
    assert metrics["n"] == 3


def test_val_trajectories_preserve_obs_dtype_for_streaming_eval() -> None:
    dataset = object.__new__(LumosAlignedLatentValDataset)
    obs = np.zeros((2, 3), dtype=np.float16)
    dataset._demos = [
        _DemoRecord(
            obs=obs,
            proprio=None,
            lang_emb=None,
            finish_step=2,
            complete=True,
            eid="demo_0",
        )
    ]

    yielded_obs, complete, finish_step, eid, extra = next(dataset.trajectories())

    assert yielded_obs.dtype == np.float16
    assert np.shares_memory(yielded_obs, obs)
    assert complete is True
    assert finish_step == 2
    assert eid == "demo_0"
    assert extra == {}


def _demo(*, complete: bool, value: float) -> _DemoRecord:
    obs = np.full((12, 2), value, dtype=np.float32)
    return _DemoRecord(
        obs=obs,
        proprio=None,
        lang_emb=None,
        finish_step=12,
        complete=complete,
        eid=f"{'success' if complete else 'failure'}_{value}",
    )


def test_wmpo_train_stream_balances_positive_and_negative_pairs() -> None:
    dataset = object.__new__(LumosAlignedLatentTrainDataset)
    dataset._demos = [
        _demo(complete=True, value=1.0),
        _demo(complete=True, value=2.0),
        _demo(complete=False, value=3.0),
        _demo(complete=True, value=4.0),
    ]
    dataset.W = 2
    dataset.K = 1
    dataset.S = 1
    dataset.window_env = 2
    dataset.seed = 0
    dataset.chunk_pool = "last"
    dataset.sampling_protocol = "wmpo"
    dataset.balance_batches = True

    stream = iter(dataset)
    labels = [int(next(stream)[1]) for _ in range(4)]

    assert labels == [1, 0, 1, 0]


def test_train_stream_can_shard_demo_ids_by_distributed_rank() -> None:
    dataset = object.__new__(LumosAlignedLatentTrainDataset)
    dataset._demos = [
        _demo(complete=True, value=1.0),
        _demo(complete=True, value=2.0),
        _demo(complete=True, value=3.0),
        _demo(complete=True, value=4.0),
    ]
    dataset.W = 2
    dataset.K = 1
    dataset.S = 1
    dataset.window_env = 2
    dataset.seed = 0
    dataset.chunk_pool = "last"
    dataset.sampling_protocol = "lumos"
    dataset.balance_batches = False
    dataset.distributed_rank = 1
    dataset.distributed_world_size = 2

    stream = iter(dataset)
    values = {float(next(stream)[0][0, 0].item()) for _ in range(4)}

    assert values <= {2.0, 4.0}
    assert values


def test_wmpo_success_negative_range_excludes_terminal_overlap() -> None:
    dataset = object.__new__(LumosAlignedLatentTrainDataset)
    dataset.W = 3
    dataset.K = 1
    dataset.S = 1
    dataset.window_env = 3
    dataset.chunk_pool = "last"

    ends = dataset._wmpo_success_negative_ends(finish_step=12)

    assert max(ends) == 9
    assert 10 not in ends
    assert 11 not in ends


def test_success_probabilities_support_bce_and_two_class_logits() -> None:
    bce_logits = torch.tensor([[-2.0], [0.0], [2.0]], dtype=torch.float32)
    ce_logits = torch.tensor([[2.0, -2.0], [0.0, 0.0], [-2.0, 2.0]], dtype=torch.float32)

    assert torch.allclose(
        _success_probabilities_from_logits(bce_logits),
        torch.sigmoid(bce_logits.squeeze(-1)),
    )
    assert torch.allclose(
        _success_probabilities_from_logits(ce_logits),
        torch.softmax(ce_logits, dim=-1)[:, 1],
    )


def test_bce_classifier_loss_uses_float_targets_and_sigmoid_predictions() -> None:
    logits = torch.tensor([[-4.0], [4.0], [0.25], [-0.25]], dtype=torch.float32)
    labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)

    loss, pred = _classifier_loss_and_predictions(
        logits,
        labels,
        loss_type="bce",
        label_smoothing=0.0,
        class_weight=None,
    )

    assert loss.item() > 0.0
    assert pred.tolist() == [0, 1, 1, 0]


def test_predict_success_supports_single_bce_logit() -> None:
    model = LatentSuccessClassifier(
        LatentSuccessClassifierConfig(
            latent_dim=1,
            window=1,
            head_type="linear",
            output_dim=1,
            granularity="action",
        )
    )
    with torch.no_grad():
        model.head.weight.fill_(1.0)
        model.head.bias.zero_()

    result = model.predict_success(
        torch.tensor([[[-2.0], [2.0]]], dtype=torch.float32),
        threshold=0.5,
        stride=1,
    )

    assert result["complete"].tolist() == [True]
    assert result["finish_step"].tolist() == [1]
    assert result["score"].item() > 0.8
