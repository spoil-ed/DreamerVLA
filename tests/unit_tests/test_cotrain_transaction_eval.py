from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import dreamervla.runtime.cotrain_eval as eval_transaction
from dreamervla.runtime.cotrain_eval import (
    CotrainEvalObserver,
    CotrainTransactionAccumulator,
    EncodedEvalTrajectory,
    binary_classification_metrics,
    closed_loop_world_model_trajectory,
)
from dreamervla.workers.actor.learner_worker import LearnerWorker
from dreamervla.workers.cotrain.messages import RealTrajectory, RealTrajectoryBatch


class _RecursiveWorldModel(torch.nn.Module):
    num_hist = 1
    chunk_size = 2
    action_dim = 1
    token_dim = 1
    max_seq_len = 2

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.history_inputs: list[torch.Tensor] = []

    def obs_to_tokens(self, obs: torch.Tensor) -> torch.Tensor:
        if int(obs.shape[1]) > self.max_seq_len:
            raise ValueError("diagnostic must stream trajectories over max_seq_len")
        return obs

    def predict_next_chunk(self, latent, actions):
        history = latent["history"]
        self.history_inputs.append(history.detach().clone())
        current = history[:, -1]
        predictions = []
        for _index in range(self.chunk_size):
            current = current + self.scale
            predictions.append(current)
        hidden_seq = torch.stack(predictions, dim=1)
        next_history = hidden_seq[:, -1:]
        return {
            "hidden": hidden_seq[:, -1],
            "hidden_seq": hidden_seq,
            "history": next_history,
            "actions": torch.zeros_like(latent["actions"]),
            "lang": latent.get("lang"),
        }


class _TaskAwareTrajectoryClassifier(torch.nn.Module):
    supports_task_conditioning = True
    supports_proprio_conditioning = False
    supports_language_conditioning = False

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def predict_success(self, video, *, task_ids, **_kwargs):
        del video
        scores = torch.where(
            task_ids == 0,
            self.scale.new_tensor(0.9),
            self.scale.new_tensor(0.1),
        )
        return {"score": scores}


def _trajectory(values: list[float], *, task_id: int, success: bool):
    hidden = torch.tensor(values, dtype=torch.float32).reshape(-1, 1, 1)
    return EncodedEvalTrajectory(
        task_id=task_id,
        success=success,
        hidden=hidden,
        actions=torch.zeros((len(values), 1), dtype=torch.float32),
    )


def _real_trajectory(
    values: list[float],
    *,
    task_id: int,
    success: bool,
) -> RealTrajectory:
    transitions = tuple(
        {
            "obs_embedding": np.asarray([[value]], dtype=np.float32),
            "action": np.zeros((1,), dtype=np.float32),
        }
        for value in values
    )
    return RealTrajectory(
        env_rank=0,
        slot_id=task_id,
        task_id=task_id,
        episode_id=task_id,
        global_step=3,
        success=success,
        transitions=transitions,
    )


def test_learner_evaluates_encoded_real_trajectories_without_training() -> None:
    learner = LearnerWorker({}, {}, {}, replay=None)
    learner.world_model = _RecursiveWorldModel()
    learner.classifier = _TaskAwareTrajectoryClassifier()
    learner.classifier_threshold = 0.5
    learner.world_model.train()
    learner.classifier.train()
    batch = RealTrajectoryBatch(
        global_step=3,
        trajectories=(
            _real_trajectory([1.0, 2.0, 3.0, 4.0, 5.0], task_id=0, success=True),
            _real_trajectory([1.0, 2.0, 3.0, 4.0, 5.0], task_id=1, success=False),
        ),
    )

    metrics = learner.evaluate_cotrain_trajectories(batch)

    assert metrics["eval/cotrain_trajectory_count"] == 2.0
    assert metrics["eval/wm_closed_loop_cosine"] == pytest.approx(1.0)
    assert metrics["eval/wm_trajectory_cosine"] == pytest.approx(1.0)
    assert metrics["eval/classifier_real_f1"] == 1.0
    assert metrics["eval/classifier_real_accuracy"] == 1.0
    assert metrics["eval/cls_trajectory_f1"] == 1.0
    assert metrics["eval/cls_trajectory_accuracy"] == 1.0
    assert metrics["eval/classifier_wm_f1"] == 1.0
    assert metrics["eval/classifier_wm_accuracy"] == 1.0
    assert learner.world_model.training is True
    assert learner.classifier.training is True


def test_closed_loop_world_model_is_autoregressive_across_full_trajectory() -> None:
    model = _RecursiveWorldModel()
    trajectory = _trajectory([0.0, 10.0, 20.0, 30.0, 40.0], task_id=0, success=False)

    result = closed_loop_world_model_trajectory(model, trajectory)

    assert result.predicted_hidden.reshape(-1).tolist() == [1.0, 2.0, 3.0, 4.0]
    assert result.target_hidden.reshape(-1).tolist() == [10.0, 20.0, 30.0, 40.0]
    assert len(model.history_inputs) == 2
    # Chunk two starts from chunk one's prediction, not the real target 20.
    assert model.history_inputs[1].reshape(-1).tolist() == [2.0]


def test_wm_summary_weights_trajectories_equally_not_frames() -> None:
    accumulator = CotrainTransactionAccumulator(
        classifier_threshold=0.45,
        threshold_source="checkpoint",
    )
    accumulator.add_world_model_metrics(
        task_id=0,
        mse_by_horizon=[0.0, 0.0, 0.0, 0.0],
        cosine_by_horizon=[1.0, 1.0, 1.0, 1.0],
    )
    accumulator.add_world_model_metrics(
        task_id=1,
        mse_by_horizon=[100.0],
        cosine_by_horizon=[0.0],
    )

    summary = accumulator.summarize()

    # Per-frame pooling would be 20; trajectory means are (0 + 100) / 2 = 50.
    assert summary["wm_closed_loop_mse"] == 50.0
    assert summary["wm_closed_loop_cosine"] == 0.5
    assert summary["wm_horizon"]["mse"] == [50.0, 0.0, 0.0, 0.0]
    assert summary["classifier_threshold"] == 0.45
    assert summary["classifier_threshold_source"] == "checkpoint"


def test_classifier_metrics_report_f1_auc_and_undefined_auc() -> None:
    metrics = binary_classification_metrics(
        labels=[1, 0, 1, 0],
        scores=[0.9, 0.2, 0.8, 0.1],
        threshold=0.5,
    )

    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc_defined"] is True
    assert metrics["pr_auc_defined"] is True
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["true_negatives"] == 2
    assert metrics["predicted_positive_fraction"] == 0.5

    undefined = binary_classification_metrics(labels=[1, 1], scores=[0.2, 0.9], threshold=0.5)
    assert undefined["roc_auc"] is None
    assert undefined["roc_auc_defined"] is False
    assert undefined["pr_auc"] is None
    assert undefined["pr_auc_defined"] is False
    assert not math.isnan(float(undefined["f1"]))


def test_accumulator_keeps_real_and_wm_classifier_results_separate() -> None:
    accumulator = CotrainTransactionAccumulator(
        classifier_threshold=0.5,
        threshold_source="checkpoint",
    )
    accumulator.add_classifier_result(
        task_id=0,
        success=True,
        real_score=0.9,
        wm_score=0.1,
    )
    accumulator.add_classifier_result(
        task_id=0,
        success=False,
        real_score=0.2,
        wm_score=0.8,
    )

    summary = accumulator.summarize()

    assert summary["real_classifier"]["f1"] == 1.0
    assert summary["wm_classifier"]["f1"] == 0.0
    assert summary["per_task"]["0"]["real_classifier"]["f1"] == 1.0
    assert summary["per_task"]["0"]["wm_classifier"]["f1"] == 0.0


def test_accumulator_merges_raw_rank_states_before_recomputing_auc() -> None:
    positive_rank = CotrainTransactionAccumulator(
        classifier_threshold=0.5,
        threshold_source="checkpoint",
    )
    positive_rank.add_world_model_metrics(
        task_id=0,
        mse_by_horizon=[1.0],
        cosine_by_horizon=[0.5],
    )
    positive_rank.add_classifier_result(
        task_id=0,
        success=True,
        real_score=0.9,
        wm_score=0.8,
    )
    negative_rank = CotrainTransactionAccumulator(
        classifier_threshold=0.5,
        threshold_source="checkpoint",
    )
    negative_rank.add_world_model_metrics(
        task_id=1,
        mse_by_horizon=[3.0],
        cosine_by_horizon=[0.25],
    )
    negative_rank.add_classifier_result(
        task_id=1,
        success=False,
        real_score=0.1,
        wm_score=0.2,
    )

    merged = CotrainTransactionAccumulator.from_rank_states(
        [positive_rank.rank_state(), negative_rank.rank_state()]
    ).summarize()

    assert merged["trajectory_count"] == 2
    assert merged["wm_closed_loop_mse"] == 2.0
    assert merged["real_classifier"]["roc_auc"] == 1.0
    assert merged["wm_classifier"]["roc_auc"] == 1.0


def test_observer_formats_global_metrics_from_raw_rank_states() -> None:
    rank_states = []
    for task_id, success, real_score, wm_score in (
        (0, True, 0.9, 0.8),
        (1, False, 0.1, 0.2),
    ):
        accumulator = CotrainTransactionAccumulator(
            classifier_threshold=0.5,
            threshold_source="checkpoint",
        )
        accumulator.add_world_model_metrics(
            task_id=task_id,
            mse_by_horizon=[float(task_id + 1)],
            cosine_by_horizon=[0.5],
        )
        accumulator.add_classifier_result(
            task_id=task_id,
            success=success,
            real_score=real_score,
            wm_score=wm_score,
        )
        rank_states.append(accumulator.rank_state())

    metrics = CotrainEvalObserver.metrics_from_rank_states(
        rank_states,
        expected_trajectories=2,
    )

    assert metrics["eval/cotrain_trajectory_count"] == 2.0
    assert metrics["eval/cotrain_expected_trajectories"] == 2.0
    assert metrics["eval/classifier_real_roc_auc"] == 1.0
    assert metrics["eval_cotrain_diagnostics"]["trajectory_count"] == 2


def test_observer_rejects_a_local_rank_count_mismatch_after_gather() -> None:
    accumulator = CotrainTransactionAccumulator(
        classifier_threshold=0.5,
        threshold_source="checkpoint",
    )

    with pytest.raises(RuntimeError, match="rank 0.*expected 1, got 0"):
        CotrainEvalObserver.metrics_from_rank_payloads(
            [
                {
                    "pending_trajectory_count": 0,
                    "expected_trajectories": 1,
                    "state": accumulator.rank_state(),
                }
            ],
            expected_trajectories=1,
        )


class _RawEvalPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.raw_images: list[np.ndarray] = []

    def prepare_raw_batch(self, transitions):
        self.raw_images.extend(
            np.asarray(transition["agentview_rgb"]).copy() for transition in transitions
        )
        batch = len(transitions)
        return {
            "input_ids": torch.ones((batch, 1), dtype=torch.long),
            "attention_mask": torch.ones((batch, 1), dtype=torch.long),
            "pixel_values": torch.ones((batch, 1), dtype=torch.float32),
        }

    def forward(self, batch):
        count = int(batch["input_ids"].shape[0])
        hidden = torch.arange(count, dtype=torch.float32).reshape(count, 1, 1)
        return (
            hidden,
            hidden.new_zeros(()),
            {
                "hidden": hidden,
                "lang_emb": torch.ones((count, 2), dtype=torch.float32),
            },
        )


def test_eval_observer_builds_one_physical_trajectory_and_discards_it(
    monkeypatch,
) -> None:
    captured: list[EncodedEvalTrajectory] = []

    def fake_evaluate(*, world_model, classifier, trajectory, accumulator):
        del world_model, classifier
        captured.append(trajectory)
        accumulator.add_world_model_metrics(
            task_id=trajectory.task_id,
            mse_by_horizon=[0.25],
            cosine_by_horizon=[0.75],
        )
        accumulator.add_classifier_result(
            task_id=trajectory.task_id,
            success=trajectory.success,
            real_score=0.8,
            wm_score=0.7,
        )

    monkeypatch.setattr(
        eval_transaction,
        "evaluate_encoded_cotrain_trajectory",
        fake_evaluate,
    )
    policy = _RawEvalPolicy()
    observer = CotrainEvalObserver(
        policy=policy,
        world_model=torch.nn.Linear(1, 1),
        classifier=torch.nn.Linear(1, 1),
        classifier_threshold=0.5,
        expected_trajectories=1,
        encode_batch_size=2,
        device=torch.device("cpu"),
    )
    env = SimpleNamespace(
        num_envs=1,
        task_ids=np.asarray([3]),
        reset_state_ids=np.asarray([17]),
    )
    reset_obs = {
        "main_images": np.asarray([[[[1], [2]], [[3], [4]]]], dtype=np.uint8),
        "states": np.asarray([[0.0, 1.0]], dtype=np.float32),
        "task_descriptions": ["do task three"],
    }
    observer.on_reset(env=env, obs=reset_obs, infos={}, epoch=0)
    next_obs = {
        "main_images": np.asarray([[[[5], [6]], [[7], [8]]]], dtype=np.uint8),
        "states": np.asarray([[2.0, 3.0]], dtype=np.float32),
        "task_descriptions": ["do task three"],
    }
    final_obs = {
        "main_images": np.asarray([[[[9], [10]], [[11], [12]]]], dtype=np.uint8),
        "states": np.asarray([[4.0, 5.0]], dtype=np.float32),
        "task_descriptions": ["do task three"],
    }
    episode = {
        "task_id": np.asarray([3]),
        "reset_state_id": np.asarray([17]),
        "success_once": np.asarray([True]),
        "episode_len": np.asarray([2]),
    }
    observer.on_chunk(
        env=env,
        obs_before=reset_obs,
        chunk_actions=np.asarray([[[0.1], [0.2]]], dtype=np.float32),
        obs_list=[next_obs, final_obs],
        rewards=np.zeros((1, 2), dtype=np.float32),
        terms=np.asarray([[False, True]]),
        truncs=np.asarray([[False, False]]),
        infos_list=[{"episode": episode}, {"episode": episode}],
        newly_done=np.asarray([True]),
        episode_info=episode,
        epoch=0,
        chunk_index=0,
    )

    assert len(captured) == 1
    trajectory = captured[0]
    assert trajectory.task_id == 3
    assert trajectory.reset_state_id == 17
    assert trajectory.success is True
    assert trajectory.actions.reshape(-1).tolist() == pytest.approx([0.1, 0.2])
    assert trajectory.proprio.tolist() == [[0.0, 1.0], [2.0, 3.0]]
    # LiberoEnv's public image is rotated; the observer restores raw RGB before
    # asking the policy to run its own deployment preprocessing exactly once.
    assert policy.raw_images[0].reshape(-1).tolist() == [4, 3, 2, 1]
    metrics = observer.finalize_metrics()
    assert metrics["eval/cotrain_trajectory_count"] == 1.0
    assert metrics["eval/wm_closed_loop_mse"] == 0.25
    assert observer.pending_trajectory_count == 0
