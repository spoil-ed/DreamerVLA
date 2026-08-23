"""Read-only trajectory evaluation for staged OpenVLA cotrain checkpoints.

The diagnostic boundary is deliberately independent from replay and optimizers.
It consumes one already encoded real trajectory at a time, rolls the world model
autoregressively for the full available horizon, and accumulates equally weighted
trajectory metrics for the real classifier and the WM->classifier composition.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from dreamervla.workers.cotrain.messages import RealTrajectory


@dataclass(frozen=True)
class EncodedEvalTrajectory:
    """One fixed real-evaluation trajectory in the current encoder space."""

    task_id: int
    success: bool
    hidden: torch.Tensor
    actions: torch.Tensor
    proprio: torch.Tensor | None = None
    lang_emb: torch.Tensor | None = None
    reset_state_id: int | None = None


@dataclass(frozen=True)
class ClosedLoopWorldModelResult:
    """Full autoregressive WM predictions and per-horizon errors."""

    predicted_hidden: torch.Tensor
    target_hidden: torch.Tensor
    mse_by_horizon: torch.Tensor
    cosine_by_horizon: torch.Tensor
    predicted_proprio: torch.Tensor | None = None


def _model_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _model_dtype(module: torch.nn.Module) -> torch.dtype:
    try:
        dtype = next(module.parameters()).dtype
    except StopIteration:
        return torch.float32
    return dtype if dtype.is_floating_point else torch.float32


def encoded_eval_trajectory_from_real(
    trajectory: RealTrajectory,
) -> EncodedEvalTrajectory:
    """Convert one fully re-encoded real trajectory into the eval contract."""

    transitions = tuple(trajectory.transitions)
    if not transitions:
        raise ValueError("cotrain eval real trajectory must contain transitions")
    missing_hidden = [
        index for index, step in enumerate(transitions) if "obs_embedding" not in step
    ]
    missing_action = [index for index, step in enumerate(transitions) if "action" not in step]
    if missing_hidden:
        raise ValueError(
            "cotrain eval requires obs_embedding on every transition; "
            f"missing indices {missing_hidden[:8]}"
        )
    if missing_action:
        raise ValueError(
            "cotrain eval requires action on every transition; "
            f"missing indices {missing_action[:8]}"
        )

    hidden = torch.stack(
        [torch.as_tensor(step["obs_embedding"]) for step in transitions],
        dim=0,
    )
    actions = torch.stack(
        [torch.as_tensor(step["action"], dtype=torch.float32).reshape(-1) for step in transitions],
        dim=0,
    )
    proprio_values = [step.get("proprio", step.get("state")) for step in transitions]
    proprio = None
    if any(value is not None for value in proprio_values):
        if not all(value is not None for value in proprio_values):
            raise ValueError("cotrain eval trajectory proprio must be present on every transition")
        proprio = torch.stack(
            [torch.as_tensor(value, dtype=torch.float32).reshape(-1) for value in proprio_values],
            dim=0,
        )
    language = next(
        (step.get("lang_emb") for step in transitions if step.get("lang_emb") is not None),
        None,
    )
    return EncodedEvalTrajectory(
        task_id=int(trajectory.task_id),
        success=bool(trajectory.success),
        hidden=hidden,
        actions=actions,
        proprio=proprio,
        lang_emb=(None if language is None else torch.as_tensor(language)),
        reset_state_id=int(trajectory.episode_id),
    )


@torch.no_grad()
def closed_loop_world_model_trajectory(
    world_model: torch.nn.Module,
    trajectory: EncodedEvalTrajectory,
) -> ClosedLoopWorldModelResult:
    """Roll ``world_model`` recursively over every complete action chunk.

    Only the first ``num_hist`` real frames seed the model. Every later chunk
    receives the exact ``history`` and ``actions`` returned by the preceding
    prediction, matching the imagined environment rather than teacher forcing.
    """

    hidden = torch.as_tensor(trajectory.hidden)
    actions = torch.as_tensor(trajectory.actions)
    if hidden.ndim < 2:
        raise ValueError("trajectory hidden must have shape [T,...]")
    if actions.ndim != 2:
        raise ValueError("trajectory actions must have shape [T,action_dim]")
    if int(hidden.shape[0]) != int(actions.shape[0]):
        raise ValueError("trajectory hidden/action lengths must match")

    history_length = int(world_model.num_hist)
    chunk_size = int(world_model.chunk_size)
    action_dim = int(world_model.action_dim)
    total_steps = int(hidden.shape[0])
    chunks = (total_steps - history_length) // chunk_size
    if chunks < 1:
        raise ValueError(
            "trajectory is too short for closed-loop WM evaluation: "
            f"T={total_steps}, num_hist={history_length}, chunk_size={chunk_size}"
        )
    if int(actions.shape[-1]) != action_dim:
        raise ValueError(f"trajectory action dim {int(actions.shape[-1])} != {action_dim}")

    device = _model_device(world_model)
    dtype = _model_dtype(world_model)
    hidden = hidden.to(device=device, dtype=dtype)
    actions = actions.to(device=device, dtype=dtype)
    # ``max_seq_len`` limits one WM encoder call, not the duration of a
    # recursively rolled evaluation. Stream token validation in bounded pieces
    # and retain the full trajectory for closed-loop scoring.
    encode_limit = max(1, int(getattr(world_model, "max_seq_len", total_steps)))
    vision_tokens = torch.cat(
        [
            world_model.obs_to_tokens(hidden[start : start + encode_limit].unsqueeze(0))
            for start in range(0, total_steps, encode_limit)
        ],
        dim=1,
    )

    proprio = None
    if trajectory.proprio is not None:
        proprio = torch.as_tensor(trajectory.proprio).to(device=device, dtype=dtype)
        if int(proprio.shape[0]) != total_steps:
            raise ValueError("trajectory proprio length must match hidden length")

    observation_tokens = vision_tokens
    observation_builder = getattr(world_model, "_observation_tokens", None)
    if proprio is not None and callable(observation_builder):
        observation_tokens = observation_builder(
            vision_tokens,
            proprio.unsqueeze(0),
        )

    action_history = torch.zeros(
        1,
        history_length,
        action_dim,
        device=device,
        dtype=dtype,
    )
    if history_length > 1:
        action_history[:, : history_length - 1] = actions[: history_length - 1].unsqueeze(0)

    language = None
    if trajectory.lang_emb is not None:
        language = torch.as_tensor(trajectory.lang_emb).to(
            device=device,
            dtype=dtype,
        )
        if language.ndim == 1:
            language = language.unsqueeze(0)

    history = observation_tokens[:, :history_length]
    latent: dict[str, torch.Tensor | None] = {
        "hidden": history[:, -1],
        "history": history,
        "actions": action_history,
        "lang": language,
    }
    if proprio is not None:
        latent["proprio"] = proprio[history_length - 1].unsqueeze(0)

    predicted_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    predicted_proprio_parts: list[torch.Tensor] = []
    visual_width = int(vision_tokens.shape[-1])
    for chunk_index in range(chunks):
        start = history_length - 1 + chunk_index * chunk_size
        action_chunk = actions[start : start + chunk_size].unsqueeze(0)
        output = world_model.predict_next_chunk(latent, action_chunk)
        predicted = torch.as_tensor(output["hidden_seq"])[0]
        predicted_visual = predicted[..., :visual_width]
        target = vision_tokens[
            0,
            history_length + chunk_index * chunk_size : history_length
            + (chunk_index + 1) * chunk_size,
        ]
        predicted_parts.append(predicted_visual)
        target_parts.append(target)
        if isinstance(output.get("proprio_seq"), torch.Tensor):
            predicted_proprio_parts.append(output["proprio_seq"][0])
        latent = {
            "hidden": output["hidden"],
            "history": output["history"],
            "actions": output["actions"],
            "lang": output.get("lang", language),
        }
        if isinstance(output.get("proprio"), torch.Tensor):
            latent["proprio"] = output["proprio"]

    predicted_hidden = torch.cat(predicted_parts, dim=0)
    target_hidden = torch.cat(target_parts, dim=0)
    predicted_flat = predicted_hidden.float().reshape(predicted_hidden.shape[0], -1)
    target_flat = target_hidden.float().reshape(target_hidden.shape[0], -1)
    mse = (predicted_flat - target_flat).square().mean(dim=-1)
    cosine = F.cosine_similarity(predicted_flat, target_flat, dim=-1)
    predicted_proprio = (
        torch.cat(predicted_proprio_parts, dim=0) if predicted_proprio_parts else None
    )
    return ClosedLoopWorldModelResult(
        predicted_hidden=predicted_hidden,
        target_hidden=target_hidden,
        mse_by_horizon=mse,
        cosine_by_horizon=cosine,
        predicted_proprio=predicted_proprio,
    )


def binary_classification_metrics(
    *,
    labels: Sequence[int | bool],
    scores: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    """Return fixed-threshold binary metrics with explicit AUC availability."""

    label_values = [int(bool(value)) for value in labels]
    score_values = [float(value) for value in scores]
    if len(label_values) != len(score_values):
        raise ValueError("classifier labels and scores must have equal lengths")
    if not label_values:
        return {
            "examples": 0,
            "positive_fraction": None,
            "predicted_positive_fraction": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "true_negatives": 0,
            "roc_auc": None,
            "roc_auc_defined": False,
            "pr_auc": None,
            "pr_auc_defined": False,
        }

    predictions = [int(score >= float(threshold)) for score in score_values]
    tp = sum(
        int(prediction == 1 and label == 1)
        for prediction, label in zip(predictions, label_values, strict=True)
    )
    fp = sum(
        int(prediction == 1 and label == 0)
        for prediction, label in zip(predictions, label_values, strict=True)
    )
    fn = sum(
        int(prediction == 0 and label == 1)
        for prediction, label in zip(predictions, label_values, strict=True)
    )
    tn = sum(
        int(prediction == 0 and label == 0)
        for prediction, label in zip(predictions, label_values, strict=True)
    )
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0

    positive_scores = [
        score for score, label in zip(score_values, label_values, strict=True) if label
    ]
    negative_scores = [
        score for score, label in zip(score_values, label_values, strict=True) if not label
    ]
    auc: float | None = None
    pr_auc: float | None = None
    if positive_scores and negative_scores:
        wins = 0.0
        for positive in positive_scores:
            for negative in negative_scores:
                wins += float(positive > negative) + 0.5 * float(positive == negative)
        auc = float(wins / (len(positive_scores) * len(negative_scores)))
        ranked = sorted(
            zip(score_values, label_values, strict=True),
            key=lambda item: item[0],
            reverse=True,
        )
        tp_running = 0
        fp_running = 0
        recall_before = 0.0
        average_precision = 0.0
        index = 0
        while index < len(ranked):
            score = ranked[index][0]
            group_labels: list[int] = []
            while index < len(ranked) and ranked[index][0] == score:
                group_labels.append(ranked[index][1])
                index += 1
            tp_running += sum(group_labels)
            fp_running += len(group_labels) - sum(group_labels)
            recall_at_threshold = tp_running / len(positive_scores)
            precision_at_threshold = tp_running / (tp_running + fp_running)
            average_precision += (recall_at_threshold - recall_before) * precision_at_threshold
            recall_before = recall_at_threshold
        pr_auc = float(average_precision)
    return {
        "examples": len(label_values),
        "positive_fraction": float(sum(label_values) / len(label_values)),
        "predicted_positive_fraction": float(sum(predictions) / len(predictions)),
        "accuracy": float((tp + tn) / len(label_values)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "roc_auc": auc,
        "roc_auc_defined": auc is not None,
        "pr_auc": pr_auc,
        "pr_auc_defined": pr_auc is not None,
    }


class CotrainTransactionAccumulator:
    """Streaming, read-only aggregation over fixed evaluation trajectories."""

    def __init__(
        self,
        *,
        classifier_threshold: float,
        threshold_source: str,
    ) -> None:
        self.classifier_threshold = float(classifier_threshold)
        self.threshold_source = str(threshold_source)
        self._wm_records: list[dict[str, Any]] = []
        self._classifier_records: list[dict[str, Any]] = []

    def add_world_model_metrics(
        self,
        *,
        task_id: int,
        mse_by_horizon: Iterable[float],
        cosine_by_horizon: Iterable[float],
    ) -> None:
        mse = [float(value) for value in mse_by_horizon]
        cosine = [float(value) for value in cosine_by_horizon]
        if not mse or len(mse) != len(cosine):
            raise ValueError("WM horizon metrics must be non-empty and aligned")
        self._wm_records.append({"task_id": int(task_id), "mse": mse, "cosine": cosine})

    def add_classifier_result(
        self,
        *,
        task_id: int,
        success: bool,
        real_score: float,
        wm_score: float,
    ) -> None:
        self._classifier_records.append(
            {
                "task_id": int(task_id),
                "success": bool(success),
                "real_score": float(real_score),
                "wm_score": float(wm_score),
            }
        )

    def rank_state(self) -> dict[str, Any]:
        """Return raw, picklable records for exact cross-rank metric recomputation."""

        return {
            "classifier_threshold": self.classifier_threshold,
            "threshold_source": self.threshold_source,
            "wm_records": list(self._wm_records),
            "classifier_records": list(self._classifier_records),
        }

    @classmethod
    def from_rank_states(
        cls,
        states: Iterable[dict[str, Any]],
    ) -> CotrainTransactionAccumulator:
        """Merge raw rank states, rejecting incompatible classifier protocols."""

        rank_states = list(states)
        if not rank_states:
            raise ValueError("cotrain eval rank states must be non-empty")
        first = rank_states[0]
        merged = cls(
            classifier_threshold=float(first["classifier_threshold"]),
            threshold_source=str(first["threshold_source"]),
        )
        for state in rank_states:
            if float(state["classifier_threshold"]) != merged.classifier_threshold:
                raise ValueError("cotrain eval classifier thresholds differ across ranks")
            if str(state["threshold_source"]) != merged.threshold_source:
                raise ValueError("cotrain eval threshold sources differ across ranks")
            merged._wm_records.extend(list(state["wm_records"]))
            merged._classifier_records.extend(list(state["classifier_records"]))
        return merged

    @staticmethod
    def _wm_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            return {
                "wm_closed_loop_mse": None,
                "wm_closed_loop_cosine": None,
                "wm_horizon": {"mse": [], "cosine": [], "counts": []},
            }
        trajectory_mse = [sum(record["mse"]) / len(record["mse"]) for record in records]
        trajectory_cosine = [sum(record["cosine"]) / len(record["cosine"]) for record in records]
        horizon = max(len(record["mse"]) for record in records)
        horizon_mse: list[float] = []
        horizon_cosine: list[float] = []
        horizon_counts: list[int] = []
        for index in range(horizon):
            available = [record for record in records if len(record["mse"]) > index]
            horizon_counts.append(len(available))
            horizon_mse.append(
                float(sum(record["mse"][index] for record in available) / len(available))
            )
            horizon_cosine.append(
                float(sum(record["cosine"][index] for record in available) / len(available))
            )
        return {
            "wm_closed_loop_mse": float(sum(trajectory_mse) / len(trajectory_mse)),
            "wm_closed_loop_cosine": float(sum(trajectory_cosine) / len(trajectory_cosine)),
            "wm_horizon": {
                "mse": horizon_mse,
                "cosine": horizon_cosine,
                "counts": horizon_counts,
            },
        }

    def _classifier_summary(
        self,
        records: list[dict[str, Any]],
        score_key: str,
    ) -> dict[str, Any]:
        return binary_classification_metrics(
            labels=[record["success"] for record in records],
            scores=[record[score_key] for record in records],
            threshold=self.classifier_threshold,
        )

    def summarize(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            **self._wm_summary(self._wm_records),
            "classifier_threshold": self.classifier_threshold,
            "classifier_threshold_source": self.threshold_source,
            "real_classifier": self._classifier_summary(self._classifier_records, "real_score"),
            "wm_classifier": self._classifier_summary(self._classifier_records, "wm_score"),
            "trajectory_count": len(self._classifier_records),
            "wm_trajectory_count": len(self._wm_records),
        }
        task_ids = sorted(
            {int(record["task_id"]) for record in [*self._wm_records, *self._classifier_records]}
        )
        per_task: dict[str, Any] = {}
        for task_id in task_ids:
            wm_records = [record for record in self._wm_records if record["task_id"] == task_id]
            cls_records = [
                record for record in self._classifier_records if record["task_id"] == task_id
            ]
            per_task[str(task_id)] = {
                **self._wm_summary(wm_records),
                "real_classifier": self._classifier_summary(cls_records, "real_score"),
                "wm_classifier": self._classifier_summary(cls_records, "wm_score"),
                "trajectory_count": len(cls_records),
            }
        summary["per_task"] = per_task
        return summary


@dataclass
class _PendingEvalTrajectory:
    task_id: int
    reset_state_id: int
    task_description: str
    images: list[np.ndarray]
    proprio: list[np.ndarray]
    actions: list[np.ndarray]


class CotrainEvalObserver:
    """Stream physical LIBERO episodes into read-only WM/CLS diagnostics.

    ``LiberoEnv`` exposes images after its fixed 180-degree display transform.
    The observer reverses that transform and then delegates all checkpoint-specific
    image/prompt preprocessing to the restored policy.  Each completed trajectory
    is encoded, evaluated and released immediately; no replay or optimizer exists
    on this path.
    """

    def __init__(
        self,
        *,
        policy: torch.nn.Module,
        world_model: torch.nn.Module,
        classifier: torch.nn.Module,
        classifier_threshold: float,
        expected_trajectories: int,
        encode_batch_size: int,
        device: torch.device,
    ) -> None:
        if int(expected_trajectories) <= 0:
            raise ValueError("cotrain eval expected_trajectories must be positive")
        if int(encode_batch_size) <= 0:
            raise ValueError("cotrain eval encode_batch_size must be positive")
        self.policy = policy
        self.world_model = world_model
        self.classifier = classifier
        self.expected_trajectories = int(expected_trajectories)
        self.encode_batch_size = int(encode_batch_size)
        self.device = torch.device(device)
        self.accumulator = CotrainTransactionAccumulator(
            classifier_threshold=float(classifier_threshold),
            threshold_source="checkpoint",
        )
        self._buffers: list[_PendingEvalTrajectory | None] = []
        self._seen_reset_state_ids: set[int] = set()

    @property
    def pending_trajectory_count(self) -> int:
        return sum(buffer is not None for buffer in self._buffers)

    def on_reset(
        self,
        *,
        env: Any,
        obs: dict[str, Any],
        infos: Any,
        epoch: int,
    ) -> None:
        """Start one fresh physical-trajectory buffer per environment slot."""

        del infos, epoch
        if self.pending_trajectory_count:
            raise RuntimeError("cotrain eval reset would discard unfinished physical trajectories")
        num_envs = int(env.num_envs)
        task_ids = np.asarray(env.task_ids).reshape(-1)
        reset_state_ids = np.asarray(env.reset_state_ids).reshape(-1)
        descriptions = list(obs["task_descriptions"])
        if not (len(task_ids) == len(reset_state_ids) == len(descriptions) == num_envs):
            raise ValueError("LIBERO cotrain eval reset metadata is not slot-aligned")
        self._buffers = [
            _PendingEvalTrajectory(
                task_id=int(task_ids[slot]),
                reset_state_id=int(reset_state_ids[slot]),
                task_description=str(descriptions[slot]),
                images=[],
                proprio=[],
                actions=[],
            )
            for slot in range(num_envs)
        ]

    @staticmethod
    def _pre_action_observation(
        *,
        obs_before: dict[str, Any],
        obs_list: Sequence[dict[str, Any]],
        action_index: int,
    ) -> dict[str, Any]:
        if int(action_index) == 0:
            return obs_before
        return obs_list[int(action_index) - 1]

    @staticmethod
    def _episode_info_for_step(
        infos_list: Sequence[dict[str, Any]],
        action_index: int,
    ) -> dict[str, Any]:
        info = infos_list[int(action_index)]
        if "final_info" in info:
            return info["final_info"]["episode"]
        return info["episode"]

    def on_chunk(
        self,
        *,
        env: Any,
        obs_before: dict[str, Any],
        chunk_actions: np.ndarray,
        obs_list: Sequence[dict[str, Any]],
        rewards: np.ndarray,
        terms: np.ndarray,
        truncs: np.ndarray,
        infos_list: Sequence[dict[str, Any]],
        newly_done: np.ndarray,
        episode_info: dict[str, Any] | None,
        epoch: int,
        chunk_index: int,
    ) -> None:
        """Append exact pre-action states and finalize slots at first physical done."""

        del rewards, newly_done, episode_info, epoch, chunk_index
        actions = np.asarray(chunk_actions)
        terminations = np.asarray(terms, dtype=bool)
        truncations = np.asarray(truncs, dtype=bool)
        if actions.ndim != 3:
            raise ValueError("cotrain eval chunk actions must be [E,K,A]")
        if terminations.shape != actions.shape[:2] or truncations.shape != actions.shape[:2]:
            raise ValueError("cotrain eval done arrays must match [E,K]")
        if len(obs_list) != int(actions.shape[1]) or len(infos_list) != int(actions.shape[1]):
            raise ValueError("cotrain eval chunk observations must match K")
        if len(self._buffers) != int(env.num_envs):
            raise RuntimeError("cotrain eval chunk arrived before reset")

        for slot in range(int(env.num_envs)):
            buffer = self._buffers[slot]
            if buffer is None:
                continue
            for action_index in range(int(actions.shape[1])):
                physical_obs = self._pre_action_observation(
                    obs_before=obs_before,
                    obs_list=obs_list,
                    action_index=action_index,
                )
                # _wrap_obs/get_libero_image rotates exactly once. Restore raw RGB;
                # the policy extractor then applies its own checkpoint metadata.
                wrapped_image = np.asarray(physical_obs["main_images"][slot], dtype=np.uint8)
                buffer.images.append(np.ascontiguousarray(wrapped_image[::-1, ::-1]))
                buffer.proprio.append(
                    np.asarray(physical_obs["states"][slot], dtype=np.float32).copy()
                )
                buffer.actions.append(
                    np.asarray(actions[slot, action_index], dtype=np.float32).copy()
                )
                if bool(terminations[slot, action_index] or truncations[slot, action_index]):
                    step_episode = self._episode_info_for_step(
                        infos_list,
                        action_index,
                    )
                    self._finish_slot(slot=slot, episode_info=step_episode)
                    break

    @torch.no_grad()
    def _encode(self, buffer: _PendingEvalTrajectory) -> tuple[torch.Tensor, torch.Tensor]:
        transitions = [
            {
                "agentview_rgb": image,
                "state": proprio,
                "proprio": proprio,
                "task_description": buffer.task_description,
            }
            for image, proprio in zip(
                buffer.images,
                buffer.proprio,
                strict=True,
            )
        ]
        hidden_parts: list[torch.Tensor] = []
        language_embedding: torch.Tensor | None = None
        for start in range(0, len(transitions), self.encode_batch_size):
            raw_batch = self.policy.prepare_raw_batch(
                transitions[start : start + self.encode_batch_size]
            )
            prepared = {
                str(key): (value.to(self.device) if isinstance(value, torch.Tensor) else value)
                for key, value in raw_batch.items()
            }
            _actions, _log_prob, extras = self.policy({"mode": "encode_raw", **prepared})
            hidden = extras.get("hidden")
            lang_emb = extras.get("lang_emb")
            if not isinstance(hidden, torch.Tensor):
                raise TypeError("cotrain eval policy encode_raw returned no hidden")
            if not isinstance(lang_emb, torch.Tensor):
                raise TypeError("cotrain eval policy encode_raw returned no lang_emb")
            hidden_parts.append(hidden.detach())
            if language_embedding is None:
                language_embedding = lang_emb[0].detach()
        if language_embedding is None:
            raise RuntimeError("cotrain eval cannot encode an empty trajectory")
        return torch.cat(hidden_parts, dim=0), language_embedding

    def _finish_slot(self, *, slot: int, episode_info: dict[str, Any]) -> None:
        buffer = self._buffers[int(slot)]
        if buffer is None:
            return
        self._buffers[int(slot)] = None
        reset_state_id = int(np.asarray(episode_info["reset_state_id"])[slot])
        task_id = int(np.asarray(episode_info["task_id"])[slot])
        success = bool(np.asarray(episode_info["success_once"])[slot])
        if reset_state_id != buffer.reset_state_id or task_id != buffer.task_id:
            raise ValueError("cotrain eval completion metadata changed within episode")
        if reset_state_id in self._seen_reset_state_ids:
            return
        self._seen_reset_state_ids.add(reset_state_id)
        hidden, lang_emb = self._encode(buffer)
        trajectory = EncodedEvalTrajectory(
            task_id=task_id,
            reset_state_id=reset_state_id,
            success=success,
            hidden=hidden,
            actions=torch.as_tensor(np.stack(buffer.actions), dtype=torch.float32),
            proprio=torch.as_tensor(np.stack(buffer.proprio), dtype=torch.float32),
            lang_emb=lang_emb,
        )
        evaluate_encoded_cotrain_trajectory(
            world_model=self.world_model,
            classifier=self.classifier,
            trajectory=trajectory,
            accumulator=self.accumulator,
        )

    @staticmethod
    def _metrics_from_summary(
        summary: dict[str, Any],
        *,
        expected_trajectories: int,
    ) -> dict[str, Any]:
        count = int(summary["trajectory_count"])
        if count != int(expected_trajectories):
            raise RuntimeError(
                "cotrain eval trajectory count mismatch: "
                f"expected {int(expected_trajectories)}, got {count}"
            )
        if int(summary["wm_trajectory_count"]) != count:
            raise RuntimeError("cotrain eval WM and classifier trajectory counts differ")
        metrics: dict[str, Any] = {
            "eval/cotrain_trajectory_count": float(count),
            "eval/cotrain_expected_trajectories": float(expected_trajectories),
            "eval/wm_closed_loop_mse": float(summary["wm_closed_loop_mse"]),
            "eval/wm_closed_loop_cosine": float(summary["wm_closed_loop_cosine"]),
            "eval/wm_trajectory_cosine": float(summary["wm_closed_loop_cosine"]),
            "eval/classifier_threshold": float(summary["classifier_threshold"]),
            "eval_cotrain_diagnostics": summary,
        }
        for source in ("real_classifier", "wm_classifier"):
            short_source = "real" if source == "real_classifier" else "wm"
            values = summary[source]
            for name in (
                "positive_fraction",
                "predicted_positive_fraction",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "true_positives",
                "false_positives",
                "false_negatives",
                "true_negatives",
            ):
                value = values[name]
                if value is not None:
                    metrics[f"eval/classifier_{short_source}_{name}"] = float(value)
            metrics[f"eval/classifier_{short_source}_roc_auc_defined"] = float(
                bool(values["roc_auc_defined"])
            )
            metrics[f"eval/classifier_{short_source}_pr_auc_defined"] = float(
                bool(values["pr_auc_defined"])
            )
        metrics["eval/cls_trajectory_f1"] = metrics["eval/classifier_real_f1"]
        metrics["eval/cls_trajectory_accuracy"] = metrics["eval/classifier_real_accuracy"]
        return metrics

    @classmethod
    def metrics_from_rank_states(
        cls,
        states: Iterable[dict[str, Any]],
        *,
        expected_trajectories: int,
    ) -> dict[str, Any]:
        """Recompute all diagnostics from raw rank records."""

        summary = CotrainTransactionAccumulator.from_rank_states(states).summarize()
        return cls._metrics_from_summary(
            summary,
            expected_trajectories=int(expected_trajectories),
        )

    def rank_payload(self) -> dict[str, Any]:
        """Return local validation metadata and raw records for one collective."""

        return {
            "pending_trajectory_count": self.pending_trajectory_count,
            "expected_trajectories": self.expected_trajectories,
            "state": self.accumulator.rank_state(),
        }

    @classmethod
    def metrics_from_rank_payloads(
        cls,
        payloads: Iterable[dict[str, Any]],
        *,
        expected_trajectories: int,
    ) -> dict[str, Any]:
        """Validate every local shard after gather, then recompute global metrics."""

        rank_payloads = list(payloads)
        states: list[dict[str, Any]] = []
        for rank, payload in enumerate(rank_payloads):
            pending = int(payload["pending_trajectory_count"])
            if pending:
                raise RuntimeError(
                    f"cotrain eval rank {rank} ended with {pending} unfinished trajectories"
                )
            state = dict(payload["state"])
            expected = int(payload["expected_trajectories"])
            classifier_count = len(state["classifier_records"])
            world_model_count = len(state["wm_records"])
            if classifier_count != expected:
                raise RuntimeError(
                    "cotrain eval rank "
                    f"{rank} trajectory count mismatch: expected {expected}, got {classifier_count}"
                )
            if world_model_count != classifier_count:
                raise RuntimeError(
                    f"cotrain eval rank {rank} WM and classifier trajectory counts differ"
                )
            states.append(state)
        return cls.metrics_from_rank_states(
            states,
            expected_trajectories=int(expected_trajectories),
        )

    def finalize_metrics(self) -> dict[str, Any]:
        """Validate the fixed protocol and return flat plus detailed metrics."""

        return self.metrics_from_rank_payloads(
            [self.rank_payload()],
            expected_trajectories=self.expected_trajectories,
        )


@torch.no_grad()
def classifier_trajectory_score(
    classifier: torch.nn.Module,
    *,
    hidden: torch.Tensor,
    task_id: int,
    threshold: float,
    proprio: torch.Tensor | None = None,
    lang_emb: torch.Tensor | None = None,
) -> float:
    """Return the maximum success probability over one full trajectory."""

    device = _model_device(classifier)
    dtype = _model_dtype(classifier)
    video = torch.as_tensor(hidden).to(device=device, dtype=dtype).unsqueeze(0)
    kwargs: dict[str, Any] = {}
    if bool(getattr(classifier, "supports_task_conditioning", False)):
        kwargs["task_ids"] = torch.tensor([int(task_id)], device=device)
    if bool(getattr(classifier, "supports_proprio_conditioning", False)):
        if proprio is None:
            raise ValueError("classifier evaluation requires trajectory proprio")
        kwargs["proprio"] = (
            torch.as_tensor(proprio)
            .to(
                device=device,
                dtype=dtype,
            )
            .unsqueeze(0)
        )
    if bool(getattr(classifier, "supports_language_conditioning", False)):
        if lang_emb is None:
            raise ValueError("classifier evaluation requires trajectory language")
        language = torch.as_tensor(lang_emb).to(device=device, dtype=dtype)
        kwargs["lang_emb"] = language.unsqueeze(0) if language.ndim == 1 else language
    result = classifier.predict_success(
        video,
        threshold=float(threshold),
        stride=1,
        min_steps=0,
        pre_pooled=False,
        **kwargs,
    )
    score = torch.as_tensor(result["score"]).detach().float().reshape(-1)
    if score.numel() != 1:
        raise ValueError("single-trajectory classifier must return one score")
    return float(score.item())


@torch.no_grad()
def evaluate_encoded_cotrain_trajectory(
    *,
    world_model: torch.nn.Module,
    classifier: torch.nn.Module,
    trajectory: EncodedEvalTrajectory,
    accumulator: CotrainTransactionAccumulator,
) -> ClosedLoopWorldModelResult:
    """Evaluate one trajectory without mutating training data or thresholds."""

    wm_result = closed_loop_world_model_trajectory(world_model, trajectory)
    accumulator.add_world_model_metrics(
        task_id=trajectory.task_id,
        mse_by_horizon=wm_result.mse_by_horizon.detach().float().cpu().tolist(),
        cosine_by_horizon=wm_result.cosine_by_horizon.detach().float().cpu().tolist(),
    )
    real_score = classifier_trajectory_score(
        classifier,
        hidden=trajectory.hidden,
        proprio=trajectory.proprio,
        lang_emb=trajectory.lang_emb,
        task_id=trajectory.task_id,
        threshold=accumulator.classifier_threshold,
    )

    history_length = int(world_model.num_hist)
    predicted_steps = int(wm_result.predicted_hidden.shape[0])
    wm_hidden = torch.cat(
        [
            torch.as_tensor(trajectory.hidden)[:history_length].to(
                device=wm_result.predicted_hidden.device,
                dtype=wm_result.predicted_hidden.dtype,
            ),
            wm_result.predicted_hidden,
        ],
        dim=0,
    )
    wm_proprio = None
    if trajectory.proprio is not None:
        real_proprio = torch.as_tensor(trajectory.proprio)
        predicted_proprio = wm_result.predicted_proprio
        if predicted_proprio is not None:
            wm_proprio = torch.cat(
                [
                    real_proprio[:history_length].to(
                        device=predicted_proprio.device,
                        dtype=predicted_proprio.dtype,
                    ),
                    predicted_proprio,
                ],
                dim=0,
            )
        else:
            # Proprio is an observed classifier sidecar when the WM does not
            # expose a reconstruction head. Keep it time-aligned while judging
            # only the recursively predicted visual-token path.
            wm_proprio = real_proprio[: history_length + predicted_steps]
    wm_score = classifier_trajectory_score(
        classifier,
        hidden=wm_hidden,
        proprio=wm_proprio,
        lang_emb=trajectory.lang_emb,
        task_id=trajectory.task_id,
        threshold=accumulator.classifier_threshold,
    )
    accumulator.add_classifier_result(
        task_id=trajectory.task_id,
        success=trajectory.success,
        real_score=real_score,
        wm_score=wm_score,
    )
    return wm_result


__all__ = [
    "ClosedLoopWorldModelResult",
    "CotrainEvalObserver",
    "CotrainTransactionAccumulator",
    "EncodedEvalTrajectory",
    "binary_classification_metrics",
    "classifier_trajectory_score",
    "closed_loop_world_model_trajectory",
    "encoded_eval_trajectory_from_real",
    "evaluate_encoded_cotrain_trajectory",
]
