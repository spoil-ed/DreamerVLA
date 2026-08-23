from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dreamervla.envs.world_model import latent_world_model_env
from dreamervla.envs.world_model.latent_world_model_env import LatentWorldModelEnv


class _TinyWM(torch.nn.Module):
    def forward(self, batch):
        latent = batch["latent"]
        action = batch["action"]
        return latent + action[..., : latent.shape[-1]]


class _TinyClassifier(torch.nn.Module):
    def forward(self, latent):
        return {"score": latent.sum(dim=-1, keepdim=True)}


def test_frozen_world_model_env_supports_phase_offload() -> None:
    env = LatentWorldModelEnv(
        _TinyWM(),
        _TinyClassifier(),
        latent_dim=2,
        action_dim=2,
        device="cpu",
    )

    env.offload_model()
    assert env._models_offloaded is True
    env.reload_model()
    assert env._models_offloaded is False
    obs, _ = env.reset()
    assert obs["latent"].shape == (2,)


class _ChunkWM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    def forward(self, batch):
        self.calls.append(dict(batch))
        assert batch["mode"] == "predict_next_chunk"
        latent = batch["latent"].reshape(batch["actions"].shape[0], -1)
        increments = batch["actions"][..., : latent.shape[-1]].cumsum(dim=1)
        hidden_seq = latent[:, None] + increments
        return {
            "hidden": hidden_seq[:, -1],
            "hidden_seq": hidden_seq,
        }


class _StatefulChunkWM(torch.nn.Module):
    """WM double whose second prediction depends on exact returned histories."""

    num_hist = 3

    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[dict[str, torch.Tensor]] = []

    def forward(self, batch):
        latent = batch["latent"]
        assert isinstance(latent, dict)
        self.inputs.append(
            {
                key: value.detach().cpu().clone()
                for key, value in latent.items()
                if isinstance(value, torch.Tensor)
            }
        )
        hidden = latent["hidden"].reshape(batch["actions"].shape[0], -1)
        increments = batch["actions"][..., : hidden.shape[-1]].cumsum(dim=1)
        hidden_seq = hidden[:, None] + increments
        returned_history = torch.stack(
            [hidden_seq[:, -1] + offset for offset in (10.0, 20.0, 30.0)],
            dim=1,
        )
        returned_actions = torch.stack(
            [batch["actions"][:, -1] + offset for offset in (1.0, 2.0, 3.0)],
            dim=1,
        )
        return {
            "hidden": hidden_seq[:, -1],
            "hidden_seq": hidden_seq,
            "history": returned_history,
            "actions": returned_actions,
        }


class _AutocastAwareChunkWM(_ChunkWM):
    def __init__(self) -> None:
        super().__init__()
        self.autocast_enabled: list[bool] = []

    def forward(self, batch):
        self.autocast_enabled.append(torch.is_autocast_enabled("cpu"))
        return super().forward(batch)


class _AutocastAwareClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.autocast_enabled: list[bool] = []

    def forward(self, latent):
        self.autocast_enabled.append(torch.is_autocast_enabled("cpu"))
        return latent.sum(dim=-1, keepdim=True)


class _InferenceModeAwareChunkWM(_ChunkWM):
    def __init__(self) -> None:
        super().__init__()
        self.inference_mode_enabled: list[bool] = []

    def forward(self, batch):
        self.inference_mode_enabled.append(torch.is_inference_mode_enabled())
        return super().forward(batch)


class _InferenceModeAwareClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inference_mode_enabled: list[bool] = []

    def forward(self, latent):
        self.inference_mode_enabled.append(torch.is_inference_mode_enabled())
        return latent.sum(dim=-1, keepdim=True)


class _ZeroTwoClassClassifier(torch.nn.Module):
    def forward(self, latent):
        return torch.zeros(latent.shape[0], 2, device=latent.device)


class _ZeroOneLogitClassifier(torch.nn.Module):
    def forward(self, latent):
        return torch.zeros(latent.shape[0], 1, device=latent.device)


class _MalformedDictClassifier(torch.nn.Module):
    def forward(self, latent):
        return {"unknown": torch.zeros(latent.shape[0], 1, device=latent.device)}


class _WindowCaptureClassifier(torch.nn.Module):
    def __init__(self, window: int) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(window=int(window))
        self.windows: list[torch.Tensor] = []

    def forward(self, latent_window, *, task_ids=None, proprio=None, lang_emb=None):
        del task_ids, proprio, lang_emb
        self.windows.append(latent_window.detach().cpu().clone())
        return torch.zeros(latent_window.shape[0], 1, device=latent_window.device)


class _ChunkWindowCaptureClassifier(torch.nn.Module):
    def __init__(self, window: int) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(
            window=int(window),
            granularity="chunk",
            chunk_pool="last",
        )
        self.windows: list[torch.Tensor] = []

    def forward(self, latent_window, *, task_ids=None, proprio=None, lang_emb=None):
        del task_ids, proprio, lang_emb
        self.windows.append(latent_window.detach().cpu().clone())
        return {"score": latent_window[:, -1].sum(dim=-1, keepdim=True)}


def test_latent_world_model_env_step_returns_env_tuple():
    env = LatentWorldModelEnv(
        world_model=_TinyWM(),
        classifier=_TinyClassifier(),
        latent_dim=2,
        action_dim=7,
        success_threshold=0.5,
    )

    obs, info = env.reset(task_id=1, episode_id=2)
    assert "latent" in obs
    assert info["task_id"] == 1

    next_obs, reward, terminated, truncated, info = env.step(np.ones(7, dtype=np.float32))

    assert "latent" in next_obs
    assert reward > 0.0
    assert terminated is True
    assert truncated is False
    assert info["wm_version"] == 0
    assert info["classifier_version"] == 0


def test_latent_world_model_env_loads_independent_versions():
    env = LatentWorldModelEnv(
        world_model=_TinyWM(),
        classifier=_TinyClassifier(),
        latent_dim=2,
        action_dim=7,
    )

    env.load_world_model_state({}, version=5)
    env.load_classifier_state({}, version=7)

    assert env.wm_version == 5
    assert env.classifier_version == 7


def test_latent_world_model_env_converts_two_class_logits_to_success_probability():
    env = LatentWorldModelEnv(
        world_model=_TinyWM(),
        classifier=_ZeroTwoClassClassifier(),
        latent_dim=2,
        action_dim=2,
        success_threshold=0.5,
    )

    env.reset()
    _next_obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))

    assert reward == pytest.approx(0.5)
    assert info["success_score"] == pytest.approx(0.5)
    assert terminated is True
    assert truncated is False


def test_latent_world_model_env_converts_single_logit_to_success_probability():
    env = LatentWorldModelEnv(
        world_model=_TinyWM(),
        classifier=_ZeroOneLogitClassifier(),
        latent_dim=2,
        action_dim=2,
        success_threshold=0.5,
    )

    env.reset()
    _next_obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))

    assert reward == pytest.approx(0.5)
    assert info["success_score"] == pytest.approx(0.5)
    assert terminated is True
    assert truncated is False


def test_latent_world_model_env_rejects_unknown_classifier_dict_output():
    env = LatentWorldModelEnv(
        world_model=_TinyWM(),
        classifier=_MalformedDictClassifier(),
        latent_dim=2,
        action_dim=2,
    )

    env.reset()
    with pytest.raises(ValueError, match="classifier output dict"):
        env.step(np.zeros(2, dtype=np.float32))


def test_latent_world_model_env_batches_independent_slots() -> None:
    env = LatentWorldModelEnv(
        world_model=_TinyWM(),
        classifier=_TinyClassifier(),
        latent_dim=2,
        action_dim=2,
        success_threshold=10.0,
        num_envs=3,
    )

    obs0, _ = env.reset_slot(0, task_id=0, episode_id=0)
    obs1, _ = env.reset_slot(1, task_id=1, episode_id=11)
    obs2, _ = env.reset_slot(2, task_id=2, episode_id=22)

    assert obs0["latent"].shape == (2,)
    assert obs1["episode_id"] == 11
    assert obs2["task_id"] == 2

    next_obs0, reward0, done0, truncated0, info0 = env.step_slot(
        0, np.array([1.0, 0.0], dtype=np.float32)
    )
    next_obs1, reward1, done1, truncated1, info1 = env.step_slot(
        1, np.array([0.0, 2.0], dtype=np.float32)
    )

    assert next_obs0["latent"].tolist() == [1.0, 0.0]
    assert next_obs1["latent"].tolist() == [0.0, 2.0]
    assert reward0 == 1.0
    assert reward1 == 2.0
    assert done0 is False
    assert done1 is False
    assert truncated0 is False
    assert truncated1 is False
    assert info0["slot_id"] == 0
    assert info1["slot_id"] == 1


def test_latent_world_model_env_chunk_step_batch_uses_one_wm_call() -> None:
    wm = _ChunkWM()
    env = LatentWorldModelEnv(
        world_model=wm,
        classifier=_TinyClassifier(),
        latent_dim=2,
        action_dim=2,
        success_threshold=99.0,
        num_envs=2,
    )
    env.reset_slot(0, task_id=0, episode_id=0)
    env.reset_slot(1, task_id=1, episode_id=10)

    observations, rewards, terminations, truncations, infos = env.chunk_step_batch(
        np.array(
            [
                [[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]],
                [[0.0, 1.0], [2.0, 0.0], [0.0, 3.0]],
            ],
            dtype=np.float32,
        ),
        env_ids=[0, 1],
    )

    assert len(wm.calls) == 1
    assert wm.calls[0]["actions"].shape == (2, 3, 2)
    assert np.asarray(rewards).shape == (2, 3)
    assert observations[0]["step"] == 3
    assert observations[1]["step"] == 3
    assert observations[0]["latent"].tolist() == [4.0, 2.0]
    assert observations[1]["latent"].tolist() == [2.0, 4.0]
    assert np.asarray(terminations).shape == (2, 3)
    assert np.asarray(truncations).shape == (2, 3)
    assert infos[0]["slot_id"] == 0
    assert infos[1]["slot_id"] == 1
    assert env.get_metrics()["wm_forward_calls"] == 1.0


def test_chunk_step_batch_carries_returned_wm_history_across_chunks() -> None:
    wm = _StatefulChunkWM()
    env = LatentWorldModelEnv(
        world_model=wm,
        classifier=None,
        latent_dim=2,
        action_dim=2,
        success_threshold=99.0,
        num_envs=1,
    )
    env.reset_slot(0, task_id=0, episode_id=0)
    first_actions = np.array([[[1.0, 0.0], [0.0, 2.0]]], dtype=np.float32)
    env.chunk_step_batch(first_actions, env_ids=[0])

    returned_history = torch.stack(
        [torch.tensor([[1.0, 2.0]]) + offset for offset in (10.0, 20.0, 30.0)],
        dim=1,
    )
    returned_actions = torch.stack(
        [torch.tensor([[0.0, 2.0]]) + offset for offset in (1.0, 2.0, 3.0)],
        dim=1,
    )
    env.chunk_step_batch(
        np.array([[[3.0, 4.0], [5.0, 6.0]]], dtype=np.float32),
        env_ids=[0],
    )

    assert len(wm.inputs) == 2
    torch.testing.assert_close(wm.inputs[1]["history"], returned_history)
    torch.testing.assert_close(wm.inputs[1]["actions"], returned_actions)
    torch.testing.assert_close(wm.inputs[1]["hidden"], returned_history[:, -1] - 30.0)


def test_latent_world_model_env_chunk_classifier_uses_rolling_history() -> None:
    classifier = _WindowCaptureClassifier(window=3)
    env = LatentWorldModelEnv(
        world_model=_ChunkWM(),
        classifier=classifier,
        latent_dim=2,
        action_dim=2,
        success_threshold=99.0,
        num_envs=1,
    )
    env.reset_slot(0, task_id=0, episode_id=0)

    env.chunk_step_batch(
        np.array([[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]], dtype=np.float32),
        env_ids=[0],
    )

    assert len(classifier.windows) == 1
    assert classifier.windows[0].tolist() == [
        [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]],
        [[1.0, 0.0], [1.0, 2.0], [4.0, 2.0]],
    ]


def test_chunk_granularity_classifier_scores_once_per_policy_chunk() -> None:
    classifier = _ChunkWindowCaptureClassifier(window=3)
    env = LatentWorldModelEnv(
        world_model=_ChunkWM(),
        classifier=classifier,
        latent_dim=2,
        action_dim=2,
        success_threshold=99.0,
        num_envs=1,
    )
    env.reset_slot(0, task_id=0, episode_id=0)

    _observations, rewards, _terminations, _truncations, infos = env.chunk_step_batch(
        np.array(
            [[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]],
            dtype=np.float32,
        ),
        env_ids=[0],
    )

    assert len(classifier.windows) == 1
    assert classifier.windows[0].tolist() == [
        [[0.0, 0.0], [0.0, 0.0], [4.0, 2.0]],
    ]
    assert rewards.tolist() == [[0.0, 0.0, 6.0]]
    assert infos[0]["classifier_evaluations"] == 1
    assert infos[0]["classifier_success_evaluations"] == 0
    assert env.get_metrics()["score_count"] == 1.0


def test_classifier_temporal_windows_do_not_cat_or_stack_slot_histories(
    monkeypatch,
) -> None:
    env = LatentWorldModelEnv(
        world_model=_ChunkWM(),
        classifier=_WindowCaptureClassifier(window=3),
        latent_dim=2,
        action_dim=2,
        num_envs=2,
    )
    cat_calls = 0
    stack_calls = 0
    original_cat = torch.cat
    original_stack = torch.stack

    def tracked_cat(*args, **kwargs):
        nonlocal cat_calls
        cat_calls += 1
        return original_cat(*args, **kwargs)

    def tracked_stack(*args, **kwargs):
        nonlocal stack_calls
        stack_calls += 1
        return original_stack(*args, **kwargs)

    monkeypatch.setattr(torch, "cat", tracked_cat)
    monkeypatch.setattr(torch, "stack", tracked_stack)

    windows, _proprio, updates, _proprio_updates = env._classifier_temporal_windows(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        window=3,
        slots=[0, 1],
    )

    assert windows.tolist() == [
        [[0.0, 0.0], [0.0, 0.0], [1.0, 2.0]],
        [[0.0, 0.0], [0.0, 0.0], [3.0, 4.0]],
    ]
    assert set(updates) == {0, 1}
    assert cat_calls == 0
    assert stack_calls == 0


def test_latent_world_model_env_reports_classifier_score_distribution() -> None:
    env = LatentWorldModelEnv(
        world_model=_ChunkWM(),
        classifier=_TinyClassifier(),
        latent_dim=2,
        action_dim=2,
        success_threshold=99.0,
        num_envs=2,
    )
    env.reset_slot(0, task_id=0, episode_id=0)
    env.reset_slot(1, task_id=1, episode_id=10)

    env.chunk_step_batch(
        np.array(
            [
                [[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]],
                [[0.0, 1.0], [2.0, 0.0], [0.0, 3.0]],
            ],
            dtype=np.float32,
        ),
        env_ids=[0, 1],
    )

    metrics = env.get_metrics(reset=True)

    assert metrics["score_mean"] == pytest.approx(20.0 / 6.0)
    assert metrics["score_p50"] == pytest.approx(3.0)
    assert metrics["score_p90"] == pytest.approx(6.0)
    assert metrics["score_max"] == pytest.approx(6.0)

    reset_metrics = env.get_metrics()
    assert reset_metrics["score_mean"] == 0.0
    assert reset_metrics["score_p50"] == 0.0
    assert reset_metrics["score_p90"] == 0.0
    assert reset_metrics["score_max"] == 0.0


def test_latent_world_model_env_chunk_step_batch_marks_done_only_on_final_step() -> None:
    env = LatentWorldModelEnv(
        world_model=_ChunkWM(),
        classifier=_TinyClassifier(),
        latent_dim=2,
        action_dim=2,
        success_threshold=0.5,
        max_episode_steps=99,
        num_envs=1,
    )
    env.reset_slot(0, task_id=0, episode_id=0)

    _observations, _rewards, terminations, truncations, infos = env.chunk_step_batch(
        np.array([[[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
        env_ids=[0],
    )

    assert terminations.tolist() == [[False, False, True]]
    assert truncations.tolist() == [[False, False, False]]
    assert infos[0]["success"] is True

    timeout_env = LatentWorldModelEnv(
        world_model=_ChunkWM(),
        classifier=_TinyClassifier(),
        latent_dim=2,
        action_dim=2,
        success_threshold=99.0,
        max_episode_steps=2,
        num_envs=1,
    )
    timeout_env.reset_slot(0, task_id=0, episode_id=0)

    _observations, _rewards, terminations, truncations, infos = timeout_env.chunk_step_batch(
        np.array([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
        env_ids=[0],
    )

    assert terminations.tolist() == [[False, False, False]]
    assert truncations.tolist() == [[False, False, True]]
    assert infos[0]["success"] is False


def test_latent_world_model_env_chunk_step_batch_avoids_unconditional_action_copy(
    monkeypatch,
) -> None:
    class _NoCopyArray(np.ndarray):
        def copy(self, *_args, **_kwargs):  # type: ignore[override]
            raise AssertionError("chunk actions should not be copied unconditionally")

    original_asarray = np.asarray

    def asarray_no_copy(value, *args, **kwargs):
        arr = original_asarray(value, *args, **kwargs)
        if arr.shape == (1, 3, 2):
            return arr.view(_NoCopyArray)
        return arr

    monkeypatch.setattr(latent_world_model_env.np, "asarray", asarray_no_copy)
    env = LatentWorldModelEnv(
        world_model=_ChunkWM(),
        classifier=_TinyClassifier(),
        latent_dim=2,
        action_dim=2,
        success_threshold=99.0,
        max_episode_steps=99,
        num_envs=1,
    )
    env.reset_slot(0, task_id=0, episode_id=0)

    _observations, rewards, terminations, _truncations, _infos = env.chunk_step_batch(
        original_asarray([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]], dtype=np.float32),
        env_ids=[0],
    )

    assert rewards.shape == (1, 3)
    assert terminations.tolist() == [[False, False, False]]


def test_latent_world_model_env_chunk_step_batch_keeps_wm_action_on_cpu_boundary(
    monkeypatch,
) -> None:
    original_cpu = torch.Tensor.cpu

    def cpu_guard(tensor, *args, **kwargs):
        if tuple(tensor.shape) == (2,):
            raise AssertionError("wm_action should not be copied back from action_t")
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu_guard)
    env = LatentWorldModelEnv(
        world_model=_ChunkWM(),
        classifier=_TinyClassifier(),
        latent_dim=2,
        action_dim=2,
        success_threshold=99.0,
        max_episode_steps=99,
        num_envs=1,
        observation_format="tensor",
    )
    env.reset_slot(0, task_id=0, episode_id=0)

    _observations, _rewards, _terminations, _truncations, infos = env.chunk_step_batch(
        np.array([[[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]]], dtype=np.float32),
        env_ids=[0],
    )

    assert np.asarray(infos[0]["wm_action"], dtype=np.float32).tolist() == [3.0, 4.0]


def test_latent_world_model_env_chunk_step_batch_uses_configured_autocast() -> None:
    wm = _AutocastAwareChunkWM()
    env = LatentWorldModelEnv(
        world_model=wm,
        classifier=_TinyClassifier(),
        latent_dim=2,
        action_dim=2,
        success_threshold=99.0,
        max_episode_steps=99,
        num_envs=1,
        inference_dtype="bf16",
    )
    env.reset_slot(0, task_id=0, episode_id=0)

    env.chunk_step_batch(
        np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        env_ids=[0],
    )

    assert wm.autocast_enabled == [True]


def test_latent_world_model_env_chunk_step_batch_autocasts_classifier() -> None:
    classifier = _AutocastAwareClassifier()
    env = LatentWorldModelEnv(
        world_model=_ChunkWM(),
        classifier=classifier,
        latent_dim=2,
        action_dim=2,
        success_threshold=99.0,
        max_episode_steps=99,
        num_envs=1,
        inference_dtype="bf16",
    )
    env.reset_slot(0, task_id=0, episode_id=0)

    env.chunk_step_batch(
        np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        env_ids=[0],
    )

    assert classifier.autocast_enabled == [True]


def test_latent_world_model_env_chunk_step_batch_uses_inference_mode() -> None:
    wm = _InferenceModeAwareChunkWM()
    classifier = _InferenceModeAwareClassifier()
    env = LatentWorldModelEnv(
        world_model=wm,
        classifier=classifier,
        latent_dim=2,
        action_dim=2,
        success_threshold=99.0,
        max_episode_steps=99,
        num_envs=1,
    )
    env.reset_slot(0, task_id=0, episode_id=0)

    env.chunk_step_batch(
        np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        env_ids=[0],
    )

    assert wm.inference_mode_enabled == [True]
    assert classifier.inference_mode_enabled == [True]


def test_latent_world_model_env_can_return_tensor_observation_snapshots() -> None:
    env = LatentWorldModelEnv(
        world_model=_TinyWM(),
        classifier=_TinyClassifier(),
        latent_dim=2,
        action_dim=2,
        success_threshold=99.0,
        observation_format="tensor",
    )

    obs, _info = env.reset()

    assert isinstance(obs["latent"], torch.Tensor)
    obs["latent"][0] = 42.0

    next_obs, _reward, _terminated, _truncated, _info = env.step(np.zeros(2, dtype=np.float32))

    assert isinstance(next_obs["latent"], torch.Tensor)
    assert next_obs["latent"].tolist() == [0.0, 0.0]


def test_latent_world_model_env_tensor_observation_uses_inference_dtype() -> None:
    env = LatentWorldModelEnv(
        world_model=_ChunkWM(),
        classifier=_TinyClassifier(),
        latent_dim=2,
        action_dim=2,
        lang_dim=2,
        initial_lang_emb=np.ones(2, dtype=np.float32),
        success_threshold=99.0,
        inference_dtype="bf16",
        observation_format="tensor",
    )

    obs, _info = env.reset()

    assert obs["latent"].dtype == torch.bfloat16
    assert obs["lang_emb"].dtype == torch.bfloat16

    observations, _rewards, _terminations, _truncations, _infos = env.chunk_step_batch(
        np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        env_ids=[0],
    )

    assert observations[0]["latent"].dtype == torch.bfloat16
    assert observations[0]["lang_emb"].dtype == torch.bfloat16


def test_classifier_history_uses_world_model_inference_dtype() -> None:
    classifier = _ChunkWindowCaptureClassifier(window=3)
    env = LatentWorldModelEnv(
        world_model=_ChunkWM(),
        classifier=classifier,
        latent_dim=2,
        action_dim=2,
        num_envs=2,
        inference_dtype="bf16",
    )

    env.reset_batch([0, 1], [0, 1])
    env.chunk_step_batch(
        np.ones((2, 2, 2), dtype=np.float32),
        env_ids=[0, 1],
    )

    assert env._classifier_latent_history.dtype == torch.bfloat16
    assert classifier.windows[0].dtype == torch.bfloat16


def test_latent_world_model_env_batches_observation_device_to_host_copy(
    monkeypatch,
) -> None:
    env = LatentWorldModelEnv(
        world_model=_ChunkWM(),
        classifier=_TinyClassifier(),
        latent_dim=2,
        action_dim=2,
        num_envs=3,
        observation_format="tensor",
    )
    original = latent_world_model_env._cpu_tensor_snapshot
    copied_shapes: list[tuple[int, ...]] = []

    def tracked_snapshot(value, *, dtype=torch.float32):
        copied_shapes.append(tuple(value.shape))
        return original(value, dtype=dtype)

    monkeypatch.setattr(
        latent_world_model_env,
        "_cpu_tensor_snapshot",
        tracked_snapshot,
    )

    observations = env._obs_batch([0, 1, 2])

    assert len(observations) == 3
    assert copied_shapes == [(3, 2)]


def test_latent_world_model_env_batches_reset_observation_copy(monkeypatch) -> None:
    env = LatentWorldModelEnv(
        world_model=_ChunkWM(),
        classifier=_TinyClassifier(),
        latent_dim=2,
        action_dim=2,
        num_envs=3,
        observation_format="tensor",
    )
    original = latent_world_model_env._cpu_tensor_snapshot
    copied_shapes: list[tuple[int, ...]] = []

    def tracked_snapshot(value, *, dtype=torch.float32):
        copied_shapes.append(tuple(value.shape))
        return original(value, dtype=dtype)

    monkeypatch.setattr(
        latent_world_model_env,
        "_cpu_tensor_snapshot",
        tracked_snapshot,
    )

    observations, infos = env.reset_batch([0, 1, 2], [10, 11, 12])

    assert len(observations) == len(infos) == 3
    assert copied_shapes == [(3, 2)]


def test_latent_world_model_env_config_modules_make_replay_transition():
    env = LatentWorldModelEnv(
        world_model={
            "target": "dreamervla.workers.actor._test_models:TinyLumosWorldModel",
            "kwargs": {"hidden_dim": 4, "action_dim": 7},
        },
        classifier={
            "target": "dreamervla.workers.actor._test_models:TinySuccessClassifier",
            "kwargs": {"hidden_dim": 4, "window": 3},
        },
        latent_dim=4,
        action_dim=7,
        image_shape=(4, 4, 3),
        max_episode_steps=3,
    )

    obs, _info = env.reset(task_id=3, episode_id=4)
    action = np.zeros(7, dtype=np.float32)
    _next_obs, reward, terminated, truncated, info = env.step(action)
    transition = env.make_transition(obs, action, reward, terminated, truncated, info)

    assert transition["obs_embedding"].shape == (4,)
    assert transition["wm_action"].shape == (7,)
    assert transition["image"].shape == (4, 4, 3)
    assert transition["task_id"] == 3
    assert transition["episode_id"] == 4


class _NoChunkWM(torch.nn.Module):
    def forward(self, batch):
        if batch.get("mode") == "predict_next_chunk":
            raise NotImplementedError("predict_next_chunk not supported")
        latent = batch["latent"]
        action = batch["action"]
        return latent + action[..., : latent.shape[-1]]


def test_chunk_step_batch_fallback_warns_once(caplog) -> None:
    env = LatentWorldModelEnv(
        world_model=_NoChunkWM(),
        classifier=_TinyClassifier(),
        latent_dim=2,
        action_dim=2,
        success_threshold=99.0,
        num_envs=1,
    )
    env.reset_slot(0, task_id=0, episode_id=0)
    actions = np.zeros((1, 2, 2), dtype=np.float32)

    with caplog.at_level("WARNING"):
        env.chunk_step_batch(actions, env_ids=[0])
        env.chunk_step_batch(actions, env_ids=[0])

    fallback_warnings = [record for record in caplog.records if "chunk mode" in record.getMessage()]
    assert len(fallback_warnings) == 1
