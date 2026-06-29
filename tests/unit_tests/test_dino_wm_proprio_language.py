from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from dreamervla.algorithms.dreamervla import _flatten_last_steps, _flatten_strided_steps
from dreamervla.dataset.pixel_hidden_sequence_dataset import PixelHiddenSequenceDataset
from dreamervla.dataset.pixel_sequence_dataset import PixelSequenceDataset

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")
QB = [
    "experiment=oft_world_model_dinowm_chunk",
    "task=openvla_onetraj_libero",
    "worldmodel=openvla_oft_input_token_chunk",
]
SMALL_WM = [
    "world_model.token_count=8",
    "world_model.token_dim=16",
    "world_model.obs_dim=128",
    "world_model.proprio_dim=8",
    "world_model.proprio_emb_dim=4",
    "world_model.num_proprio_repeat=1",
    "world_model.lang_dim=12",
    "world_model.lang_emb_dim=6",
    "world_model.num_lang_repeat=1",
    "world_model.action_emb_dim=3",
    "world_model.num_action_repeat=1",
    "world_model.model_dim=29",
    "world_model.depth=1",
    "world_model.heads=1",
    "world_model.dim_head=8",
    "world_model.mlp_dim=32",
    "world_model.dropout=0.0",
    "world_model.reward_loss_scale=0.0",
]


def _qb_cfg(overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="train", overrides=[*QB, *(overrides or [])])


def _qb_world_model():
    cfg = _qb_cfg(SMALL_WM)
    return instantiate(cfg.world_model), cfg


def test_query_before_cosine_off_and_no_multichunk() -> None:
    wm, cfg = _qb_world_model()
    assert float(cfg.world_model.cosine_loss_scale) == 0.0
    assert int(cfg.world_model.chunk_rollout_chunks) == 1
    assert float(cfg.world_model.chunk_rollout_loss_scale) == 0.0

    pred = torch.randn(2, 3, 4, wm.obs_token_dim)
    tgt = torch.randn(2, 3, 4, wm.obs_token_dim)
    loss, mse, cosine = wm._hidden_loss_terms(pred, tgt)

    assert torch.allclose(loss, wm.hidden_loss_scale * mse)
    assert cosine.numel() == 1


def test_observation_tokens_concat_proprio() -> None:
    wm, _ = _qb_world_model()
    assert wm.proprio_condition_dim == 4
    assert wm.obs_token_dim == wm.token_dim + 4
    bsz, steps, slots = 2, 3, wm.token_count
    vision = torch.randn(bsz, steps, slots, wm.token_dim)
    proprio = torch.randn(bsz, steps, wm.proprio_dim)

    obs = wm._observation_tokens(vision, proprio)

    assert obs.shape == (bsz, steps, slots, wm.obs_token_dim)
    assert torch.allclose(obs[..., : wm.token_dim], vision)
    assert torch.allclose(obs[:, :, 0, wm.token_dim :], obs[:, :, 1, wm.token_dim :])


def test_condition_tokens_layout_and_model_dim() -> None:
    wm, cfg = _qb_world_model()
    assert wm.lang_condition_dim == 6
    assert wm.model_dim == wm.obs_token_dim + wm.lang_condition_dim + wm.action_condition_dim
    assert wm.model_dim == int(cfg.world_model.model_dim)
    bsz, steps, slots = 2, 3, wm.token_count
    obs = torch.randn(bsz, steps, slots, wm.obs_token_dim)
    lang = torch.randn(bsz, wm.lang_dim)
    act = torch.randn(bsz, steps, wm.action_dim)

    z = wm._condition_tokens(obs, lang, act)

    assert z.shape == (bsz, steps, slots, wm.model_dim)
    assert torch.allclose(z[..., : wm.obs_token_dim], obs)
    lo = wm.obs_token_dim
    hi = wm.obs_token_dim + wm.lang_condition_dim
    assert torch.allclose(z[:, 0, 0, lo:hi], z[:, 2, 5 % slots, lo:hi])


def test_predict_next_chunk_threads_proprio_and_lang() -> None:
    wm, _ = _qb_world_model()
    wm.eval()
    bsz, hist, slots, chunk = 2, wm.num_hist, wm.token_count, wm.chunk_size
    history = torch.randn(bsz, hist, slots, wm.obs_token_dim)
    actions = torch.zeros(bsz, hist, wm.action_dim)
    lang = torch.randn(bsz, wm.lang_dim)
    latent = {
        "hidden": history[:, -1],
        "history": history,
        "actions": actions,
        "lang": lang,
    }

    out = wm.predict_next_chunk(latent, torch.zeros(bsz, chunk, wm.action_dim))

    assert out["hidden_seq"].shape == (bsz, chunk, slots, wm.obs_token_dim)
    assert out["proprio_seq"].shape == (bsz, chunk, wm.proprio_dim)
    assert out["history"].shape == (bsz, hist, slots, wm.obs_token_dim)
    assert out["lang"] is lang


def test_predict_next_forward_threads_batch_lang_emb() -> None:
    wm, _ = _qb_world_model()
    wm.eval()
    bsz, hist, slots = 2, wm.num_hist, wm.token_count
    history = torch.randn(bsz, hist, slots, wm.obs_token_dim)
    lang = torch.randn(bsz, wm.lang_dim)

    out = wm(
        {
            "mode": "predict_next",
            "latent": history[:, -1],
            "actions": torch.zeros(bsz, 1, wm.action_dim),
            "lang_emb": lang,
        }
    )

    assert out["hidden"].shape == (bsz, slots, wm.obs_token_dim)
    assert out["lang"] is lang


def test_predict_next_forward_folds_batch_proprio_into_raw_tokens() -> None:
    wm, _ = _qb_world_model()
    wm.eval()
    bsz, slots = 2, wm.token_count
    hidden = torch.randn(bsz, slots, wm.token_dim)
    lang = torch.randn(bsz, wm.lang_dim)
    proprio = torch.randn(bsz, wm.proprio_dim)

    out = wm(
        {
            "mode": "predict_next",
            "latent": hidden,
            "actions": torch.zeros(bsz, 1, wm.action_dim),
            "lang_emb": lang,
            "proprio": proprio,
        }
    )

    assert out["hidden"].shape == (bsz, slots, wm.obs_token_dim)
    assert out["proprio"].shape == (bsz, wm.proprio_dim)
    assert out["lang"] is lang


def test_chunk_loss_trains_raw_proprio_head_for_classifier_scoring() -> None:
    wm, _ = _qb_world_model()
    bsz, steps, slots = 2, wm.num_hist + wm.chunk_size, wm.token_count
    batch = {
        "obs_embedding": torch.randn(bsz, steps, slots, wm.token_dim),
        "proprio": torch.randn(bsz, steps, wm.proprio_dim),
        "lang_emb": torch.randn(bsz, wm.lang_dim),
        "actions": torch.zeros(bsz, steps, wm.action_dim),
        "rewards": torch.zeros(bsz, steps),
        "success_to_go": torch.zeros(bsz, steps),
    }

    out = wm.chunk_loss(batch)

    assert "proprio_reconstruction_loss" in out
    assert torch.isfinite(out["proprio_reconstruction_loss"])


def test_observe_sequence_threads_proprio_and_lang_for_lumos() -> None:
    wm, _ = _qb_world_model()
    bsz, steps, slots = 2, wm.num_hist + 3, wm.token_count
    lang = torch.randn(bsz, wm.lang_dim)
    batch = {
        "mode": "observe_sequence",
        "obs_embedding": torch.randn(bsz, steps, slots, wm.token_dim),
        "proprio": torch.randn(bsz, steps, wm.proprio_dim),
        "lang_emb": lang,
        "actions": torch.zeros(bsz, steps, wm.action_dim),
    }

    latent = wm(batch)["latent"]

    assert latent["hidden"].shape == (bsz, steps, slots, wm.obs_token_dim)
    assert latent["history"].shape == (bsz, steps, wm.num_hist, slots, wm.obs_token_dim)
    assert latent["actions"].shape == (bsz, steps, wm.num_hist, wm.action_dim)
    assert latent["lang"] is lang


def test_lumos_flatten_repeats_episode_level_lang() -> None:
    bsz, steps, starts, slots, dim, lang_dim = 2, 5, 3, 4, 6, 7
    lang = torch.randn(bsz, lang_dim)
    latent = {
        "hidden": torch.randn(bsz, steps, slots, dim),
        "history": torch.randn(bsz, steps, 2, slots, dim),
        "actions": torch.randn(bsz, steps, 2, 3),
        "lang": lang,
    }

    flat_last = _flatten_last_steps(latent, starts)
    flat_strided = _flatten_strided_steps(latent, starts, min_start=1)

    expected = lang.repeat_interleave(starts, dim=0)
    assert flat_last["lang"].shape == (bsz * starts, lang_dim)
    assert flat_strided["lang"].shape == (bsz * starts, lang_dim)
    assert torch.allclose(flat_last["lang"], expected)
    assert torch.allclose(flat_strided["lang"], expected)


def test_actor_and_critic_inputs_strip_predicted_proprio() -> None:
    wm, _ = _qb_world_model()
    bsz, slots = 2, wm.token_count
    hidden = torch.randn(bsz, slots, wm.obs_token_dim)
    latent = {"hidden": hidden}

    actor_input = wm.actor_input(latent)
    critic_input = wm.critic_input(latent)

    assert actor_input.shape == (bsz, slots, wm.token_dim)
    assert torch.allclose(actor_input, hidden[..., : wm.token_dim])
    assert critic_input.shape == (bsz, wm.token_dim)
    assert torch.allclose(critic_input, hidden[..., : wm.token_dim].mean(dim=1))


def test_classifier_input_strips_predicted_proprio() -> None:
    wm, _ = _qb_world_model()
    bsz, slots = 2, wm.token_count
    hidden = torch.randn(bsz, slots, wm.obs_token_dim)

    classifier_input = wm(
        {"mode": "classifier_input", "latent": {"hidden": hidden}}
    )

    assert classifier_input.shape == (bsz, slots, wm.token_dim)
    assert torch.allclose(classifier_input, hidden[..., : wm.token_dim])


def test_chunk_loss_with_proprio_language() -> None:
    wm, _ = _qb_world_model()
    bsz, steps, slots = 2, wm.num_hist + wm.chunk_size, wm.token_count
    batch = {
        "obs_embedding": torch.randn(bsz, steps, slots, wm.token_dim),
        "proprio": torch.randn(bsz, steps, wm.proprio_dim),
        "lang_emb": torch.randn(bsz, wm.lang_dim),
        "actions": torch.zeros(bsz, steps, wm.action_dim),
        "rewards": torch.zeros(bsz, steps),
        "success_to_go": torch.zeros(bsz, steps),
    }

    out = wm.chunk_loss(batch)

    assert torch.isfinite(out["loss"])
    assert wm._last_hidden_target_width == wm.obs_token_dim


def _write_reward_hdf5(root: Path, *, length: int = 5) -> Path:
    root.mkdir()
    path = root / "demo.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=np.zeros((length, 7), dtype=np.float32))
        demo.create_dataset("rewards", data=np.zeros((length,), dtype=np.float32))
        demo.create_dataset("dones", data=np.zeros((length,), dtype=np.float32))
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", data=np.zeros((length, 8, 8, 3), dtype=np.uint8))
        obs.create_dataset("eye_in_hand_rgb", data=np.zeros((length, 8, 8, 3), dtype=np.uint8))
        obs.create_dataset("ee_pos", data=np.ones((length, 3), dtype=np.float64))
        obs.create_dataset("ee_ori", data=np.full((length, 3), 2.0, dtype=np.float64))
        obs.create_dataset("gripper_states", data=np.full((length, 2), 3.0, dtype=np.float64))
    return path


def test_dataset_reads_proprio(tmp_path: Path) -> None:
    _write_reward_hdf5(tmp_path / "reward")

    dataset = PixelSequenceDataset(
        tmp_path / "reward",
        sequence_length=4,
        image_size=8,
        proprio_keys=("ee_pos", "ee_ori", "gripper_states"),
    )
    item = dataset[0]

    assert item["proprio"].shape == (4, 8)
    assert item["proprio"].dtype == torch.float32
    assert torch.allclose(item["proprio"][0], torch.tensor([1, 1, 1, 2, 2, 2, 3, 3], dtype=torch.float32))


def test_dataset_reads_lang_emb(tmp_path: Path) -> None:
    reward_path = _write_reward_hdf5(tmp_path / "reward")
    hidden_dir = tmp_path / "hidden"
    hidden_dir.mkdir()
    with h5py.File(hidden_dir / reward_path.name, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset("obs_embedding", data=np.zeros((5, 8, 16), dtype=np.float32))
        demo.create_dataset("lang_emb", data=np.arange(12, dtype=np.float16))

    dataset = PixelHiddenSequenceDataset(
        tmp_path / "reward",
        hidden_dir,
        sequence_length=4,
        image_size=8,
        require_preprocess_config=False,
        lang_emb_dir=hidden_dir,
    )
    item = dataset[0]

    assert item["lang_emb"].shape == (12,)
    assert item["lang_emb"].dtype == torch.float32
    assert torch.allclose(item["lang_emb"], torch.arange(12, dtype=torch.float32))
