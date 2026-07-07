from __future__ import annotations

import json
import types

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from dreamervla.dataset.pixel_hidden_sequence_dataset import (
    PixelHiddenSequenceDataset,
)
from dreamervla.algorithms.actor import (
    LatentToActionHiddenActor,
    LatentToOpenVLADiscreteTokenActor,
    OpenVLADiscreteTokenActor,
    RynnVLAActionHiddenActor,
    VLAActionHeadActor,
)
from dreamervla.algorithms.critic import LatentSuccessClassifier
from dreamervla.models.embodiment.world_model.wm import WorldModel
from dreamervla.runners.dreamervla_runner import DreamerVLARunner
from dreamervla.runners.embodied_eval_runner import EmbodiedEvalRunner


def test_rynn_hidden_sidecar_validates_action_head_type(tmp_path) -> None:
    (tmp_path / "preprocess_config.json").write_text(
        json.dumps(
            {
                "model_path": "/tmp/model",
                "encoder_state_ckpt": "/tmp/encoder.ckpt",
                "time_horizon": 5,
                "action_head_type": "legacy",
                "save_actor_sequence": True,
            }
        ),
        encoding="utf-8",
    )
    dataset = PixelHiddenSequenceDataset.__new__(PixelHiddenSequenceDataset)
    dataset.hidden_dir = tmp_path
    dataset.load_actor_sequence = False

    with pytest.raises(ValueError, match="action_head_type mismatch"):
        dataset._validate_hidden_sidecar(
            expected_model_path=None,
            expected_encoder_state_ckpt=None,
            expected_time_horizon=5,
            expected_action_head_type="old_head",
            require_preprocess_config=True,
        )


def test_rynn_hidden_sidecar_requires_expected_path_metadata(tmp_path) -> None:
    (tmp_path / "preprocess_config.json").write_text(
        json.dumps(
            {
                "time_horizon": 5,
                "action_head_type": "legacy",
            }
        ),
        encoding="utf-8",
    )
    dataset = PixelHiddenSequenceDataset.__new__(PixelHiddenSequenceDataset)
    dataset.hidden_dir = tmp_path
    dataset.load_actor_sequence = False

    with pytest.raises(ValueError, match="model_path mismatch"):
        dataset._validate_hidden_sidecar(
            expected_model_path="/tmp/model",
            expected_encoder_state_ckpt=None,
            expected_time_horizon=5,
            expected_action_head_type="legacy",
            require_preprocess_config=True,
        )

    with pytest.raises(ValueError, match="encoder_state_ckpt mismatch"):
        dataset._validate_hidden_sidecar(
            expected_model_path=None,
            expected_encoder_state_ckpt="/tmp/encoder.ckpt",
            expected_time_horizon=5,
            expected_action_head_type="legacy",
            require_preprocess_config=True,
        )


def test_rynn_hidden_sidecar_accepts_legacy_ckpts_checkpoint_alias(tmp_path) -> None:
    (tmp_path / "preprocess_config.json").write_text(
        json.dumps(
            {
                "model_path": "/old/workspace/DreamerVLA/data/ckpts/VLA_model_256/libero_goal",
                "time_horizon": 5,
                "action_head_type": "legacy",
            }
        ),
        encoding="utf-8",
    )
    dataset = PixelHiddenSequenceDataset.__new__(PixelHiddenSequenceDataset)
    dataset.hidden_dir = tmp_path
    dataset.load_actor_sequence = False

    dataset._validate_hidden_sidecar(
        expected_model_path="data/checkpoints/VLA_model_256/libero_goal",
        expected_encoder_state_ckpt=None,
        expected_time_horizon=5,
        expected_action_head_type="legacy",
        require_preprocess_config=True,
    )


def test_input_token_sidecar_validates_declared_hidden_dim(tmp_path) -> None:
    (tmp_path / "preprocess_config.json").write_text(
        json.dumps(
            {
                "action_head_type": "oft_discrete_token",
                "obs_hidden_source": "input_token_embedding",
                "hidden_key": "obs_embedding",
                "token_count": 3,
                "token_dim": 4,
                "hidden_dim": 12,
            }
        ),
        encoding="utf-8",
    )
    with h5py.File(tmp_path / "shard.hdf5", "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.create_dataset("obs_embedding", data=np.zeros((2, 11), dtype=np.float16))

    dataset = PixelHiddenSequenceDataset.__new__(PixelHiddenSequenceDataset)
    dataset.hidden_dir = tmp_path
    dataset.load_actor_sequence = False

    with pytest.raises(ValueError, match="hidden_dim mismatch"):
        dataset._validate_hidden_sidecar(
            expected_model_path=None,
            expected_encoder_state_ckpt=None,
            expected_time_horizon=None,
            expected_action_head_type="oft_discrete_token",
            expected_obs_hidden_source="input_token_embedding",
            require_preprocess_config=True,
        )


def test_vla_action_head_actor_uses_rynnvla_action_tokens() -> None:
    actor = VLAActionHeadActor(
        hidden_dim=16,
        action_dim=3,
        time_horizon=4,
        vla_hidden_size=16,
        hidden_size_factor=0.25,
        num_encoder_layers=1,
        adapter_type="identity",
        action_head_type="legacy",
    )

    chunk = actor({"mode": "sample", "hidden": torch.randn(2, 16), "deterministic": True})[2][
        "action_chunk"
    ]

    assert actor.action_token_embeddings.weight.shape == (1, 4 * 3 * 16)
    assert chunk.shape == (2, 4, 3)


def test_vla_action_head_actor_rejects_ckpt_without_action_head(tmp_path) -> None:
    path = tmp_path / "vla_without_action_head.ckpt"
    torch.save({"state_dicts": {"encoder": {"backbone.other.weight": torch.ones(1)}}}, path)

    with pytest.raises(RuntimeError, match="action_head"):
        VLAActionHeadActor(
            hidden_dim=16,
            action_dim=3,
            time_horizon=4,
            vla_hidden_size=16,
            hidden_size_factor=0.25,
            num_encoder_layers=1,
            adapter_type="identity",
            action_head_type="legacy",
            init_action_head_ckpt=str(path),
        )


def test_vla_action_head_actor_loads_hf_action_head(tmp_path) -> None:
    source = VLAActionHeadActor(
        hidden_dim=16,
        action_dim=3,
        time_horizon=4,
        vla_hidden_size=16,
        hidden_size_factor=0.25,
        num_encoder_layers=1,
        adapter_type="identity",
        action_head_type="legacy",
    )
    hf_dir = tmp_path / "vla_hf"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    torch.save(
        {f"action_head.{key}": value for key, value in source.state_dict().items()},
        hf_dir / "pytorch_model.bin",
    )

    actor = VLAActionHeadActor(
        hidden_dim=16,
        action_dim=3,
        time_horizon=4,
        vla_hidden_size=16,
        hidden_size_factor=0.25,
        num_encoder_layers=1,
        adapter_type="identity",
        action_head_type="legacy",
        init_action_head_ckpt=str(hf_dir),
    )

    assert torch.equal(
        actor.action_token_embeddings.weight,
        source.action_token_embeddings.weight,
    )


def test_dreamer_eval_keeps_rynnvla_action_hidden_tokens_for_wm() -> None:
    workspace = EmbodiedEvalRunner.__new__(EmbodiedEvalRunner)
    workspace.cfg = OmegaConf.create(
        {
            "eval": {"obs_hidden_source": "action_query", "target_token_id": 10004},
            "encoder": {"action_head_type": "legacy"},
        }
    )

    hidden_states = torch.randn(1, 3, 6)
    input_ids = torch.tensor([[11, 12, 13, 10004]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    action_hidden = torch.arange(1 * 5 * 4, dtype=torch.float32).reshape(1, 5, 4)

    class DummyEncoder:
        def extract_action_hidden(self, **kwargs):
            assert kwargs["hidden_states"] is hidden_states
            assert kwargs["input_ids"] is input_ids
            assert kwargs["attention_mask"] is attention_mask
            assert kwargs["target_token_id"] == 10004
            return action_hidden

    workspace.encoder = DummyEncoder()
    workspace._wm_io_mode = lambda: "hidden"  # type: ignore[method-assign]
    workspace._encode_hidden_sequence_from_tokenized = lambda _tokens: (  # type: ignore[method-assign]
        hidden_states,
        input_ids,
        attention_mask,
    )

    obs_embedding = workspace._obs_embedding_for_wm([[11, 12, 13]])

    assert obs_embedding.shape == (1, 5, 4)
    assert obs_embedding.tolist() == action_hidden.tolist()


def test_dreamer_eval_accepts_plain_dict_checkpoint_cfg() -> None:
    cfg = EmbodiedEvalRunner._checkpoint_cfg_from_payload(
        {
            "cfg": {
                "training": {"out_dir": "/tmp/eval"},
                "eval": {"task_suite_name": "libero_goal"},
            }
        }
    )

    assert OmegaConf.select(cfg, "training.out_dir") == "/tmp/eval"
    assert OmegaConf.select(cfg, "eval.task_suite_name") == "libero_goal"


def test_dreamer_eval_adds_missing_trainer_node_for_manual_checkpoint(
    tmp_path,
) -> None:
    workspace = EmbodiedEvalRunner.__new__(EmbodiedEvalRunner)
    workspace.distributed = types.SimpleNamespace(is_main_process=False)
    workspace._output_dir = str(tmp_path)
    workspace.device = torch.device("cpu")
    workspace.console_banner = lambda *args, **kwargs: None
    workspace.console_metrics = lambda *args, **kwargs: None
    workspace._init_policy_trace = lambda _cfg: None
    workspace._init_real_relabel_export = lambda _cfg: None
    workspace._build_dreamer_modules = lambda _cfg, _payload: None
    workspace.evaluate_libero = lambda epoch: {
        "eval_success_rate": 0.0,
        "eval_total_episodes": 0.0,
        "eval_total_successes": 0.0,
        "eval_tasks": 0.0,
        "eval_episodes_per_task": 3.0,
    }

    eval_cfg = OmegaConf.create(
        {
            "eval": {
                "task_suite_name": "libero_goal",
                "num_episodes_per_task": 3,
            },
            "init": {
                "vla_ckpt_path": "/tmp/openvla",
                "encoder_state_ckpt": None,
            },
            "encoder": {"time_horizon": 5},
            "trainer": {"device": "cpu"},
            "training": {"device": "cpu"},
        }
    )
    payload = {
        "cfg": {
            "training": {"out_dir": "/tmp/manual"},
            "init": {},
            "encoder": {},
            "eval": {"task_suite_name": "libero_goal"},
        },
        "state_dicts": {"world_model": {}, "policy": {}},
    }

    workspace._run_dreamer_eval(eval_cfg, "/tmp/manual_cotrain.ckpt", payload)

    assert OmegaConf.select(workspace.cfg, "trainer.device") == "cpu"


def test_dreamer_eval_normalizes_manual_ray_checkpoint_components(
    tmp_path,
) -> None:
    workspace = EmbodiedEvalRunner.__new__(EmbodiedEvalRunner)
    workspace.distributed = types.SimpleNamespace(is_main_process=False)
    workspace._output_dir = str(tmp_path)
    workspace.device = torch.device("cpu")
    workspace.console_banner = lambda *args, **kwargs: None
    workspace.console_metrics = lambda *args, **kwargs: None
    workspace._init_policy_trace = lambda _cfg: None
    workspace._init_real_relabel_export = lambda _cfg: None
    captured: dict[str, object] = {}
    workspace._build_dreamer_modules = lambda cfg, _payload: captured.setdefault(
        "cfg", cfg
    )
    workspace.evaluate_libero = lambda epoch: {
        "eval_success_rate": 0.0,
        "eval_total_episodes": 0.0,
        "eval_total_successes": 0.0,
        "eval_tasks": 0.0,
        "eval_episodes_per_task": 3.0,
    }

    eval_cfg = OmegaConf.create(
        {
            "eval": {"task_suite_name": "libero_goal", "num_episodes_per_task": 3},
            "init": {"vla_ckpt_path": "/tmp/openvla", "encoder_state_ckpt": None},
            "encoder": {"_target_": "default.RynnVLAEncoder", "time_horizon": 5},
            "trainer": {"device": "cpu"},
            "training": {"device": "cpu"},
        }
    )
    payload = {
        "cfg": {
            "training": {"out_dir": "/tmp/manual"},
            "init": {},
            "eval": {},
            "task": {
                "openvla_oft": {
                    "input_tokens": {
                        "expected_obs_hidden_source": "input_token_embedding"
                    }
                }
            },
            "rollout": {
                "encoder_cfg": {
                    "target": "dreamervla.workers.inference.oft_rollout:OFTRolloutBundle",
                    "kwargs": {"history": 1},
                }
            },
            "actor": {
                "policy_cfg": {
                    "target": "pkg.Policy",
                    "kwargs": {"alpha": 1},
                }
            },
            "learner": {
                "model_cfg": {
                    "world_model": {
                        "target": "pkg.WorldModel",
                        "kwargs": {"beta": 2},
                    },
                    "classifier": {
                        "target": "pkg.Classifier",
                        "kwargs": {"gamma": 3},
                    },
                }
            },
        },
        "state_dicts": {"world_model": {}, "policy": {}},
    }

    workspace._run_dreamer_eval(eval_cfg, "/tmp/manual_cotrain.ckpt", payload)

    cfg = captured["cfg"]
    assert OmegaConf.select(cfg, "encoder", default=None) is None
    assert OmegaConf.select(cfg, "policy._target_") == "pkg.Policy"
    assert OmegaConf.select(cfg, "policy.alpha") == 1
    assert OmegaConf.select(cfg, "world_model._target_") == "pkg.WorldModel"
    assert OmegaConf.select(cfg, "world_model.beta") == 2
    assert OmegaConf.select(cfg, "classifier._target_") == "pkg.Classifier"
    assert OmegaConf.select(cfg, "classifier.gamma") == 3
    assert OmegaConf.select(cfg, "eval.obs_hidden_source") == "input_token_embedding"


def test_dreamer_oft_eval_builds_processor_adapter(monkeypatch) -> None:
    import dreamervla.runners.embodied_eval_runner as eval_runner_mod

    workspace = EmbodiedEvalRunner.__new__(EmbodiedEvalRunner)
    workspace.device = torch.device("cpu")
    workspace.distributed = types.SimpleNamespace(is_main_process=False)
    workspace._dreamer_policy_source = "ckpt"
    workspace._tdmpc_mpc_enabled = False
    workspace._attach_image_token_mapping = lambda: None
    workspace._load_module_state = lambda _module, _state, _name: None

    def build_oft_extractor(_cfg) -> None:
        workspace._dreamer_oft_extractor = object()

    workspace._build_oft_eval_extractor = build_oft_extractor

    class FakeModule(nn.Module):
        pass

    monkeypatch.setattr(
        eval_runner_mod.hydra.utils,
        "instantiate",
        lambda *_args, **_kwargs: FakeModule(),
    )

    cfg = OmegaConf.create(
        {
            "rollout": {
                "encoder_cfg": {
                    "target": "dreamervla.workers.inference.oft_rollout:OFTRolloutBundle"
                }
            },
            "world_model": {"_target_": "fake.WorldModel"},
            "policy": {"_target_": "fake.Policy"},
            "training": {"fsdp_mixed_precision": "fp32"},
            "eval": {},
        }
    )

    workspace._build_dreamer_modules(
        cfg, {"state_dicts": {"world_model": {}, "policy": {}}}
    )

    assert workspace.encoder is not None
    assert workspace.encoder._build_processor(torch.device("cpu")) is None
    assert workspace.encoder.eval() is workspace.encoder


def test_dreamer_eval_oft_bridge_maps_libero_obs_and_sidecars() -> None:
    workspace = EmbodiedEvalRunner.__new__(EmbodiedEvalRunner)
    workspace.device = torch.device("cpu")
    workspace.cfg = OmegaConf.create({"eval": {}})

    class FakeOFTOutput:
        hidden_state = torch.arange(6, dtype=torch.float16).reshape(2, 3)
        lang_emb = torch.arange(4, dtype=torch.float16)

        def __getitem__(self, index: int):
            return [None, self.hidden_state][index]

    class FakeExtractor:
        def __init__(self) -> None:
            self.last_obs = None
            self.last_task = None

        def step(self, obs, task_description):
            self.last_obs = obs
            self.last_task = task_description
            return FakeOFTOutput()

    extractor = FakeExtractor()
    workspace._dreamer_oft_extractor = extractor
    third = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    wrist = np.arange(12, 24, dtype=np.uint8).reshape(2, 2, 3)
    state = np.arange(8, dtype=np.float32)
    workspace._libero_current_raw_obs = {
        "agentview_image": third,
        "robot0_eye_in_hand_image": wrist,
    }

    obs_embedding, input_ids = workspace._dreamer_obs_embedding_from_eval_inputs(
        None,
        [],
        state,
        "Pick the block",
    )

    assert input_ids is None
    assert torch.equal(obs_embedding["obs_embedding"], FakeOFTOutput.hidden_state[None].float())
    assert torch.equal(obs_embedding["lang_emb"], FakeOFTOutput.lang_emb[None].float())
    assert torch.equal(obs_embedding["proprio"], torch.from_numpy(state)[None])
    assert np.array_equal(extractor.last_obs["agentview_rgb"], third)
    assert np.array_equal(extractor.last_obs["eye_in_hand_rgb"], wrist)
    assert np.array_equal(extractor.last_obs["state"], state)
    assert extractor.last_task == "Pick the block"


def test_dreamer_online_update_latent_preserves_eval_sidecars() -> None:
    workspace = EmbodiedEvalRunner.__new__(EmbodiedEvalRunner)

    class DummyWorldModel:
        def __init__(self) -> None:
            self.calls = []

        def __call__(self, batch):
            self.calls.append(batch)
            return {
                "hidden": batch["hidden"],
                "history": batch["hidden"][:, None],
                "actions": torch.zeros(batch["hidden"].shape[0], 1, 7),
            }

    wm = DummyWorldModel()
    workspace.world_model = wm
    obs = {
        "obs_embedding": torch.ones(1, 2, 3),
        "lang_emb": torch.ones(1, 4),
        "proprio": torch.ones(1, 8),
    }

    latent = workspace._dreamer_online_update_latent(obs)

    assert wm.calls[0]["mode"] == "encode_latent"
    assert wm.calls[0]["hidden"] is obs["obs_embedding"]
    assert latent["lang"] is obs["lang_emb"]
    assert latent["proprio"] is obs["proprio"]

    workspace._dreamer_online_prev_action = torch.zeros(1, 7)
    latent = workspace._dreamer_online_update_latent(obs)

    assert wm.calls[1]["mode"] == "observe_next"
    assert wm.calls[1]["hidden"] is obs["obs_embedding"]
    assert latent["lang"] is obs["lang_emb"]
    assert latent["proprio"] is obs["proprio"]


def test_rynnvla_action_hidden_actor_decodes_flattened_action_hidden() -> None:
    actor = RynnVLAActionHiddenActor(
        action_hidden_dim=4,
        action_dim=3,
        time_horizon=5,
        adapter_type="identity",
    )
    assert actor.hidden_dim == 5 * 3 * 4

    action, log_prob, extra = actor(
        {
            "mode": "sample",
            "hidden": torch.randn(2, actor.hidden_dim),
            "deterministic": True,
        }
    )

    assert action.shape == (2, 3)
    assert log_prob.shape == (2,)
    assert extra["action_chunk"].shape == (2, 5, 3)


def test_latent_to_action_hidden_actor_bridges_tokenized_input_latents() -> None:
    actor = LatentToActionHiddenActor(
        hidden_dim=6 * 4,
        source_token_count=6,
        source_token_dim=4,
        action_hidden_dim=8,
        action_dim=2,
        time_horizon=3,
        bridge_hidden_dim=16,
        num_bridge_layers=1,
        num_bridge_heads=2,
        freeze_output_projection=False,
    )

    action, log_prob, extra = actor(
        {
            "mode": "sample",
            "hidden": torch.randn(2, 6, 4),
            "deterministic": True,
            "return_chunk": True,
        }
    )

    assert action.shape == (2, 3, 2)
    assert log_prob.shape == (2,)
    assert extra["action_hidden"].shape == (2, 6, 8)
    assert extra["action_chunk"].shape == (2, 3, 2)


def test_latent_to_action_hidden_actor_accepts_flat_latents() -> None:
    actor = LatentToActionHiddenActor(
        hidden_dim=5 * 4,
        source_token_count=5,
        source_token_dim=4,
        action_hidden_dim=8,
        action_dim=2,
        time_horizon=3,
        bridge_hidden_dim=16,
        num_bridge_layers=1,
        num_bridge_heads=2,
        adapter_type="identity",
    )

    action, _, _ = actor(
        {
            "mode": "sample",
            "hidden": torch.randn(2, 5 * 4),
            "deterministic": True,
            "return_chunk": True,
        }
    )

    assert action.shape == (2, 3, 2)


def test_openvla_discrete_token_actor_uses_token_categorical_log_probs() -> None:
    actor = OpenVLADiscreteTokenActor(
        hidden_dim=2 * 2 * 4,
        action_hidden_dim=4,
        action_dim=2,
        time_horizon=2,
        vocab_size=16,
        action_token_bins=4,
        adapter_type="identity",
        freeze_lm_head=False,
    )
    with torch.no_grad():
        actor.lm_head.weight.zero_()

    action_chunk, log_prob, extra = actor(
        {
            "mode": "sample",
            "hidden": torch.zeros(3, 4, 4),
            "deterministic": True,
            "return_chunk": True,
        }
    )

    assert action_chunk.shape == (3, 2, 2)
    assert extra["action_token_ids"].shape == (3, 2, 2)
    assert torch.all(extra["action_token_ids"] == 12)
    assert torch.allclose(log_prob, torch.full((3,), -4 * torch.log(torch.tensor(4.0))))

    eval_log_prob, entropy, _ = actor(
        {
            "mode": "evaluate",
            "hidden": torch.zeros(3, 4, 4),
            "action": action_chunk,
            "action_token_ids": extra["action_token_ids"],
        }
    )

    assert torch.allclose(eval_log_prob, log_prob)
    assert torch.allclose(entropy, torch.full((3,), 4 * torch.log(torch.tensor(4.0))))


def test_latent_to_openvla_discrete_actor_bridges_input_tokens_without_l1_head() -> None:
    actor = LatentToOpenVLADiscreteTokenActor(
        hidden_dim=5 * 4,
        source_token_count=5,
        source_token_dim=4,
        action_hidden_dim=8,
        action_dim=2,
        time_horizon=3,
        vocab_size=16,
        action_token_bins=4,
        bridge_hidden_dim=8,
        num_bridge_layers=1,
        num_bridge_heads=2,
        bridge_dropout=0.0,
        adapter_type="identity",
        freeze_lm_head=False,
    )
    with torch.no_grad():
        actor.lm_head.weight.zero_()

    action_chunk, log_prob, extra = actor(
        {
            "mode": "sample",
            "hidden": torch.randn(2, 5, 4),
            "deterministic": True,
            "return_chunk": True,
        }
    )

    assert action_chunk.shape == (2, 3, 2)
    assert extra["action_hidden"].shape == (2, 6, 8)
    assert extra["action_token_ids"].shape == (2, 3, 2)
    assert torch.all(extra["action_token_ids"] == 12)
    assert torch.allclose(log_prob, torch.full((2,), -6 * torch.log(torch.tensor(4.0))))

    eval_log_prob, entropy, _ = actor(
        {
            "mode": "evaluate",
            "hidden": torch.randn(2, 5 * 4),
            "action": action_chunk,
            "action_token_ids": extra["action_token_ids"],
        }
    )
    assert torch.allclose(eval_log_prob, log_prob)
    assert torch.allclose(entropy, torch.full((2,), 6 * torch.log(torch.tensor(4.0))))


def test_rynn_wm_derives_flat_action_hidden_dimensions() -> None:
    model = WorldModel(
        obs_dim=None,
        action_dim=3,
        token_count=None,
        token_dim=4,
        time_horizon=5,
        model_dim=16,
        depth=1,
        heads=4,
        mlp_dim=32,
        max_seq_len=8,
    )

    assert model.token_count == 5 * 3
    assert model.obs_dim == 5 * 3 * 4


def test_rynn_wm_accepts_tokenized_action_hidden_without_flattening() -> None:
    model = WorldModel(
        obs_dim=None,
        action_dim=3,
        token_count=None,
        token_dim=4,
        time_horizon=5,
        model_dim=16,
        depth=1,
        heads=4,
        mlp_dim=32,
        max_seq_len=8,
    )
    tokens = torch.randn(2, 15, 4)
    flat = tokens.reshape(2, -1)

    assert model.obs_to_tokens(tokens).shape == (2, 1, 15, 4)
    assert torch.allclose(model.obs_to_tokens(tokens)[:, 0], tokens)
    assert torch.allclose(model.obs_to_tokens(flat)[:, 0], tokens)


def test_rynn_wm_encode_latent_preserves_action_hidden_tokens() -> None:
    model = WorldModel(
        obs_dim=None,
        action_dim=3,
        token_count=None,
        token_dim=4,
        time_horizon=5,
        model_dim=16,
        depth=1,
        heads=4,
        mlp_dim=32,
        max_seq_len=8,
        num_hist=2,
    )
    tokens = torch.randn(2, 15, 4)

    latent = model.encode_latent(tokens)

    assert latent["hidden"].shape == (2, 15, 4)
    assert latent["history"].shape == (2, 2, 15, 4)
    assert model.actor_input(latent).shape == (2, 15, 4)
    assert model.critic_input(latent).shape == (2, 4)


def test_latent_success_classifier_derives_latent_dim() -> None:
    classifier = LatentSuccessClassifier(
        latent_dim=None,
        action_dim=3,
        time_horizon=5,
        token_dim=4,
        window=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        head_type="transformer",
    )

    assert classifier.cfg.latent_dim == 5 * 3 * 4


def test_latent_success_classifier_accepts_tokenized_windows() -> None:
    classifier = LatentSuccessClassifier(
        latent_dim=None,
        action_dim=3,
        time_horizon=5,
        token_dim=4,
        window=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        head_type="linear",
    )

    logits = classifier(torch.randn(3, 2, 15, 4))

    assert logits.shape == (3, 2)


def test_latent_success_classifier_can_mean_pool_tokenized_frames() -> None:
    classifier = LatentSuccessClassifier(
        latent_dim=None,
        action_dim=3,
        time_horizon=5,
        token_dim=4,
        window=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        head_type="transformer",
        token_pool="mean",
    )

    logits = classifier(torch.randn(3, 2, 64, 4))

    assert classifier.cfg.latent_dim == 4
    assert logits.shape == (3, 2)


def test_rynnvla_action_hidden_actor_loads_vla_output_projection(tmp_path) -> None:
    source = RynnVLAActionHiddenActor(
        action_hidden_dim=4,
        action_dim=3,
        time_horizon=5,
        adapter_type="identity",
    )
    ckpt = {
        "state_dicts": {
            "encoder": {
                f"backbone.action_head.output_projection.{key}": value.detach().clone()
                for key, value in source.output_projection.state_dict().items()
            }
        }
    }
    path = tmp_path / "vla.ckpt"
    torch.save(ckpt, path)

    actor = RynnVLAActionHiddenActor(
        action_hidden_dim=4,
        action_dim=3,
        time_horizon=5,
        adapter_type="identity",
        init_action_head_ckpt=str(path),
    )

    for key, value in source.output_projection.state_dict().items():
        assert torch.equal(actor.output_projection.state_dict()[key], value)


def test_rynnvla_action_hidden_actor_loads_hf_bin_output_projection(tmp_path) -> None:
    source = RynnVLAActionHiddenActor(
        action_hidden_dim=4,
        action_dim=3,
        time_horizon=5,
        adapter_type="identity",
    )
    hf_dir = tmp_path / "checkpoint-1"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    torch.save(
        {
            f"action_head.output_projection.{key}": value.detach().clone()
            for key, value in source.output_projection.state_dict().items()
        },
        hf_dir / "pytorch_model.bin",
    )

    actor = RynnVLAActionHiddenActor(
        action_hidden_dim=4,
        action_dim=3,
        time_horizon=5,
        adapter_type="identity",
        init_action_head_ckpt=str(tmp_path),
    )

    for key, value in source.output_projection.state_dict().items():
        assert torch.equal(actor.output_projection.state_dict()[key], value)


def test_rynnvla_action_hidden_actor_rejects_ckpt_without_output_projection(
    tmp_path,
) -> None:
    path = tmp_path / "vla_without_projection.ckpt"
    torch.save({"state_dicts": {"encoder": {"backbone.other.weight": torch.ones(1)}}}, path)

    with pytest.raises(RuntimeError, match="output_projection"):
        RynnVLAActionHiddenActor(
            action_hidden_dim=4,
            action_dim=3,
            time_horizon=5,
            adapter_type="identity",
            init_action_head_ckpt=str(path),
        )


def test_dreamervla_init_loader_filters_and_remaps_compatible_state() -> None:
    class TinyWorldModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.reward_head = nn.Module()
            self.reward_head.net = nn.Module()
            self.reward_head.net.net = nn.Linear(2, 1)
            self.keep = nn.Linear(2, 2)

    class DummyDistributed:
        def __init__(self) -> None:
            self.rank0_only_args: list[bool] = []
            self.is_main_process = False

        def model_state_dict_context(self, _module: nn.Module, rank0_only: bool = True):
            self.rank0_only_args.append(rank0_only)

            class Ctx:
                def __enter__(self):
                    return None

                def __exit__(self, *_exc):
                    return False

            return Ctx()

    workspace = DreamerVLARunner.__new__(DreamerVLARunner)
    workspace.world_model = nn.Module()
    workspace.world_model.module = TinyWorldModel()
    workspace.distributed = DummyDistributed()

    target = workspace.world_model.module
    original_keep_weight = target.keep.weight.detach().clone()
    reward_weight = torch.full_like(target.reward_head.net.net.weight, 0.25)
    reward_bias = torch.full_like(target.reward_head.net.net.bias, -0.5)

    workspace._load_compatible_module_state(
        "world_model",
        {
            "reward_head.net.weight": reward_weight,
            "reward_head.net.bias": reward_bias,
            "keep.weight": torch.zeros(3, 3),
        },
    )

    assert workspace.distributed.rank0_only_args == [False]
    assert torch.equal(target.reward_head.net.net.weight, reward_weight)
    assert torch.equal(target.reward_head.net.net.bias, reward_bias)
    assert torch.equal(target.keep.weight, original_keep_weight)
