"""Online cotrain runner: one-trajectory VLA -> parallel online rollout ->
replay -> WM/classifier warmup -> WM/classifier + slow-policy RL cotrain.

Single Hydra `train` call. Phase switch is by step count
(`training.warmup_steps`). Reuses the existing rollout/replay/WM/classifier/RL
machinery (``dreamervla.*`` only); the offline ``DreamerVLARunner.run()`` is left
untouched (this runner builds its own components from the same Hydra config + the
inherited helpers).

``latent_type``:
  * ``action_hidden``  — WM after the Action Query, before the Action Head;
    online rollout latent = the env's action-query hidden (``obs_embedding``).
    Fully supported online.
  * ``backbone_latent`` — WM before the Action Query (DINO-style visual-language
    latent). Online env rollout is NOT wired (``DreamerVLAOnlineTrainEnv`` only
    emits the ``action_query`` latent), so this runner raises a clear error and
    points to the offline input-token path. See the tutorial doc.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# Force CPU software rendering for robosuite offscreen cameras BEFORE any import
# below pulls in robosuite (which otherwise defaults MUJOCO_GL=egl). The GPU/EGL
# backend's ``read_pixels`` aborts (SIGABRT) intermittently mid-rollout; osmesa is
# stable. This MUST precede the ``train_env`` import and matches the collector
# (collect_parallel_rollouts). setdefault so an explicit MUJOCO_GL still wins.
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from dreamervla.algorithms.dreamervla import world_model_pretrain_step
from dreamervla.algorithms.registry import get_actor_update_route
from dreamervla.constants import DEFAULT_ACTION_TOKEN_ID
from dreamervla.models.reward import build_classifier
from dreamervla.runners.dreamervla_runner import DreamerVLARunner
from dreamervla.runners.online_dreamervla import (
    _unwrap,
    online_classifier_update_step,
)
from dreamervla.runners.online_replay import (
    OnlineReplay,
    get_replay_task_stats_global,
)
from dreamervla.runners.online_utils import (
    obs_to_action_hidden,
    obs_to_input_token_embedding,
)
from dreamervla.runners.vec_rollout_env import default_env_factory
from dreamervla.runners.vectorized_collect import (
    dreamer_image_from_record,
    extractor_obs_from_record,
    proprio_from_record,
)
from dreamervla.utils.hf_checkpoint import is_hf_checkpoint
from dreamervla.utils.hf_module import load_module_pretrained, save_module_pretrained
from dreamervla.utils.optim import build_optimizer
from dreamervla.utils.torch_utils import freeze_module


def build_cotrain_replay_transition(
    rec: dict[str, Any],
    obs_embedding: np.ndarray,
    wm_action: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    *,
    task_id: int,
    task_description: str,
    step: int,
    is_first: bool,
    image_size: int,
) -> dict[str, Any]:
    """Replay transition rebuilt in the parent from a child ``full_record``.

    Multi-env env instances live in child processes, so the parent cannot call
    ``DreamerVLAOnlineTrainEnv.make_transition``. This rebuilds the same record
    from the child's ``full_record`` + per-slot state and is numerically
    equivalent to ``make_transition`` for the OFT ``action_hidden`` rollout
    (env action scale == ``wm_action`` scale; ``info['wm_action']`` is the
    executed env-scale action)."""
    done = bool(terminated or truncated)
    wm = np.asarray(wm_action, dtype=np.float32).reshape(-1)[:7]
    return {
        "image": dreamer_image_from_record(rec, image_size),
        "state": proprio_from_record(rec),
        "action": wm,
        "wm_action": wm,
        "obs_embedding": np.asarray(obs_embedding, dtype=np.float32),
        "reward": np.float32(reward),
        "done": np.float32(done),
        "discount": np.float32(0.0 if terminated else 1.0),
        "is_first": bool(is_first),
        "is_terminal": bool(terminated),
        "is_last": bool(done),
        "task_id": int(task_id),
        "step": int(step),
        "task_description": str(task_description),
    }


def validate_rollout_cfg(num_envs: int, render_backend: str, latent_type: str) -> None:
    """Early validation for the online rollout knobs (RLinf-style fail-fast).

    ``num_envs>1`` enables the vectorized egl path, which supports the OFT
    ``action_hidden`` rollout only and a real render backend per child."""
    if num_envs < 1:
        raise ValueError(f"online_rollout.num_envs must be >= 1, got {num_envs}")
    if num_envs > 1:
        if render_backend not in ("egl", "osmesa"):
            raise ValueError(
                "online_rollout.render_backend must be 'egl' or 'osmesa' for "
                f"num_envs>1, got {render_backend!r}"
            )
        if latent_type == "backbone_latent":
            raise ValueError(
                "vectorized rollout (num_envs>1) supports the OFT action_hidden "
                "path only; backbone_latent requires num_envs=1"
            )


def build_rollout_progress_metrics(
    *,
    counters: dict[str, int],
    env_step: int,
    num_envs: int,
    episode_horizon: int,
    active_episode_steps: list[int] | tuple[int, ...] | None = None,
) -> dict[str, float]:
    """Episode-denominator and active-step rollout metrics.

    ``rollout/success_rate`` is an episode-level metric. Before the first episode
    completes it has no denominator, so ``rollout/success_rate_valid`` marks
    whether the scalar is an actual completed-episode statistic.
    """
    n_episodes = int(counters.get("n_episodes", 0))
    n_success = int(counters.get("n_success", 0))
    success_rate = (n_success / n_episodes) if n_episodes > 0 else 0.0
    current_episodes = int(counters.get("current_episodes", 0))
    current_success = int(counters.get("current_success", 0))
    current_success_rate = (
        current_success / current_episodes if current_episodes > 0 else 0.0
    )

    if active_episode_steps is None:
        steps = [int(env_step)]
    else:
        steps = [max(0, int(s)) for s in active_episode_steps]
    if not steps:
        steps = [0]
    horizon = max(1, int(episode_horizon))
    step_min = float(min(steps))
    step_max = float(max(steps))
    step_mean = float(sum(steps) / len(steps))

    return {
        "rollout/success_rate": float(success_rate),
        "rollout/success_rate_valid": float(n_episodes > 0),
        "rollout/current_success_rate": float(current_success_rate),
        "rollout/current_success_rate_valid": float(current_episodes > 0),
        "rollout/current_episodes": float(current_episodes),
        "rollout/current_successes": float(current_success),
        "rollout/avg_success_rate": float(success_rate),
        "rollout/avg_success_rate_valid": float(n_episodes > 0),
        "rollout/episodes": float(n_episodes),
        "rollout/successes": float(n_success),
        "rollout/env_steps": float(env_step),
        "rollout/num_envs": float(max(1, int(num_envs))),
        "rollout/episode_horizon": float(horizon),
        "rollout/active_episode_step_min": step_min,
        "rollout/active_episode_step_mean": step_mean,
        "rollout/active_episode_step_max": step_max,
        "rollout/episode_progress_max": float(min(1.0, step_max / horizon)),
    }


def build_rollout_vec_env(
    *,
    render_backend: str,
    num_envs: int,
    cfg_kwargs: dict[str, Any],
    env_vars: dict[str, str],
) -> Any:
    """Select + construct the rollout vec env for the vectorized cotrain path.

    Two backends (the user-chosen approaches):

    * ``render_backend == "egl"`` -> ``OnlineEglVecEnv`` (approach 1): each env runs
      through RLinf's vendored ``SubprocVectorEnv`` with RLinf's per-child egl device
      regime. The physical-GPU pool is read from this process's
      ``CUDA_VISIBLE_DEVICES`` (mirrors the ray runner's ``_egl_device_pool``; empty ->
      egl device 0). The adapter applies ``MUJOCO_GL=egl`` + per-child CUDA/EGL device
      vars itself, so the render env vars are stripped before forwarding.
    * otherwise -> ``VecRolloutEnv`` (approach 2): the proven osmesa path, unchanged.

    Module-level so the backend selection is unit-testable without a GPU / full runner.
    """
    # TODO(unify-vec-env): VecRolloutEnv (osmesa) and OnlineEglVecEnv (egl) are both
    # SubprocVectorEnv-style send-all/recv-all wrappers over the SAME env protocol; the egl
    # one only adds RLinf's vendored classes + the per-child device regime. They should
    # collapse into ONE vec env taking the render backend as a parameter (osmesa = skip the
    # egl device regime). Deferred until the egl path is GPU-verified at low per-GPU
    # concurrency, so the merge can be validated against a known-good baseline.
    if render_backend == "egl":
        from dreamervla.envs.online_egl_venv import OnlineEglVecEnv

        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        egl_device_pool = [int(x) for x in cvd.split(",") if x.strip().isdigit()]
        adapter_env_vars = {
            k: v for k, v in env_vars.items() if k not in ("MUJOCO_GL", "PYOPENGL_PLATFORM")
        }
        return OnlineEglVecEnv(
            num_envs=num_envs,
            cfg_kwargs=cfg_kwargs,
            egl_device_pool=egl_device_pool,
            env_vars=adapter_env_vars,
        )

    from dreamervla.runners.vec_rollout_env import VecRolloutEnv

    return VecRolloutEnv(num_envs=num_envs, cfg_kwargs=cfg_kwargs, env_vars=env_vars)


class OnlineCotrainRunner(DreamerVLARunner):
    """Hydra runner for the unified online cotrain pipeline (see module docstring)."""

    runner_name = "online_cotrain"
    runner_status = "current"
    runner_family = "actor"

    # Checkpoint keys: extend the parent's so the trainable cotrain scalar
    # (classifier_threshold) round-trips and the frozen reference policy is never
    # checkpointed. encoder/_unwrapped_world_model are already excluded by parent.
    include_keys = (*DreamerVLARunner.include_keys, "classifier_threshold")
    exclude_keys = (*DreamerVLARunner.exclude_keys, "ref_policy")

    # ------------------------------------------------------------------ helpers
    @property
    def _rank(self) -> int:
        return int(getattr(self.distributed, "rank", 0) or 0)

    @property
    def _world_size(self) -> int:
        return int(getattr(self.distributed, "world_size", 1) or 1)

    def _build_trainable_classifier(self, cfg: DictConfig) -> None:
        """Build a TRAINABLE LatentSuccessClassifier + optimizer (warm-start from
        init.classifier_state_ckpt when given, else fresh from cfg.classifier)."""
        self.classifier_threshold = float(
            OmegaConf.select(cfg, "algorithm.lumos.classifier_threshold", default=0.5)
            or 0.5
        )
        cls_blob = OmegaConf.select(cfg, "classifier", default=None)
        cls_kwargs: dict[str, Any] = (
            dict(OmegaConf.to_container(cls_blob, resolve=True))
            if cls_blob is not None
            else {}
        )
        # Default latent_dim to the WM obs dim, EXCEPT when token_pool="mean":
        # then the classifier pools tokens and its latent_dim must stay token_dim
        # (the flat obs_dim would make the input projection enormous, e.g. the
        # 512*4096 backbone latent -> a multi-billion-param Linear).
        if cls_kwargs.get("latent_dim") is None and str(
            cls_kwargs.get("token_pool", "flat")
        ) != "mean":
            cls_kwargs["latent_dim"] = int(OmegaConf.select(cfg, "world_model.obs_dim"))
        self._classifier_target = str(
            cls_kwargs.get("_target_") or "dreamervla.models.reward.LatentSuccessClassifier"
        )
        self._classifier_cls_kwargs = {
            key: value for key, value in cls_kwargs.items() if key != "_target_"
        }
        classifier = build_classifier(cls_kwargs).to(self.device)
        warm = OmegaConf.select(cfg, "init.classifier_state_ckpt", default=None)
        if warm:
            if is_hf_checkpoint(str(warm)):
                src = load_module_pretrained(str(warm))
                classifier.load_state_dict(src.state_dict())
            else:
                payload = torch.load(str(warm), map_location="cpu", weights_only=False)
                model_sd = payload.get("model", payload.get("state_dicts", {}).get("model"))
                classifier.load_state_dict(model_sd)
                self.classifier_threshold = float(
                    payload.get("threshold", self.classifier_threshold)
                )
            if self.distributed.is_main_process:
                print(f"[online-cotrain] classifier warm-started from {warm}", flush=True)
        for p in classifier.parameters():
            p.requires_grad_(True)
        classifier.train()
        self.classifier = self.distributed.wrap_trainable_module(classifier)
        cls_optim_cfg = OmegaConf.select(cfg, "optim.classifier")
        if cls_optim_cfg is None:
            raise ValueError("online cotrain requires `optim.classifier`.")
        self.classifier_optimizer = build_optimizer(self.classifier, cls_optim_cfg)
        self._cls_window = int(cls_kwargs.get("window", 8))

    def _load_world_model_init_ckpt(self, ckpt_path: str) -> None:
        """HF-aware override: load from HF dir or fall back to torch ckpt (parent)."""
        if is_hf_checkpoint(ckpt_path):
            src = load_module_pretrained(ckpt_path)
            self._unwrapped_world_model.load_state_dict(src.state_dict())
            if self.distributed.is_main_process:
                print(f"[init] world_model loaded from HF dir: {ckpt_path}", flush=True)
        else:
            super()._load_world_model_init_ckpt(ckpt_path)

    def _assert_optimizers_disjoint(self) -> None:
        seen: set[int] = set()
        for name, opt in (
            ("world_model", self.world_model_optimizer),
            ("policy", self.policy_optimizer),
            ("critic", self.critic_optimizer),
            ("classifier", getattr(self, "classifier_optimizer", None)),
        ):
            if opt is None:
                continue
            for group in opt.param_groups:
                for p in group["params"]:
                    if id(p) in seen:
                        raise RuntimeError(
                            f"optimizer parameter sets overlap at {name} — phase "
                            "isolation (freeze) would be violated."
                        )
                    seen.add(id(p))
        if self.distributed.is_main_process:
            print(
                "[ok] phase isolation: optimizer param sets are disjoint",
                flush=True,
            )

    def _env_cfg_kwargs(self, cfg: DictConfig) -> dict[str, Any]:
        """DreamerVLAOnlineTrainEnvConfig kwargs shared by the single-env builder and
        the vectorized VecRolloutEnv children — one source of truth so both render
        the same observations (action_input='normalized', same history/rotate/etc.)."""
        env_cfg = OmegaConf.select(cfg, "env", default={}) or {}
        task_ids = OmegaConf.select(env_cfg, "task_ids", default=None)
        task_ids = tuple(int(x) for x in task_ids) if task_ids is not None else None
        seed = int(OmegaConf.select(cfg, "seed", default=7)) + self._rank * 1000
        return {
            "_target_": str(
                OmegaConf.select(
                    env_cfg,
                    "_target_",
                    default="dreamervla.envs.train_env.DreamerVLAOnlineTrainEnv",
                )
            ),
            "task_suite_name": str(
                OmegaConf.select(env_cfg, "task_suite_name", default="libero_goal")
            ),
            "task_id": int(task_ids[0]) if task_ids else 0,
            "task_ids": task_ids,
            "seed": seed,
            "max_steps": int(OmegaConf.select(env_cfg, "episode_horizon", default=200)),
            "action_input": "normalized",
            "history_length": int(OmegaConf.select(env_cfg, "history_length", default=2)),
            "include_state": bool(OmegaConf.select(env_cfg, "include_state", default=True)),
            "vla_rotate_180": bool(OmegaConf.select(env_cfg, "vla_rotate_180", default=True)),
            "obs_hidden_source": str(
                OmegaConf.select(env_cfg, "obs_hidden_source", default="action_query")
            ),
            "action_head_type": str(
                OmegaConf.select(env_cfg, "action_head_type", default="legacy")
            ),
        }

    def _build_env(self, cfg: DictConfig) -> Any:
        # MUJOCO_GL=osmesa is forced at module import (before robosuite loads); see
        # the top of this file. Setting it here would be too late (egl already locked).
        return default_env_factory(self._env_cfg_kwargs(cfg))

    @torch.no_grad()
    def _actor_action_and_latent(self, world_model, policy, obs_embedding, latent, prev_action, is_first):
        """WM-latent step + trained-actor sample from an ``action_hidden`` embedding.

        Returns ``(action[7], new_latent)``. The KL/BC-anchored RL actor (an adapter split
        out of the VLA, init from the OFT head so it starts ≈ base) DRIVES the env: this
        both deploys the policy PPO is optimizing and collects on-policy data for the WM,
        instead of freezing the base. Runs in the caller's no_grad scope."""
        if latent is None or is_first:
            latent = world_model({"mode": "encode_latent", "hidden": obs_embedding})
        else:
            latent = world_model(
                {
                    "mode": "observe_next",
                    "latent": latent,
                    "hidden": obs_embedding,
                    "actions": prev_action,
                    "is_first": False,
                }
            )
        feat = world_model({"mode": "actor_input", "latent": latent}).float()
        action_chunk, _lp, _x = policy(
            {"mode": "sample", "hidden": feat, "deterministic": False, "return_chunk": True}
        )
        chunk = action_chunk.reshape(-1, action_chunk.shape[-1]).detach().cpu().float().numpy()
        return np.asarray(chunk[0][:7], dtype=np.float32), latent

    def _rollout_action(self, world_model, policy, processor, obs, latent, prev_action, target_token_id):
        """One env-step action + the WM-input latent (``obs_embedding``) for this
        frame. Returns (action[7], obs_embedding, latent).

        The env is driven by the TRAINED actor (``_actor_action_and_latent``): the OFT
        extractor supplies only the ``action_hidden`` / backbone latent (the per-step
        ``obs_embedding`` the WM + classifier consume); the action itself comes from the
        WM-latent + actor sample. The actor is an adapter split out of the VLA, KL/BC-anchored
        to a frozen ≈base reference, so deploying it is safe and closes the on-policy loop
        (earlier this path froze the base, which left the optimized actor undeployed and made
        rollout success unmovable by PPO).

        ``obs_embedding`` is the post-Action-Query action-hidden (``action_hidden``) or the
        pre-Action-Query backbone input-token latent (``backbone_latent``)."""
        is_first = bool(obs.get("is_first", False))
        task_desc = str(obs.get("task_description", ""))
        backbone = getattr(self, "_latent_type", "action_hidden") == "backbone_latent"
        extractor_attr = "_oft_input_token_extractor" if backbone else "_oft_action_hidden_extractor"
        extractor = getattr(self, extractor_attr, None)
        if extractor is not None:
            if is_first and hasattr(extractor, "reset"):
                extractor.reset()
            _chunk, flat_hidden = extractor.step(obs, task_desc)
            obs_embedding = flat_hidden.reshape(1, -1).to(self.device).float()
        elif backbone:
            obs_embedding = obs_to_input_token_embedding(
                self.encoder, processor, obs, self.device, getattr(self, "_num_views", 2)
            )
        else:
            obs_embedding = obs_to_action_hidden(
                self.encoder, processor, obs, self.device, target_token_id
            )
        action, latent = self._actor_action_and_latent(
            world_model, policy, obs_embedding, latent, prev_action, is_first
        )
        return action, obs_embedding, latent

    # ------------------------------------------------------------------ main
    def run(self) -> list[dict[str, float | str | int]]:  # noqa: C901
        cfg = copy.deepcopy(self.cfg)
        latent_type = str(OmegaConf.select(cfg, "latent_type", default="action_hidden"))
        if self.distributed.is_main_process:
            print(f"[online-cotrain] runner begin. latent_type={latent_type}", flush=True)
        if latent_type not in ("action_hidden", "backbone_latent"):
            raise ValueError(
                f"unknown latent_type={latent_type!r}; expected "
                "'action_hidden' or 'backbone_latent'"
            )
        self._latent_type = latent_type
        # backbone_latent online rollout extracts the pre-Action-Query visual-
        # language latent (current-frame VQ image tokens through the backbone
        # input-embedding table); the env carries the matching obs_hidden_source.
        env_image_keys = OmegaConf.select(
            cfg, "env.image_keys", default=["agentview_rgb", "eye_in_hand_rgb"]
        )
        self._num_views = len(list(env_image_keys)) if env_image_keys is not None else 2
        if latent_type == "backbone_latent":
            OmegaConf.update(cfg, "env.obs_hidden_source", "input_token_embedding", force_add=True)

        self._build_components(cfg)
        return self._online_cotrain_loop(cfg)

    def build_encoder_cfg(self, cfg: DictConfig) -> DictConfig:
        """Return the frozen encoder config declared by Hydra."""
        return super().build_encoder_cfg(cfg)

    def _build_components(self, cfg: DictConfig) -> None:
        # ---- components (reuse hydra targets + inherited helpers; no offline run() touch)
        total_env_steps = int(OmegaConf.select(cfg, "online_rollout.total_env_steps", default=1))
        if total_env_steps <= 0:
            self.encoder = None
            self.processor = None
            self._oft_input_token_extractor = None
            self._oft_action_hidden_extractor = None
        else:
            encoder_cfg = self._build_frozen_encoder_cfg(cfg)
            self.encoder = hydra.utils.instantiate(encoder_cfg).to(self.device)
            freeze_module(self.encoder)
            self.processor = (
                self.encoder._build_processor(self.device)
                if hasattr(self.encoder, "_build_processor")
                else None
            )
            self._oft_input_token_extractor = self._build_oft_input_token_extractor(cfg)
            self._oft_action_hidden_extractor = self._build_oft_action_hidden_extractor(cfg)

        self.world_model = hydra.utils.instantiate(OmegaConf.select(cfg, "world_model")).to(
            device=self.device, dtype=torch.bfloat16
        )
        self._unwrapped_world_model = self.world_model
        wm_ckpt = OmegaConf.select(cfg, "init.world_model_state_ckpt", default=None)
        if wm_ckpt:
            self._load_world_model_init_ckpt(str(wm_ckpt))
        self.world_model = self.distributed.wrap_trainable_module(self.world_model)
        self.world_model_optimizer = build_optimizer(
            self.world_model, OmegaConf.select(cfg, "optim.world_model")
        )

        policy_module = hydra.utils.instantiate(OmegaConf.select(cfg, "policy")).to(self.device)
        algo = OmegaConf.select(cfg, "algorithm")
        if (
            float(OmegaConf.select(algo, "kl_coef", default=0.0)) > 0.0
            or float(OmegaConf.select(algo, "actor_bc_to_ref_scale", default=0.0)) > 0.0
        ):
            self.ref_policy = copy.deepcopy(policy_module).to(self.device)
            freeze_module(self.ref_policy)
            self.ref_policy.eval()
        self.policy = self.distributed.wrap_trainable_module(policy_module)
        self.policy_optimizer = build_optimizer(
            self.policy, OmegaConf.select(cfg, "optim.policy")
        )

        self.critic = hydra.utils.instantiate(OmegaConf.select(cfg, "critic")).to(self.device)
        self.critic = self.distributed.wrap_trainable_module(self.critic)
        self.critic_optimizer = build_optimizer(
            self.critic, OmegaConf.select(cfg, "optim.critic")
        )

        self._build_trainable_classifier(cfg)
        self._assert_optimizers_disjoint()

    def _build_oft_input_token_extractor(self, cfg: DictConfig) -> Any | None:
        if getattr(self, "_latent_type", "action_hidden") != "backbone_latent":
            return None
        encoder = getattr(self, "encoder", None)
        if encoder is None or not hasattr(encoder, "vla") or not hasattr(encoder, "processor"):
            return None

        stats_path = OmegaConf.select(
            cfg,
            "task.openvla_oft.dataset_statistics_path",
            default=None,
        )
        if stats_path is not None and hasattr(encoder, "vla"):
            path = Path(str(stats_path)).expanduser()
            if path.is_file():
                with path.open("r", encoding="utf-8") as handle:
                    encoder.vla.norm_stats = json.load(handle)

        from dreamervla.runners.rollout_hidden_extractor import OFTRolloutHiddenExtractor

        env_cfg = OmegaConf.select(cfg, "env", default={}) or {}
        image_keys = OmegaConf.select(env_cfg, "image_keys", default=["agentview_rgb"])
        return OFTRolloutHiddenExtractor(
            encoder,
            image_keys=list(image_keys),
            history=int(OmegaConf.select(env_cfg, "history_length", default=1)),
            rotate_images_180=bool(
                OmegaConf.select(env_cfg, "vla_rotate_180", default=True)
            ),
            center_crop=True,
            unnorm_key=str(
                OmegaConf.select(
                    cfg,
                    "task.openvla_oft.dataset_statistics_key",
                    default="libero_goal_no_noops",
                )
            ),
            obs_hidden_source="input_token_embedding",
        )

    def _build_oft_action_hidden_extractor(self, cfg: DictConfig) -> Any | None:
        """OFT action-query hidden extractor for the action_hidden online rollout.

        Mirrors ``_build_oft_input_token_extractor`` but with
        ``obs_hidden_source="action_query"`` so it emits the (56*4096,)=229376
        action-query hidden matching the coldstart action_hidden sidecars. Returns
        ``None`` for non-OFT (RynnVLA) encoders, which keep the ``obs_to_action_hidden``
        path."""
        if getattr(self, "_latent_type", "action_hidden") != "action_hidden":
            return None
        encoder = getattr(self, "encoder", None)
        if encoder is None or not hasattr(encoder, "vla") or not hasattr(encoder, "processor"):
            return None

        stats_path = OmegaConf.select(
            cfg,
            "task.openvla_oft.dataset_statistics_path",
            default=None,
        )
        if stats_path is not None and hasattr(encoder, "vla"):
            path = Path(str(stats_path)).expanduser()
            if path.is_file():
                with path.open("r", encoding="utf-8") as handle:
                    encoder.vla.norm_stats = json.load(handle)

        from dreamervla.runners.rollout_hidden_extractor import OFTRolloutHiddenExtractor

        env_cfg = OmegaConf.select(cfg, "env", default={}) or {}
        image_keys = OmegaConf.select(env_cfg, "image_keys", default=["agentview_rgb"])
        return OFTRolloutHiddenExtractor(
            encoder,
            image_keys=list(image_keys),
            history=int(OmegaConf.select(env_cfg, "history_length", default=1)),
            rotate_images_180=bool(
                OmegaConf.select(env_cfg, "vla_rotate_180", default=True)
            ),
            center_crop=True,
            unnorm_key=str(
                OmegaConf.select(
                    cfg,
                    "task.openvla_oft.dataset_statistics_key",
                    default="libero_goal_no_noops",
                )
            ),
            obs_hidden_source="action_query",
        )

    def _online_cotrain_loop(self, cfg: DictConfig) -> list:  # noqa: C901
        # Mid-cotrain resume: restore module weights, optimizer state, and
        # global_step from checkpoints/latest.ckpt when training.resume=true.
        # The env rollout loop warm-restarts (env_step/replay are not serialized).
        self.resume()
        processor = self.processor
        algo = OmegaConf.select(cfg, "algorithm")
        # ---- run-control knobs
        oc = OmegaConf.select(cfg, "online_rollout", default={}) or {}
        warmup_steps = int(OmegaConf.select(cfg, "training.warmup_steps", default=5000))
        train_actor_after = bool(
            OmegaConf.select(cfg, "training.train_actor_after_warmup", default=True)
        )
        train_cls_inline = bool(
            OmegaConf.select(cfg, "training.train_classifier_inline", default=True)
        )
        cls_bs = int(OmegaConf.select(cfg, "training.classifier_batch_size", default=16))
        target_token_id = int(
            OmegaConf.select(
                cfg, "env.target_token_id", default=DEFAULT_ACTION_TOKEN_ID
            )
        )
        seq_len = int(OmegaConf.select(oc, "sequence_length", default=24))
        batch_size = int(OmegaConf.select(cfg, "dataloader.batch_size", default=4))
        min_replay = int(OmegaConf.select(oc, "min_replay", default=seq_len * batch_size))
        min_eps = int(OmegaConf.select(oc, "min_episodes_per_task", default=1))
        train_every = int(OmegaConf.select(oc, "train_every", default=8))
        updates_per_train = int(OmegaConf.select(oc, "updates_per_train", default=1))
        total_env_steps = int(OmegaConf.select(oc, "total_env_steps", default=200000))
        max_train_updates = OmegaConf.select(oc, "max_train_updates", default=None)
        buffer_size = int(OmegaConf.select(oc, "buffer_size", default=20000))
        replay_capacity_mode = str(
            OmegaConf.select(oc, "replay_capacity_mode", default="per_task")
        )
        episode_horizon = int(OmegaConf.select(cfg, "env.episode_horizon", default=200))
        optim_cfg = OmegaConf.select(cfg, "optim")
        early_neg_stride = int(OmegaConf.select(oc, "classifier_early_neg_stride", default=8))
        ckpt_every = int(OmegaConf.select(cfg, "training.checkpoint_every", default=2000))
        # RLinf-aligned default: vectorized egl multi-env rollout (was 1 = legacy
        # single-env osmesa). 4 matches the shipped cotrain pipeline config, so bare/
        # ad-hoc runs no longer fall back to single-env osmesa. backbone_latent must
        # override to 1 (validate_rollout_cfg rejects num_envs>1 for backbone_latent).
        num_envs = int(OmegaConf.select(oc, "num_envs", default=4))
        render_backend = str(OmegaConf.select(oc, "render_backend", default="egl"))
        validate_rollout_cfg(
            num_envs, render_backend, getattr(self, "_latent_type", "action_hidden")
        )

        if bool(OmegaConf.select(cfg, "training.debug", default=False)):
            total_env_steps = int(OmegaConf.select(oc, "debug_total_env_steps", default=64))
            warmup_steps = int(OmegaConf.select(oc, "debug_warmup_steps", default=2))
            min_replay = int(OmegaConf.select(oc, "debug_min_replay", default=seq_len))
            max_train_updates = int(OmegaConf.select(oc, "debug_max_train_updates", default=4))
            episode_horizon = int(OmegaConf.select(oc, "debug_episode_horizon", default=30))
            OmegaConf.update(cfg, "env.episode_horizon", episode_horizon)

        env_task_ids = OmegaConf.select(cfg, "env.task_ids", default=[0]) or [0]
        env_task_ids = tuple(int(x) for x in env_task_ids)
        replay = OnlineReplay(
            capacity=buffer_size,
            sequence_length=seq_len,
            task_ids=env_task_ids,
            capacity_mode=replay_capacity_mode,
            rank=self._rank,
        )
        is_dist = self._world_size > 1
        # Seed the online replay from the warmup data so RL starts RIGHT AFTER the offline
        # warmup instead of idling through a cold-start refill. The training burst is gated
        # on ready_for_training (>= min_episodes_per_task for EVERY task on EVERY rank); with
        # a fresh empty buffer that gate stays false until the online rollout re-covers all
        # tasks (slow at horizon=300), so WM/classifier/actor sit idle and never imagine. A
        # small per-task seed makes the gate pass at warmup end while leaving buffer room for
        # fresh online experience. Skipped when no offline warmup data is configured (a bare
        # standalone online run).
        seed_dir = OmegaConf.select(cfg, "offline_warmup.data_dir", default=None)
        if seed_dir is not None:
            seed_cap = int(
                OmegaConf.select(oc, "warmup_seed_episodes_per_task", default=max(min_eps + 2, 3))
            )
            if seed_cap > 0:
                from dreamervla.runners.offline_seed import seed_replay_from_offline

                seed_task = OmegaConf.select(cfg, "offline_warmup.task_id", default=None)
                n_seed = seed_replay_from_offline(
                    replay,
                    data_dir=seed_dir,
                    hidden_dir=OmegaConf.select(cfg, "offline_warmup.hidden_dir"),
                    default_task_id=(int(seed_task) if seed_task is not None else None),
                    max_episodes_per_task=seed_cap,
                )
                if self.distributed.is_main_process:
                    print(
                        f"[online-cotrain] seeded online replay with {n_seed} warmup episodes "
                        f"(<= {seed_cap}/task) -> {replay.num_transitions} transitions; "
                        "RL starts at warmup end",
                        flush=True,
                    )
        if self.distributed.is_main_process:
            print(
                f"[online-cotrain] warmup_steps={warmup_steps} "
                f"train_actor_after={train_actor_after} train_cls_inline={train_cls_inline} "
                f"buffer_size={buffer_size} seq_len={seq_len} train_every={train_every} "
                f"episode_horizon={episode_horizon} world_size={self._world_size}",
                flush=True,
            )

        # Shared run-control knobs + counters: the legacy single-env loop and the
        # vectorized (num_envs>1) rollout both drive the same training burst.
        knobs = {
            "min_replay": min_replay,
            "min_eps": min_eps,
            "is_dist": is_dist,
            "train_every": train_every,
            "updates_per_train": updates_per_train,
            "max_train_updates": max_train_updates,
            "warmup_steps": warmup_steps,
            "train_actor_after": train_actor_after,
            "train_cls_inline": train_cls_inline,
            "cls_bs": cls_bs,
            "early_neg_stride": early_neg_stride,
            "batch_size": batch_size,
            "optim_cfg": optim_cfg,
            "algo": algo,
            "actor_update_route": get_actor_update_route(
                str(OmegaConf.select(algo, "update_type", default="LUMOS"))
            ),
            "ckpt_every": ckpt_every,
            "num_envs": num_envs,
            "episode_horizon": episode_horizon,
        }
        counters = {
            "n_episodes": 0,
            "n_success": 0,
            "current_episodes": 0,
            "current_success": 0,
        }
        history: list[dict[str, float | str | int]] = []

        if num_envs > 1:
            return self._run_vectorized_cotrain(
                cfg,
                replay=replay,
                num_envs=num_envs,
                render_backend=render_backend,
                total_env_steps=total_env_steps,
                episode_horizon=episode_horizon,
                env_task_ids=env_task_ids,
                knobs=knobs,
                counters=counters,
                history=history,
            )

        env = self._build_env(cfg)
        obs, _info = env.reset()
        latent: Any = None
        prev_action: torch.Tensor | None = None
        episode: list[dict[str, Any]] = []
        stop = False

        for env_step in range(1, total_env_steps + 1):
            if stop:
                break
            self.console_progress(env_step, total_env_steps, "cotrain", unit="env")
            policy_action, obs_embedding, latent = self._rollout_action(
                self.world_model, self.policy, processor, obs, latent, prev_action, target_token_id
            )
            next_obs, reward, terminated, truncated, info = env.step(policy_action)
            done = bool(terminated or truncated)
            wm_action = np.asarray(info.get("wm_action", policy_action), dtype=np.float32).reshape(-1)[:7]
            transition = env.make_transition(obs, policy_action, reward, terminated, truncated, info)
            transition["obs_embedding"] = (
                obs_embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
            )
            episode.append(transition)
            prev_action = torch.from_numpy(wm_action).to(self.device, dtype=obs_embedding.dtype).unsqueeze(0)

            obs = next_obs
            if done:
                rec = replay.add_episode(episode)
                if rec is not None:
                    counters["n_episodes"] += 1
                    success = bool(rec["success"])
                    counters["n_success"] += int(success)
                    counters["current_episodes"] += 1
                    counters["current_success"] += int(success)
                    self.console_record_success(success)
                episode = []
                obs, _info = env.reset()
                latent, prev_action = None, None

            stop = self._run_training_bursts(
                env_step,
                total_env_steps,
                replay=replay,
                env_task_ids=env_task_ids,
                knobs=knobs,
                counters=counters,
                history=history,
                active_episode_steps=[len(episode)],
            )

        if self.distributed.is_main_process:
            self._save_cotrain_ckpt()
        try:
            env.close()
        except Exception:
            pass
        return history

    def _run_training_bursts(
        self,
        env_step: int,
        total_env_steps: int,
        *,
        replay: OnlineReplay,
        env_task_ids: tuple[int, ...],
        knobs: dict[str, Any],
        counters: dict[str, int],
        history: list,
        active_episode_steps: list[int] | tuple[int, ...] | None = None,
    ) -> bool:
        """Run the WM/classifier/RL training bursts for one env-step. Returns True
        when ``max_train_updates`` is reached (caller should stop). Shared by the
        legacy single-env loop and the vectorized (num_envs>1) rollout — the burst
        math is identical; only the rollout that fills ``replay`` differs."""
        # ---- training bursts (lockstep across ranks via global-ready flag)
        # Readiness (and its per-step DDP all_reduce) is only consulted on
        # train_every boundaries, so skip the replay scan + collective on the other
        # steps — identical training cadence, fewer per-step scans/collectives. All
        # ranks gate on the shared env_step, so the all_reduce stays in lockstep.
        if env_step % knobs["train_every"] != 0:
            return False
        _stats, _cov_ready, all_ready = get_replay_task_stats_global(
            replay,
            task_ids=env_task_ids,
            min_transitions=knobs["min_replay"],
            min_episodes_per_task=knobs["min_eps"],
            device=self.device,
            is_dist=knobs["is_dist"],
            world_size=self._world_size,
        )
        num_updates = knobs["updates_per_train"] if all_ready else 0
        for _ in range(num_updates):
            if (
                knobs["max_train_updates"] is not None
                and self.global_step >= int(knobs["max_train_updates"])
            ):
                return True
            in_warmup = self.global_step < knobs["warmup_steps"]
            metrics: dict[str, float | str | int] = {
                "global_step": int(self.global_step),
                "phase": "warmup" if in_warmup else "cotrain",
                "buffer/size": float(replay.num_transitions),
            }
            metrics.update(
                build_rollout_progress_metrics(
                    counters=counters,
                    env_step=env_step,
                    num_envs=int(knobs["num_envs"]),
                    episode_horizon=int(knobs["episode_horizon"]),
                    active_episode_steps=active_episode_steps,
                )
            )
            # Phase WM (always) — policy/actor frozen (eval + no policy optim step)
            self.world_model.train()
            self.policy.eval()
            self.critic.eval()
            _unwrap(self.classifier).eval()
            wm_batch = self._build_wm_pretrain_batch(replay.sample(knobs["batch_size"]))
            if wm_batch is not None:
                wm_metrics = world_model_pretrain_step(
                    policy=self.policy,
                    world_model=self.world_model,
                    optimizer=self.world_model_optimizer,
                    batch=wm_batch,
                    device=self.device,
                    optim_cfg=knobs["optim_cfg"],
                )
                metrics["wm/loss"] = float(wm_metrics.get("loss", 0.0))

            # Phase CLS (always) — WM/actor frozen; classifier trains
            if knobs["train_cls_inline"] and replay.classifier_window_count(
                window=self._cls_window,
                chunk_size=int(getattr(_unwrap(self.classifier).cfg, "chunk_size", 1)),
            ) > 0:
                cls_metrics = online_classifier_update_step(
                    classifier=self.classifier,
                    optimizer=self.classifier_optimizer,
                    replay=replay,
                    device=self.device,
                    batch_size=knobs["cls_bs"],
                    early_neg_stride=knobs["early_neg_stride"],
                    grad_clip=float(
                        OmegaConf.select(knobs["optim_cfg"], "grad_clip_norm", default=1.0)
                    ),
                )
                metrics["cls/loss"] = float(cls_metrics["loss"])
                metrics["cls/acc"] = float(cls_metrics["acc"])
                metrics["cls/f1"] = float(cls_metrics["f1"])
                metrics["cls/pos_frac"] = float(cls_metrics.get("pos_frac", 0.0))
                metrics["cls/prob_mean"] = float(cls_metrics.get("prob_mean", 0.0))
                metrics["cls/grad_norm"] = float(cls_metrics.get("grad_norm", 0.0))

            # Phase RL (cotrain only) — WM + classifier frozen; slow policy
            if (not in_warmup) and knobs["train_actor_after"]:
                self.world_model.eval()
                _unwrap(self.classifier).eval()
                assert not _unwrap(self.world_model).training, "WM must be frozen in RL phase"
                assert not _unwrap(self.classifier).training, "classifier must be frozen in RL phase"
                rl_batch = replay.sample(knobs["batch_size"])
                # raw replay fields (tokenized obs_embedding [B,T,N,D] kept as-is);
                # mirrors online_dreamervla (do NOT route through the offline
                # _build_actor_critic_batch, which expects flat [B,T,D]).
                obs_for_update = {
                    k: rl_batch[k]
                    for k in (
                        "obs_embedding", "actions", "rewards", "dones",
                        "is_first", "is_terminal", "is_last",
                    )
                }
                actor_update_route = knobs["actor_update_route"]
                if actor_update_route.world_model_arg != "chunk_world_model":
                    raise ValueError(
                        "online cotrain requires a chunk-world-model actor update "
                        f"route, got {actor_update_route.name!r}"
                    )
                ac_metrics = actor_update_route.step_fn(
                    policy=self.policy,
                    chunk_world_model=self.world_model,
                    classifier=_unwrap(self.classifier),  # predict_success: DDP wrapper hides it
                    classifier_threshold=self.classifier_threshold,
                    actor_optimizer=self.policy_optimizer,
                    obs=obs_for_update,
                    device=self.device,
                    algorithm_cfg=knobs["algo"],
                    optim_cfg=knobs["optim_cfg"],
                    ref_policy=self.ref_policy,
                )
                metrics["rl/actor_loss"] = float(ac_metrics.get("actor_loss", 0.0))
                metrics["rl/returns_mean"] = float(ac_metrics.get("returns_mean", 0.0))
                metrics["rl/returns_std"] = float(ac_metrics.get("returns_std", 0.0))
                metrics["rl/advantage_std"] = float(
                    ac_metrics.get("advantage_std", 0.0)
                )
                metrics["rl/advantage_mag"] = float(
                    ac_metrics.get("advantage_mag", 0.0)
                )
                metrics["rl/policy_grad_norm"] = float(ac_metrics.get("actor_grad_norm", 0.0))
                metrics["rl/ppo_step_applied"] = float(
                    ac_metrics.get("ppo_step_applied", 0.0)
                )
                for key in (
                    "LUMOS/success_rate",
                    "LUMOS/score_mean",
                    "LUMOS/score_std",
                    "LUMOS/group_var_keep_frac",
                    "LUMOS/num_mixed_groups",
                    "LUMOS/num_all_success_groups",
                    "LUMOS/num_all_fail_groups",
                ):
                    metrics[key] = float(ac_metrics.get(key, 0.0))

            self.console_metrics(
                f"{metrics['phase']} · env {env_step}/{total_env_steps} "
                f"({100.0 * env_step / max(1, total_env_steps):.0f}%) · upd {self.global_step}",
                metrics,
            )
            if self.distributed.is_main_process:
                self.log_metrics(metrics, step=int(self.global_step))
            history.append(metrics)
            counters["current_episodes"] = 0
            counters["current_success"] = 0
            if (
                self.distributed.is_main_process
                and knobs["ckpt_every"] > 0
                and (self.global_step + 1) % knobs["ckpt_every"] == 0
            ):
                self._save_cotrain_ckpt()
            self.global_step += 1
        return False

    def _run_vectorized_cotrain(
        self,
        cfg: DictConfig,
        *,
        replay: OnlineReplay,
        num_envs: int,
        render_backend: str,
        total_env_steps: int,
        episode_horizon: int,
        env_task_ids: tuple[int, ...],
        knobs: dict[str, Any],
        counters: dict[str, int],
        history: list,
    ) -> list:
        """num_envs>1 path: spawn K env children (egl-isolated GL contexts) and run the
        continuous vectorized rollout, interleaving the same training burst per env-step.
        The vec env backend is chosen by ``render_backend`` (``build_rollout_vec_env``):
        egl -> RLinf-vendored ``OnlineEglVecEnv``, osmesa -> ``VecRolloutEnv``. The legacy
        single-env osmesa path (num_envs==1) is untouched."""
        main_extractor = getattr(self, "_oft_action_hidden_extractor", None)
        if main_extractor is None:
            raise RuntimeError(
                "vectorized cotrain (num_envs>1) requires an OFT action_hidden "
                "extractor; none was built (non-OFT encoder?). Use num_envs=1."
            )
        # One extractor per slot (isolated history); reuse the main one for slot 0.
        extractors = [main_extractor] + [
            self._build_oft_action_hidden_extractor(cfg) for _ in range(num_envs - 1)
        ]

        # Children render with `render_backend` (egl isolated per child -> no robosuite
        # read_pixels SIGABRT); spawn does not inherit runtime env edits, so pass them.
        env_vars = {
            k: os.environ[k]
            for k in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "DVLA_DATA_ROOT", "LIBERO_CONFIG_PATH")
            if k in os.environ
        }
        env_vars["MUJOCO_GL"] = render_backend
        # PyOpenGL's platform must match mujoco's GL backend. The parent forces
        # PYOPENGL_PLATFORM=osmesa at module import; a spawned egl child would inherit
        # that osmesa and mujoco's egl init raises "Cannot use EGL rendering platform"
        # (PYOPENGL_PLATFORM must be unset or 'egl'). Pair them — RLinf exports
        # MUJOCO_GL=egl + PYOPENGL_PLATFORM=egl together.
        env_vars["PYOPENGL_PLATFORM"] = render_backend

        image_size = int(OmegaConf.select(cfg, "env.image_size", default=64))
        cfg_kwargs = {
            **self._env_cfg_kwargs(cfg),
            "full_record": True,
            "image_size": image_size,
        }
        if self.distributed.is_main_process:
            print(
                f"[online-cotrain] vectorized rollout: {num_envs} envs, "
                f"render_backend={render_backend}",
                flush=True,
            )
        vec = build_rollout_vec_env(
            render_backend=render_backend,
            num_envs=num_envs,
            cfg_kwargs=cfg_kwargs,
            env_vars=env_vars,
        )
        try:
            def _train_hook(
                env_step: int,
                active_episode_steps: list[int] | tuple[int, ...] | None = None,
            ) -> bool:
                return self._run_training_bursts(
                    env_step,
                    total_env_steps,
                    replay=replay,
                    env_task_ids=env_task_ids,
                    knobs=knobs,
                    counters=counters,
                    history=history,
                    active_episode_steps=active_episode_steps,
                )

            self._vectorized_cotrain_rollout(
                vec=vec,
                extractors=extractors,
                replay=replay,
                num_envs=num_envs,
                total_env_steps=total_env_steps,
                episode_horizon=episode_horizon,
                action_steps=None,  # full-chunk open-loop, matching the single-env path
                image_size=image_size,
                task_ids=env_task_ids,
                train_hook=_train_hook,
                counters=counters,
            )
        finally:
            try:
                vec.close()
            except Exception:
                pass
        if self.distributed.is_main_process:
            self._save_cotrain_ckpt()
        return history

    @torch.no_grad()
    def _vectorized_cotrain_rollout(
        self,
        *,
        vec: Any,
        extractors: list,
        replay: Any,
        num_envs: int,
        total_env_steps: int,
        episode_horizon: int,
        action_steps: int | None,
        image_size: int,
        task_ids,
        train_hook=None,
        counters: dict[str, int] | None = None,
    ) -> None:
        """Continuous K-slot online rollout over ``vec`` (VecRolloutEnv).

        Mirrors ``vectorized_collect.collect_vectorized``: one action queue and one
        OFT extractor per slot (isolated history), all slots stepped in parallel via
        the send-all/recv-all barrier; finished slots refill immediately (infinite
        episodes until ``total_env_steps``). Transitions are rebuilt in the parent
        from each child ``full_record`` (env lives in the child) and pushed to
        ``replay``. ``train_hook(env_step, active_episode_steps) -> stop``
        interleaves the training burst per env-step exactly like the single-env
        loop; it is None in unit tests."""
        if counters is None:
            counters = {
                "n_episodes": 0,
                "n_success": 0,
                "current_episodes": 0,
                "current_success": 0,
            }
        task_cycle = [int(t) for t in task_ids] or [0]
        episodes: list[list[dict[str, Any]]] = [[] for _ in range(num_envs)]
        slot_task = [-1] * num_envs
        slot_desc = [""] * num_envs
        slot_ep = [-1] * num_envs
        slot_step = [0] * num_envs
        # Per-slot WM belief + last executed action: the trained actor drives the env via
        # the WM latent (sequential observe_next), so each slot carries its own latent and
        # previous action (mirrors the single-env loop's latent/prev_action threading).
        slot_latent: list[Any] = [None] * num_envs
        slot_prev_action: list[Any] = [None] * num_envs
        ep_counter = 0

        def _start_slot(k: int) -> dict[str, Any]:
            nonlocal ep_counter
            tid = task_cycle[ep_counter % len(task_cycle)]
            ep = ep_counter
            ep_counter += 1
            if slot_task[k] != tid:  # reconfigure env only on task boundary
                slot_desc[k] = vec.set_task([tid], env_ids=[k])[0]
                slot_task[k] = tid
            rec = vec.reset([tid], [ep], env_ids=[k])[0]
            extractors[k].reset()
            episodes[k] = []
            slot_ep[k] = ep
            slot_step[k] = 0
            slot_latent[k] = None       # fresh WM belief at episode start (is_first)
            slot_prev_action[k] = None
            return rec

        recs: list[Any] = [None] * num_envs
        for k in range(num_envs):
            recs[k] = _start_slot(k)

        env_step = 0
        ids = list(range(num_envs))
        while env_step < total_env_steps:
            # Per slot: the OFT extractor supplies the action_hidden (the obs_embedding the
            # WM + classifier consume); the TRAINED actor produces the EXECUTED action from
            # the WM latent. (Was: the frozen OFT base action — which left the actor PPO
            # optimizes undeployed.)
            obs_embs: list[Any] = [None] * num_envs
            actions = []
            for k in ids:
                _chunk, flat_hidden = extractors[k].step(
                    extractor_obs_from_record(recs[k]), slot_desc[k]
                )
                obs_embs[k] = flat_hidden.reshape(1, -1).to(self.device).float()
                act, slot_latent[k] = self._actor_action_and_latent(
                    self.world_model,
                    self.policy,
                    obs_embs[k],
                    slot_latent[k],
                    slot_prev_action[k],
                    is_first=(slot_step[k] == 0),
                )
                actions.append(act)
            step_results = vec.step(actions, env_ids=ids)
            for k in ids:
                reward, terminated, truncated, info, rec_after = step_results[k]
                wm_action = np.asarray(
                    info.get("wm_action", actions[k]), dtype=np.float32
                ).reshape(-1)[:7]
                emb = obs_embs[k].reshape(-1).detach().cpu().float().numpy()
                tr = build_cotrain_replay_transition(
                    recs[k],
                    np.asarray(emb, dtype=np.float32),
                    wm_action,
                    reward,
                    terminated,
                    truncated,
                    task_id=slot_task[k],
                    task_description=slot_desc[k],
                    step=slot_step[k],
                    is_first=(slot_step[k] == 0),
                    image_size=image_size,
                )
                episodes[k].append(tr)
                # The next WM observe_next for this slot uses the action just executed.
                slot_prev_action[k] = (
                    torch.from_numpy(wm_action).to(self.device, dtype=obs_embs[k].dtype).unsqueeze(0)
                )
                slot_step[k] += 1
                env_step += 1
                recs[k] = rec_after
                done = bool(terminated or truncated) or slot_step[k] >= episode_horizon
                if done:
                    rec_added = replay.add_episode(episodes[k])
                    if rec_added is not None:
                        counters["n_episodes"] += 1
                        success = bool(rec_added["success"])
                        counters["n_success"] += int(success)
                        counters["current_episodes"] += 1
                        counters["current_success"] += int(success)
                        self.console_record_success(success)
                    recs[k] = _start_slot(k)
                # The rollout runs under the method-level no_grad, but the training
                # burst (forward + backward) must build a graph — re-enable grad
                # just for the hook, matching the single-env loop where no_grad is
                # scoped to ``_rollout_action`` only.
                if train_hook is not None:
                    with torch.enable_grad():
                        stop = train_hook(env_step, list(slot_step))
                    if stop:
                        return
            self.console_progress(
                min(env_step, total_env_steps), total_env_steps, "cotrain", unit="env"
            )

    def _save_cotrain_ckpt(self) -> None:
        # Torch checkpoint (the resume artifact: all 4 modules + their optimizers
        # + global_step + classifier_threshold) goes through the inherited base
        # saver, which also emits the HF sidecars via _save_checkpoint_sidecars.
        if self.checkpoint_save_torch():
            path = Path(self.save_checkpoint())
        elif self.checkpoint_save_hf():
            # HF-only export (not resumable): write just the sidecars.
            path = self.get_checkpoint_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._save_checkpoint_sidecars(path, payload={})
        else:
            return
        print(f"[online-cotrain] ckpt -> {path.parent}", flush=True)

    def _save_checkpoint_sidecars(self, path: Path, payload: dict) -> None:
        # Portable per-component HF artifacts next to the torch checkpoint, e.g.
        # checkpoints/latest_hf_world_model. Self-contained (weights in
        # model.safetensors); the policy block drops its external action-head path.
        if not self.checkpoint_save_hf():
            return
        ckpt_dir = path.parent
        stem = path.stem
        for name, module, cfg_key in (
            ("world_model", self.world_model, "world_model"),
            ("policy", self.policy, "policy"),
            ("critic", self.critic, "critic"),
        ):
            blk = OmegaConf.to_container(OmegaConf.select(self.cfg, cfg_key), resolve=True)
            target = blk.pop("_target_")
            if name == "policy":
                blk.pop("init_action_head_ckpt", None)
            save_module_pretrained(
                _unwrap(module),
                str(ckpt_dir / f"{stem}_hf_{name}"),
                target=target,
                init_args=blk,
            )
        save_module_pretrained(
            _unwrap(self.classifier),
            str(ckpt_dir / f"{stem}_hf_classifier"),
            target=str(
                getattr(
                    self,
                    "_classifier_target",
                    "dreamervla.models.reward.LatentSuccessClassifier",
                )
            ),
            init_args=getattr(self, "_classifier_cls_kwargs", {}),
        )
