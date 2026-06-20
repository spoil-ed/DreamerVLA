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

from dreamervla.algorithms.dreamervla import world_model_pretrain_step
from dreamervla.algorithms.ppo import dino_wmpo_outcome_step
from dreamervla.envs.train_env import DreamerVLAOnlineTrainEnv
from dreamervla.models.reward import (
    LatentSuccessClassifier,
    LatentSuccessClassifierConfig,
)
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
    SuccessTracker,
    obs_to_action_hidden,
    obs_to_input_token_embedding,
)
from dreamervla.utils.console import fmt_value, metric_box
from dreamervla.utils.optim import build_optimizer
from dreamervla.utils.torch_utils import freeze_module



class OnlineCotrainRunner(DreamerVLARunner):
    """Hydra runner for the unified online cotrain pipeline (see module docstring)."""

    runner_name = "online_cotrain"
    runner_status = "current"
    runner_family = "actor"

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
            OmegaConf.select(cfg, "algorithm.wmpo.classifier_threshold", default=0.5)
            or 0.5
        )
        cls_blob = OmegaConf.select(cfg, "classifier", default=None)
        cls_kwargs: dict[str, Any] = (
            dict(OmegaConf.to_container(cls_blob, resolve=True))
            if cls_blob is not None
            else {}
        )
        cls_kwargs.pop("_target_", None)
        # Default latent_dim to the WM obs dim, EXCEPT when token_pool="mean":
        # then the classifier pools tokens and its latent_dim must stay token_dim
        # (the flat obs_dim would make the input projection enormous, e.g. the
        # 512*4096 backbone latent -> a multi-billion-param Linear).
        if cls_kwargs.get("latent_dim") is None and str(
            cls_kwargs.get("token_pool", "flat")
        ) != "mean":
            cls_kwargs["latent_dim"] = int(OmegaConf.select(cfg, "world_model.obs_dim"))
        cls_cfg = LatentSuccessClassifierConfig(**cls_kwargs)
        classifier = LatentSuccessClassifier(cls_cfg).to(self.device)
        warm = OmegaConf.select(cfg, "init.classifier_state_ckpt", default=None)
        if warm:
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
        self._cls_window = int(getattr(cls_cfg, "window", 8))

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

    def _build_env(self, cfg: DictConfig) -> DreamerVLAOnlineTrainEnv:
        env_cfg = OmegaConf.select(cfg, "env", default={}) or {}
        task_ids = OmegaConf.select(env_cfg, "task_ids", default=None)
        task_ids = [int(x) for x in task_ids] if task_ids is not None else None
        seed = int(OmegaConf.select(cfg, "seed", default=7)) + self._rank * 1000
        return DreamerVLAOnlineTrainEnv(
            {
                "task_suite_name": str(
                    OmegaConf.select(env_cfg, "task_suite_name", default="libero_goal")
                ),
                "task_id": int(task_ids[0]) if task_ids else 0,
                "task_ids": task_ids,
                "seed": seed,
                "max_steps": int(
                    OmegaConf.select(env_cfg, "episode_horizon", default=200)
                ),
                "action_input": "normalized",
                "history_length": int(
                    OmegaConf.select(env_cfg, "history_length", default=2)
                ),
                "include_state": bool(
                    OmegaConf.select(env_cfg, "include_state", default=True)
                ),
                "vla_rotate_180": bool(
                    OmegaConf.select(env_cfg, "vla_rotate_180", default=True)
                ),
                "obs_hidden_source": str(
                    OmegaConf.select(env_cfg, "obs_hidden_source", default="action_query")
                ),
                "action_head_type": str(
                    OmegaConf.select(env_cfg, "action_head_type", default="legacy")
                ),
            }
        )

    @torch.no_grad()
    def _rollout_action(self, world_model, policy, processor, obs, latent, prev_action, target_token_id):
        """One env-step action via WM latent -> policy sample. Mirrors
        online_dreamervla. Returns (policy_action[7], obs_embedding, latent).

        ``obs_embedding`` is the latent the WM consumes: the post-Action-Query
        action-hidden (``action_hidden``) or the pre-Action-Query backbone
        input-token latent (``backbone_latent``)."""
        is_first = bool(obs.get("is_first", False)) or latent is None
        if getattr(self, "_latent_type", "action_hidden") == "backbone_latent":
            oft_extractor = getattr(self, "_oft_input_token_extractor", None)
            if oft_extractor is not None:
                if is_first and hasattr(oft_extractor, "reset"):
                    oft_extractor.reset()
                _action_chunk, flat_hidden = oft_extractor.step(
                    obs,
                    str(obs.get("task_description", "")),
                )
                obs_embedding = flat_hidden.reshape(1, -1).to(self.device).float()
            else:
                obs_embedding = obs_to_input_token_embedding(
                    self.encoder, processor, obs, self.device, getattr(self, "_num_views", 2)
                )
        else:
            obs_embedding = obs_to_action_hidden(
                self.encoder, processor, obs, self.device, target_token_id
            )
        if is_first:
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
        return np.asarray(chunk[0][:7], dtype=np.float32), obs_embedding, latent

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

    def _build_components(self, cfg: DictConfig) -> None:
        # ---- components (reuse hydra targets + inherited helpers; no offline run() touch)
        total_env_steps = int(OmegaConf.select(cfg, "online_rollout.total_env_steps", default=1))
        if total_env_steps <= 0:
            self.encoder = None
            self.processor = None
            self._oft_input_token_extractor = None
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

    def _online_cotrain_loop(self, cfg: DictConfig) -> list:  # noqa: C901
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
        target_token_id = int(OmegaConf.select(cfg, "env.target_token_id", default=10004))
        seq_len = int(OmegaConf.select(oc, "sequence_length", default=24))
        batch_size = int(OmegaConf.select(cfg, "dataloader.batch_size", default=4))
        min_replay = int(OmegaConf.select(oc, "min_replay", default=seq_len * batch_size))
        min_eps = int(OmegaConf.select(oc, "min_episodes_per_task", default=1))
        train_every = int(OmegaConf.select(oc, "train_every", default=8))
        updates_per_train = int(OmegaConf.select(oc, "updates_per_train", default=1))
        total_env_steps = int(OmegaConf.select(oc, "total_env_steps", default=200000))
        max_train_updates = OmegaConf.select(oc, "max_train_updates", default=None)
        buffer_size = int(OmegaConf.select(oc, "buffer_size", default=20000))
        episode_horizon = int(OmegaConf.select(cfg, "env.episode_horizon", default=200))
        optim_cfg = OmegaConf.select(cfg, "optim")
        early_neg_stride = int(OmegaConf.select(oc, "classifier_early_neg_stride", default=8))
        ckpt_every = int(OmegaConf.select(cfg, "training.checkpoint_every", default=2000))

        if bool(OmegaConf.select(cfg, "training.debug", default=False)):
            total_env_steps = int(OmegaConf.select(oc, "debug_total_env_steps", default=64))
            warmup_steps = int(OmegaConf.select(oc, "debug_warmup_steps", default=2))
            min_replay = int(OmegaConf.select(oc, "debug_min_replay", default=seq_len))
            max_train_updates = int(OmegaConf.select(oc, "debug_max_train_updates", default=4))
            episode_horizon = int(OmegaConf.select(oc, "debug_episode_horizon", default=30))
            OmegaConf.update(cfg, "env.episode_horizon", episode_horizon)

        env_task_ids = OmegaConf.select(cfg, "env.task_ids", default=[0]) or [0]
        env_task_ids = tuple(int(x) for x in env_task_ids)
        env = self._build_env(cfg)
        replay = OnlineReplay(
            capacity=buffer_size,
            sequence_length=seq_len,
            task_ids=env_task_ids,
            rank=self._rank,
        )
        is_dist = self._world_size > 1
        if self.distributed.is_main_process:
            print(
                f"[online-cotrain] warmup_steps={warmup_steps} "
                f"train_actor_after={train_actor_after} train_cls_inline={train_cls_inline} "
                f"buffer_size={buffer_size} seq_len={seq_len} train_every={train_every} "
                f"episode_horizon={episode_horizon} world_size={self._world_size}",
                flush=True,
            )

        history: list[dict[str, float | str | int]] = []
        os.makedirs(os.path.join(self.output_dir, "ckpt"), exist_ok=True)
        obs, _info = env.reset()
        latent: Any = None
        prev_action: torch.Tensor | None = None
        episode: list[dict[str, Any]] = []
        n_episodes = 0
        n_success = 0
        stop = False
        success_window = int(OmegaConf.select(cfg, "console.success_window", default=50))
        tracker = SuccessTracker(window=success_window)
        log_every = int(OmegaConf.select(cfg, "console.log_every", default=1))
        banner_width = int(OmegaConf.select(cfg, "console.banner_width", default=65))
        update_idx = 0

        for env_step in range(1, total_env_steps + 1):
            if stop:
                break
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
                    n_episodes += 1
                    success = bool(rec["success"])
                    n_success += int(success)
                    tracker.update(success)
                episode = []
                obs, _info = env.reset()
                latent, prev_action = None, None

            # ---- training bursts (lockstep across ranks via global-ready flag)
            _stats, _cov_ready, all_ready = get_replay_task_stats_global(
                replay,
                task_ids=env_task_ids,
                min_transitions=min_replay,
                min_episodes_per_task=min_eps,
                device=self.device,
                is_dist=is_dist,
                world_size=self._world_size,
            )
            num_updates = 0
            if all_ready and env_step % train_every == 0:
                num_updates = updates_per_train
            for _ in range(num_updates):
                if max_train_updates is not None and self.global_step >= int(max_train_updates):
                    stop = True
                    break
                in_warmup = self.global_step < warmup_steps
                metrics: dict[str, float | str | int] = {
                    "global_step": int(self.global_step),
                    "phase": "warmup" if in_warmup else "cotrain",
                    "rollout/success_rate": (n_success / n_episodes) if n_episodes else 0.0,
                    "rollout/success_rate_windowed": tracker.rate(),
                    "buffer/size": float(replay.num_transitions),
                }
                # Phase WM (always) — policy/actor frozen (eval + no policy optim step)
                self.world_model.train()
                self.policy.eval()
                self.critic.eval()
                _unwrap(self.classifier).eval()
                wm_batch = self._build_wm_pretrain_batch(replay.sample(batch_size))
                if wm_batch is not None:
                    wm_metrics = world_model_pretrain_step(
                        policy=self.policy,
                        world_model=self.world_model,
                        optimizer=self.world_model_optimizer,
                        batch=wm_batch,
                        device=self.device,
                        optim_cfg=optim_cfg,
                    )
                    metrics["wm/loss"] = float(wm_metrics.get("loss", 0.0))

                # Phase CLS (always) — WM/actor frozen; classifier trains
                if train_cls_inline and replay.classifier_window_count(
                    window=self._cls_window,
                    chunk_size=int(getattr(_unwrap(self.classifier).cfg, "chunk_size", 1)),
                ) > 0:
                    cls_metrics = online_classifier_update_step(
                        classifier=self.classifier,
                        optimizer=self.classifier_optimizer,
                        replay=replay,
                        device=self.device,
                        batch_size=cls_bs,
                        early_neg_stride=early_neg_stride,
                        grad_clip=float(OmegaConf.select(optim_cfg, "grad_clip_norm", default=1.0)),
                    )
                    metrics["cls/loss"] = float(cls_metrics["loss"])
                    metrics["cls/acc"] = float(cls_metrics["acc"])
                    metrics["cls/f1"] = float(cls_metrics["f1"])

                # Phase RL (cotrain only) — WM + classifier frozen; slow policy
                if (not in_warmup) and train_actor_after:
                    self.world_model.eval()
                    _unwrap(self.classifier).eval()
                    assert not _unwrap(self.world_model).training, "WM must be frozen in RL phase"
                    assert not _unwrap(self.classifier).training, "classifier must be frozen in RL phase"
                    rl_batch = replay.sample(batch_size)
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
                    ac_metrics = dino_wmpo_outcome_step(
                        policy=self.policy,
                        chunk_world_model=self.world_model,
                        classifier=_unwrap(self.classifier),  # predict_success: DDP wrapper hides it
                        classifier_threshold=self.classifier_threshold,
                        actor_optimizer=self.policy_optimizer,
                        obs=obs_for_update,
                        device=self.device,
                        algorithm_cfg=algo,
                        optim_cfg=optim_cfg,
                        ref_policy=self.ref_policy,
                    )
                    metrics["rl/actor_loss"] = float(ac_metrics.get("actor_loss", 0.0))
                    metrics["rl/returns_mean"] = float(ac_metrics.get("returns_mean", 0.0))
                    metrics["rl/policy_grad_norm"] = float(ac_metrics.get("actor_grad_norm", 0.0))

                if self.distributed.is_main_process:
                    update_idx += 1
                    if update_idx % log_every == 0:
                        rows = []
                        if not in_warmup:
                            rows.append(
                                f"VLA    succ@{success_window}={fmt_value(tracker.rate())} "
                                f"(Δ {tracker.delta():+.3f} · best {tracker.best:.3f})   "
                                f"return={fmt_value(metrics.get('rl/returns_mean', 0.0))}"
                            )
                        rows.append(
                            f"train  wm={fmt_value(metrics.get('wm/loss', float('nan')))}  "
                            f"actor={fmt_value(metrics.get('rl/actor_loss', float('nan')))}  "
                            f"cls_acc={fmt_value(metrics.get('cls/acc', float('nan')))}"
                        )
                        rows.append(
                            f"data   buf={fmt_value(metrics['buffer/size'])}  "
                            f"ep={n_episodes}  cum_succ={fmt_value(metrics['rollout/success_rate'])}"
                        )
                        header = f"{metrics['phase']} · step {self.global_step}"
                        print(metric_box(header, rows, width=banner_width), flush=True)
                        tracker.mark_printed()
                    self.log_metrics(metrics, step=int(self.global_step))
                history.append(metrics)
                if self.distributed.is_main_process and ckpt_every > 0 and (self.global_step + 1) % ckpt_every == 0:
                    self._save_cotrain_ckpt()
                self.global_step += 1

        if self.distributed.is_main_process:
            self._save_cotrain_ckpt()
        try:
            env.close()
        except Exception:
            pass
        return history

    def _save_cotrain_ckpt(self) -> None:
        path = os.path.join(self.output_dir, "ckpt", "latest.ckpt")
        torch.save(
            {
                "global_step": int(self.global_step),
                "world_model": _unwrap(self.world_model).state_dict(),
                "policy": _unwrap(self.policy).state_dict(),
                "critic": _unwrap(self.critic).state_dict(),
                "classifier": _unwrap(self.classifier).state_dict(),
                "classifier_threshold": float(self.classifier_threshold),
            },
            path,
        )
        print(f"[online-cotrain] ckpt -> {path}", flush=True)
