"""Offline-warmup -> online-cotrain pipeline runner.

Pre-seeds the OnlineReplay buffer from previously-collected cold-start
trajectory HDF5, warms up the world model + success classifier on that unified
buffer (same step functions as the online phase, so zero semantic drift), then
runs the existing OnlineCotrainRunner online loop with RL enabled. WM and
classifier warmup checkpoints are saved separately for independent resume.

See docs/superpowers/specs/2026-06-17-offline-warmup-online-cotrain-pipeline-design.md
"""
from __future__ import annotations

import os
from typing import Any

import torch
from omegaconf import OmegaConf

from dreamervla.algorithms.dreamervla import world_model_pretrain_step
from dreamervla.runners.offline_seed import seed_replay_from_offline
from dreamervla.runners.online_cotrain_runner import OnlineCotrainRunner
from dreamervla.runners.online_dreamervla import _unwrap, online_classifier_update_step
from dreamervla.utils.console import count_trainable
from dreamervla.utils.hf_module import load_module_pretrained, save_module_pretrained


class OnlineCotrainPipelineRunner(OnlineCotrainRunner):
    """Offline-seeded warmup then online cotrain (see module docstring)."""

    runner_name = "online_cotrain_pipeline"
    runner_status = "current"
    runner_family = "actor"

    # ------------------------------------------------------------------ warmup
    def _offline_warmup_wm(self, replay, *, steps: int, batch_size: int, optim_cfg) -> float:
        self.world_model.train()
        last = 0.0
        for i in range(int(steps)):
            wm_batch = self._build_wm_pretrain_batch(replay.sample(batch_size))
            if wm_batch is None:
                continue
            m = world_model_pretrain_step(
                policy=self.policy,
                world_model=self.world_model,
                optimizer=self.world_model_optimizer,
                batch=wm_batch,
                device=self.device,
                optim_cfg=optim_cfg,
            )
            last = float(m.get("loss", 0.0))
            if i % 50 == 0:
                print(f"[pipeline][wm-warmup] step={i}/{steps} loss={last:.4f}", flush=True)
        return last

    def _offline_warmup_classifier(
        self, replay, *, steps: int, batch_size: int, early_neg_stride: int, grad_clip: float
    ) -> float:
        last_acc = 0.0
        for i in range(int(steps)):
            m = online_classifier_update_step(
                classifier=self.classifier,
                optimizer=self.classifier_optimizer,
                replay=replay,
                device=self.device,
                batch_size=batch_size,
                early_neg_stride=early_neg_stride,
                grad_clip=grad_clip,
            )
            last_acc = float(m["acc"])
            if i % 50 == 0:
                print(f"[pipeline][cls-warmup] step={i}/{steps} loss={float(m['loss']):.4f} acc={last_acc:.3f}", flush=True)
        return last_acc

    # ------------------------------------------------------------ split ckpts
    def _wm_warmup_ckpt(self) -> str:
        return os.path.join(self.output_dir, "ckpt", "wm_warmup.ckpt")

    def _cls_warmup_ckpt(self) -> str:
        return os.path.join(self.output_dir, "ckpt", "classifier_warmup.ckpt")

    def _wm_warmup_hf_dir(self) -> str:
        return os.path.join(self.output_dir, "ckpt", "wm_warmup_hf")

    def _cls_warmup_hf_dir(self) -> str:
        return os.path.join(self.output_dir, "ckpt", "classifier_warmup_hf")

    def _save_wm_warmup(self) -> None:
        if self.checkpoint_save_torch():
            torch.save({"global_step": int(self.global_step),
                        "world_model": _unwrap(self.world_model).state_dict()}, self._wm_warmup_ckpt())
        if self.checkpoint_save_hf():
            wm_cfg = OmegaConf.to_container(OmegaConf.select(self.cfg, "world_model"), resolve=True)
            target = wm_cfg.pop("_target_")
            save_module_pretrained(_unwrap(self.world_model), self._wm_warmup_hf_dir(),
                                   target=target, init_args=wm_cfg)

    def _save_cls_warmup(self) -> None:
        if self.checkpoint_save_torch():
            torch.save({"global_step": int(self.global_step),
                        "classifier": _unwrap(self.classifier).state_dict(),
                        "classifier_threshold": float(self.classifier_threshold)}, self._cls_warmup_ckpt())
        if self.checkpoint_save_hf():
            cls_kwargs = getattr(self, "_classifier_cls_kwargs", {})
            save_module_pretrained(
                _unwrap(self.classifier),
                self._cls_warmup_hf_dir(),
                target="dreamervla.models.reward.latent_success_classifier.LatentSuccessClassifier",
                init_args=cls_kwargs,
            )

    # ------------------------------------------------------------- debug swap
    @staticmethod
    def _apply_debug_overrides(cfg) -> None:
        """When training.debug is set, swap every full knob for its debug_* value.

        Applied once at the top of run() (force_add) so every downstream read —
        warmup steps and the online loop alike — sees the small smoke values.
        """
        if not bool(OmegaConf.select(cfg, "training.debug", default=False)):
            return
        # full key -> debug key + fallback when the debug key is absent
        swaps = [
            ("training.wm_warmup_steps", "offline_warmup.debug_wm_warmup_steps", 2),
            ("training.classifier_warmup_steps", "offline_warmup.debug_classifier_warmup_steps", 2),
            ("online_rollout.total_env_steps", "online_rollout.debug_total_env_steps", 160),
            ("online_rollout.max_train_updates", "online_rollout.debug_max_train_updates", 4),
            ("online_rollout.episode_horizon", "online_rollout.debug_episode_horizon", 50),
            ("online_rollout.min_replay", "online_rollout.debug_min_replay", 48),
            ("dataloader.batch_size", "dataloader.debug_batch_size", 2),
            ("algorithm.imagination_horizon", "algorithm.debug_imagination_horizon", 3),
            ("algorithm.ppo_rollouts_per_start", "algorithm.debug_ppo_rollouts_per_start", 2),
            ("algorithm.wmpo.episode_max_steps", "algorithm.wmpo.debug_episode_max_steps", 150),
        ]
        for full_key, debug_key, fallback in swaps:
            value = OmegaConf.select(cfg, debug_key, default=fallback)
            if value is None:
                continue
            OmegaConf.update(cfg, full_key, value, force_add=True)

    # ------------------------------------------------------------------ main
    def run(self) -> list[dict[str, Any]]:
        import copy

        from dreamervla.runners.online_replay import OnlineReplay

        cfg = copy.deepcopy(self.cfg)
        self._apply_debug_overrides(cfg)
        latent_type = str(OmegaConf.select(cfg, "latent_type", default="action_hidden"))
        if latent_type not in ("action_hidden", "backbone_latent"):
            raise ValueError(f"unknown latent_type={latent_type!r}")
        self._latent_type = latent_type
        env_image_keys = OmegaConf.select(cfg, "env.image_keys", default=["agentview_rgb", "eye_in_hand_rgb"])
        self._num_views = len(list(env_image_keys)) if env_image_keys is not None else 2
        if latent_type == "backbone_latent":
            OmegaConf.update(cfg, "env.obs_hidden_source", "input_token_embedding", force_add=True)

        self._build_components(cfg)
        if self.distributed.is_main_process:
            trainable = {
                "world_model": count_trainable(self.world_model),
                "policy": count_trainable(self.policy),
                "critic": count_trainable(self.critic),
                "classifier": count_trainable(self.classifier),
            }
            total = sum(trainable.values())
            self.append_model_summary(
                {"total_trainable": total, "trainable_params": trainable}
            )
            print(f"[ok] model ready · {total/1e6:.1f}M trainable", flush=True)
        os.makedirs(os.path.join(self.output_dir, "ckpt"), exist_ok=True)

        # warmup knobs (debug values, if any, were applied by _apply_debug_overrides)
        wm_steps = int(OmegaConf.select(cfg, "training.wm_warmup_steps", default=2000))
        cls_steps = int(OmegaConf.select(cfg, "training.classifier_warmup_steps", default=2000))
        bs = int(OmegaConf.select(cfg, "dataloader.batch_size", default=4))
        cls_bs = int(OmegaConf.select(cfg, "training.classifier_batch_size", default=16))
        optim_cfg = OmegaConf.select(cfg, "optim")
        early_neg_stride = int(OmegaConf.select(cfg, "online_rollout.classifier_early_neg_stride", default=8))
        grad_clip = float(OmegaConf.select(optim_cfg, "grad_clip_norm", default=1.0))
        seq_len = int(OmegaConf.select(cfg, "online_rollout.sequence_length", default=24))
        buffer_size = int(OmegaConf.select(cfg, "online_rollout.buffer_size", default=20000))
        env_task_ids = tuple(int(x) for x in (OmegaConf.select(cfg, "env.task_ids", default=[0]) or [0]))
        default_task_id = OmegaConf.select(cfg, "offline_warmup.task_id", default=None)
        resume = bool(OmegaConf.select(cfg, "training.resume", default=False))

        need_wm = not (resume and (os.path.exists(self._wm_warmup_ckpt()) or os.path.isdir(self._wm_warmup_hf_dir())))
        need_cls = not (resume and (os.path.exists(self._cls_warmup_ckpt()) or os.path.isdir(self._cls_warmup_hf_dir())))

        warmup_replay = OnlineReplay(capacity=buffer_size, sequence_length=seq_len,
                                     task_ids=env_task_ids, rank=self._rank)
        if need_wm or need_cls:
            n = seed_replay_from_offline(
                warmup_replay,
                data_dir=OmegaConf.select(cfg, "offline_warmup.data_dir"),
                hidden_dir=OmegaConf.select(cfg, "offline_warmup.hidden_dir"),
                default_task_id=(int(default_task_id) if default_task_id is not None else None),
            )
            if self.distributed.is_main_process:
                print(f"[pipeline] seeded {n} offline episodes, {warmup_replay.num_transitions} transitions", flush=True)
            if n == 0 or warmup_replay.num_transitions == 0:
                raise RuntimeError("offline seeding produced an empty replay buffer")

        if need_wm:
            self.console_banner("[1/3] WM WARMUP", subtitle=f"{wm_steps} steps")
            wm_last = self._offline_warmup_wm(warmup_replay, steps=wm_steps, batch_size=bs, optim_cfg=optim_cfg)
            if self.distributed.is_main_process:
                self._save_wm_warmup()
                self.console_banner("[1/3] WM WARMUP", subtitle=f"wm_loss {wm_last:.3f}", done=True)
        else:
            if os.path.exists(self._wm_warmup_ckpt()):
                payload = torch.load(self._wm_warmup_ckpt(), map_location="cpu", weights_only=False)
                _unwrap(self.world_model).load_state_dict(payload["world_model"])
            elif os.path.isdir(self._wm_warmup_hf_dir()):
                src = load_module_pretrained(self._wm_warmup_hf_dir())
                _unwrap(self.world_model).load_state_dict(src.state_dict())

        if need_cls:
            self.console_banner("[2/3] CLASSIFIER WARMUP", subtitle=f"{cls_steps} steps")
            cls_last = self._offline_warmup_classifier(warmup_replay, steps=cls_steps, batch_size=cls_bs,
                                            early_neg_stride=early_neg_stride, grad_clip=grad_clip)
            if self.distributed.is_main_process:
                self._save_cls_warmup()
                self.console_banner("[2/3] CLASSIFIER WARMUP", subtitle=f"acc {cls_last:.3f}", done=True)
        else:
            if os.path.exists(self._cls_warmup_ckpt()):
                payload = torch.load(self._cls_warmup_ckpt(), map_location="cpu", weights_only=False)
                _unwrap(self.classifier).load_state_dict(payload["classifier"])
                self.classifier_threshold = float(payload.get("classifier_threshold", self.classifier_threshold))
            elif os.path.isdir(self._cls_warmup_hf_dir()):
                src = load_module_pretrained(self._cls_warmup_hf_dir())
                _unwrap(self.classifier).load_state_dict(src.state_dict())

        # online cotrain with RL from the start (already warm): force warmup_steps=0.
        # Debug runs would otherwise re-read online_rollout.debug_warmup_steps in the online
        # loop, re-defeating the 0 — zero it too so the "already warm" intent holds in every mode.
        OmegaConf.update(cfg, "training.warmup_steps", 0, force_add=True)
        OmegaConf.update(cfg, "online_rollout.debug_warmup_steps", 0, force_add=True)
        self.cfg = cfg
        total_env_steps = int(OmegaConf.select(cfg, "online_rollout.total_env_steps", default=0))
        if total_env_steps <= 0:
            self.console_banner("[3/3] ONLINE COTRAIN", subtitle="skipped · total_env_steps=0", done=True)
            return []
        self.console_banner("[3/3] ONLINE COTRAIN", subtitle=f"{total_env_steps} env steps")
        return self._online_cotrain_loop(cfg)
