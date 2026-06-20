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
from dreamervla.utils.console import count_trainable, phase_banner


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

    def _save_wm_warmup(self) -> None:
        torch.save({"global_step": int(self.global_step),
                    "world_model": _unwrap(self.world_model).state_dict()}, self._wm_warmup_ckpt())

    def _save_cls_warmup(self) -> None:
        torch.save({"global_step": int(self.global_step),
                    "classifier": _unwrap(self.classifier).state_dict(),
                    "classifier_threshold": float(self.classifier_threshold)}, self._cls_warmup_ckpt())

    # ------------------------------------------------------------------ main
    def run(self) -> list[dict[str, Any]]:
        import copy

        from dreamervla.runners.online_replay import OnlineReplay

        cfg = copy.deepcopy(self.cfg)
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

        # warmup knobs
        ow = OmegaConf.select(cfg, "offline_warmup", default={}) or {}
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

        if bool(OmegaConf.select(cfg, "training.debug", default=False)):
            wm_steps = int(OmegaConf.select(ow, "debug_wm_warmup_steps", default=2))
            cls_steps = int(OmegaConf.select(ow, "debug_classifier_warmup_steps", default=2))

        need_wm = not (resume and os.path.exists(self._wm_warmup_ckpt()))
        need_cls = not (resume and os.path.exists(self._cls_warmup_ckpt()))

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
            if self.distributed.is_main_process:
                print(phase_banner("[1/3] WM WARMUP", subtitle=f"{wm_steps} steps"), flush=True)
            wm_last = self._offline_warmup_wm(warmup_replay, steps=wm_steps, batch_size=bs, optim_cfg=optim_cfg)
            if self.distributed.is_main_process:
                self._save_wm_warmup()
                print(phase_banner("[1/3] WM WARMUP", subtitle=f"wm_loss {wm_last:.3f}", done=True), flush=True)
        else:
            payload = torch.load(self._wm_warmup_ckpt(), map_location="cpu", weights_only=False)
            _unwrap(self.world_model).load_state_dict(payload["world_model"])

        if need_cls:
            if self.distributed.is_main_process:
                print(phase_banner("[2/3] CLASSIFIER WARMUP", subtitle=f"{cls_steps} steps"), flush=True)
            cls_last = self._offline_warmup_classifier(warmup_replay, steps=cls_steps, batch_size=cls_bs,
                                            early_neg_stride=early_neg_stride, grad_clip=grad_clip)
            if self.distributed.is_main_process:
                self._save_cls_warmup()
                print(phase_banner("[2/3] CLASSIFIER WARMUP", subtitle=f"acc {cls_last:.3f}", done=True), flush=True)
        else:
            payload = torch.load(self._cls_warmup_ckpt(), map_location="cpu", weights_only=False)
            _unwrap(self.classifier).load_state_dict(payload["classifier"])
            self.classifier_threshold = float(payload.get("classifier_threshold", self.classifier_threshold))

        # online cotrain with RL from the start (already warm): force warmup_steps=0.
        # Debug runs would otherwise re-read online_rollout.debug_warmup_steps in the online
        # loop, re-defeating the 0 — zero it too so the "already warm" intent holds in every mode.
        OmegaConf.update(cfg, "training.warmup_steps", 0, force_add=True)
        OmegaConf.update(cfg, "online_rollout.debug_warmup_steps", 0, force_add=True)
        self.cfg = cfg
        total_env_steps = int(OmegaConf.select(cfg, "online_rollout.total_env_steps", default=0))
        if total_env_steps <= 0:
            if self.distributed.is_main_process:
                print(phase_banner("[3/3] ONLINE COTRAIN", subtitle="skipped · total_env_steps=0", done=True), flush=True)
            return []
        if self.distributed.is_main_process:
            print(phase_banner("[3/3] ONLINE COTRAIN", subtitle=f"{total_env_steps} env steps"), flush=True)
        return self._online_cotrain_loop(cfg)
