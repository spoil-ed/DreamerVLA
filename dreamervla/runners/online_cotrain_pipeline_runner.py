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
import time
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import OmegaConf

from dreamervla.algorithms.dreamervla import world_model_pretrain_step
from dreamervla.runners.offline_seed import seed_replay_from_offline
from dreamervla.runners.online_cotrain_runner import OnlineCotrainRunner
from dreamervla.runners.online_dreamervla import _unwrap, online_classifier_update_step
from dreamervla.utils.console import count_trainable
from dreamervla.utils.hf_module import load_module_pretrained, save_module_pretrained


def _assert_offline_seed_present(*, data_dir: Any, hidden_dir: Any) -> None:
    """Fail fast if the collected cold-start dump is missing — BEFORE loading models.

    Warmup seeds the replay buffer from collected reward + hidden shards. Without
    this guard, ``run()`` would build the heavy WM/encoder/classifier first and only
    then crash inside ``seed_replay_from_offline``. Checking up front turns "load
    everything, then fail" into an immediate, actionable error.
    """
    reward = Path(str(data_dir)).expanduser()
    hidden = Path(str(hidden_dir)).expanduser()
    if not reward.is_dir() or not any(reward.glob("*.hdf5")):
        raise FileNotFoundError(
            f"offline warmup needs collected reward shards but found none under {reward} — "
            "run cold-start collection first, or set training.resume with existing warmup "
            "checkpoints to skip seeding."
        )
    if not hidden.is_dir() or not any(hidden.glob("*.hdf5")):
        raise FileNotFoundError(
            f"offline warmup needs collected hidden sidecars but found none under {hidden} — "
            "run cold-start collection first."
        )


class OnlineCotrainPipelineRunner(OnlineCotrainRunner):
    """Offline-seeded warmup then online cotrain (see module docstring)."""

    runner_name = "online_cotrain_pipeline"
    runner_status = "current"
    runner_family = "actor"

    # ------------------------------------------------------------------ warmup
    def _log_replay_warmup_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        """Log replay-warmup progress under the training namespace."""
        if hasattr(self, "log_metrics"):
            self.log_metrics(metrics, step=int(step))

    def _replay_warmup_log_every(self) -> int:
        """Return the configured replay-warmup metric cadence in learner updates."""
        cfg = getattr(self, "cfg", None)
        if cfg is None:
            return 1
        value = OmegaConf.select(cfg, "training.replay_warmup_log_every", default=1)
        return max(1, int(value))

    def _print_pipeline_event(self, message: str) -> None:
        """Print one pipeline progress line from rank 0 only."""
        distributed = getattr(self, "distributed", None)
        if distributed is not None and not bool(getattr(distributed, "is_main_process", True)):
            return
        print(message, flush=True)

    def _maybe_warmup_checkpoint(
        self,
        *,
        current: int,
        total: int,
        every: int,
        checkpoint_fn: Callable[[], None] | None,
        label: str,
    ) -> None:
        """Save an optional mid-warmup component checkpoint."""
        if checkpoint_fn is None or int(every) <= 0:
            return
        current_i = int(current)
        total_i = int(total)
        if current_i >= total_i or current_i % int(every) != 0:
            return
        checkpoint_fn()
        self._print_pipeline_event(
            f"[pipeline][{label}] checkpoint saved step={current_i}/{total_i}"
        )

    def _offline_warmup_wm(
        self,
        replay,
        *,
        steps: int,
        batch_size: int,
        optim_cfg,
        checkpoint_every: int = 0,
        checkpoint_fn: Callable[[], None] | None = None,
    ) -> float:
        self.world_model.train()
        last = 0.0
        for i in range(int(steps)):
            wm_batch = self._build_wm_pretrain_batch(
                replay.sample(batch_size, include_images=False)
            )
            if wm_batch is None:
                self.console_progress(i + 1, int(steps), "wm-warmup", unit="update")
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
            if i % self._replay_warmup_log_every() == 0:
                self._log_replay_warmup_metrics(
                    {"train/wm_warmup_loss": last},
                    step=i,
                )
                print(f"[pipeline][wm-warmup] step={i}/{steps} loss={last:.4f}", flush=True)
            self._maybe_warmup_checkpoint(
                current=i + 1,
                total=int(steps),
                every=checkpoint_every,
                checkpoint_fn=checkpoint_fn,
                label="wm-warmup",
            )
            self.console_progress(i + 1, int(steps), "wm-warmup", unit="update")
        return last

    def _offline_warmup_classifier(
        self,
        replay,
        *,
        steps: int,
        batch_size: int,
        early_neg_stride: int,
        grad_clip: float,
        log_step_offset: int = 0,
        checkpoint_every: int = 0,
        checkpoint_fn: Callable[[], None] | None = None,
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
            if i % self._replay_warmup_log_every() == 0:
                self._log_replay_warmup_metrics(
                    {
                        "train/classifier_warmup_loss": float(m["loss"]),
                        "train/classifier_warmup_acc": last_acc,
                        "train/classifier_warmup_f1": float(m.get("f1", 0.0)),
                        "train/classifier_warmup_pos_frac": float(m.get("pos_frac", 0.0)),
                    },
                    step=int(log_step_offset) + i,
                )
                print(
                    f"[pipeline][cls-warmup] step={i}/{steps} "
                    f"loss={float(m['loss']):.4f} acc={last_acc:.3f} "
                    f"f1={float(m.get('f1', 0.0)):.3f} "
                    f"pos={float(m.get('pos_frac', 0.0)):.3f}",
                    flush=True,
                )
            self._maybe_warmup_checkpoint(
                current=i + 1,
                total=int(steps),
                every=checkpoint_every,
                checkpoint_fn=checkpoint_fn,
                label="classifier-warmup",
            )
            self.console_progress(i + 1, int(steps), "classifier-warmup", unit="update")
        return last_acc

    @staticmethod
    def _steps_for_replay_epochs(replay, *, replay_epochs: int, batch_size: int) -> int:
        epochs = int(replay_epochs)
        if epochs <= 0:
            return 0
        windows = int(replay.sampleable_window_count())
        if windows <= 0:
            return 0
        return int(epochs) * max(1, (windows + int(batch_size) - 1) // int(batch_size))

    @staticmethod
    def _steps_for_classifier_replay_epochs(
        replay,
        *,
        replay_epochs: int,
        batch_size: int,
        window: int,
        chunk_size: int,
    ) -> int:
        epochs = int(replay_epochs)
        if epochs <= 0:
            return 0
        windows = int(
            replay.classifier_window_count(
                window=int(window),
                chunk_size=int(chunk_size),
            )
        )
        if windows <= 0:
            return 0
        return epochs * max(1, (windows + int(batch_size) - 1) // int(batch_size))

    @classmethod
    def _resolve_warmup_steps(
        cls,
        replay,
        *,
        wm_steps: int,
        cls_steps: int,
        replay_epochs: int,
        replay_max_steps: int,
        wm_batch_size: int,
        cls_batch_size: int,
        cls_window: int,
        cls_chunk_size: int,
    ) -> tuple[int, int]:
        epoch_count = int(replay_epochs)
        if epoch_count <= 0:
            return int(wm_steps), int(cls_steps)
        resolved_wm = cls._steps_for_replay_epochs(
            replay,
            replay_epochs=epoch_count,
            batch_size=int(wm_batch_size),
        )
        resolved_cls = cls._steps_for_classifier_replay_epochs(
            replay,
            replay_epochs=epoch_count,
            batch_size=int(cls_batch_size),
            window=int(cls_window),
            chunk_size=int(cls_chunk_size),
        )
        max_steps = int(replay_max_steps)
        if max_steps > 0:
            resolved_wm = min(resolved_wm, max_steps)
            resolved_cls = min(resolved_cls, max_steps)
        return resolved_wm, resolved_cls

    def _offline_warmup_alternating(
        self,
        replay,
        *,
        wm_steps: int,
        cls_steps: int,
        wm_batch_size: int,
        cls_batch_size: int,
        optim_cfg,
        early_neg_stride: int,
        grad_clip: float,
    ) -> tuple[float, float]:
        self.world_model.train()
        wm_last = 0.0
        cls_last = 0.0
        cls_loss = 0.0
        cls_f1 = 0.0
        cls_pos_frac = 0.0
        total = max(int(wm_steps), int(cls_steps))
        for i in range(total):
            if i < int(wm_steps):
                wm_batch = self._build_wm_pretrain_batch(
                    replay.sample(wm_batch_size, include_images=False)
                )
                if wm_batch is not None:
                    wm_metrics = world_model_pretrain_step(
                        policy=self.policy,
                        world_model=self.world_model,
                        optimizer=self.world_model_optimizer,
                        batch=wm_batch,
                        device=self.device,
                        optim_cfg=optim_cfg,
                    )
                    wm_last = float(wm_metrics.get("loss", 0.0))
            if i < int(cls_steps):
                cls_metrics = online_classifier_update_step(
                    classifier=self.classifier,
                    optimizer=self.classifier_optimizer,
                    replay=replay,
                    device=self.device,
                    batch_size=cls_batch_size,
                    early_neg_stride=early_neg_stride,
                    grad_clip=grad_clip,
                )
                cls_loss = float(cls_metrics.get("loss", 0.0))
                cls_last = float(cls_metrics["acc"])
                cls_f1 = float(cls_metrics.get("f1", 0.0))
                cls_pos_frac = float(cls_metrics.get("pos_frac", 0.0))
            if i % self._replay_warmup_log_every() == 0:
                self._log_replay_warmup_metrics(
                    {
                        "train/wm_warmup_loss": wm_last,
                        "train/classifier_warmup_loss": cls_loss,
                        "train/classifier_warmup_acc": cls_last,
                        "train/classifier_warmup_f1": cls_f1,
                        "train/classifier_warmup_pos_frac": cls_pos_frac,
                    },
                    step=i,
                )
                print(
                    f"[pipeline][replay-warmup] learner_update={i}/{total} "
                    f"wm_loss={wm_last:.4f} cls_loss={cls_loss:.4f} "
                    f"cls_acc={cls_last:.3f} cls_f1={cls_f1:.3f} "
                    f"cls_pos={cls_pos_frac:.3f}",
                    flush=True,
                )
            self.console_progress(i + 1, total, "replay-warmup", unit="update")
        return wm_last, cls_last

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
                target=str(
                    getattr(
                        self,
                        "_classifier_target",
                        "dreamervla.models.reward.LatentSuccessClassifier",
                    )
                ),
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
            ("training.warmup_replay_epochs", "offline_warmup.debug_warmup_replay_epochs", 0),
            ("online_rollout.total_env_steps", "online_rollout.debug_total_env_steps", 160),
            ("online_rollout.max_train_updates", "online_rollout.debug_max_train_updates", 4),
            ("online_rollout.episode_horizon", "online_rollout.debug_episode_horizon", 50),
            ("online_rollout.min_replay", "online_rollout.debug_min_replay", 48),
            ("dataloader.batch_size", "dataloader.debug_batch_size", 2),
            ("algorithm.imagination_horizon", "algorithm.debug_imagination_horizon", 3),
            ("algorithm.ppo_rollouts_per_start", "algorithm.debug_ppo_rollouts_per_start", 2),
            ("algorithm.lumos.ppo_rollouts_per_start_min", "algorithm.debug_ppo_rollouts_per_start", 2),
            ("algorithm.lumos.ppo_rollouts_per_start_max", "algorithm.debug_ppo_rollouts_per_start", 2),
            ("algorithm.lumos.episode_max_steps", "algorithm.lumos.debug_episode_max_steps", 150),
        ]
        for full_key, debug_key, fallback in swaps:
            value = OmegaConf.select(cfg, debug_key, default=fallback)
            if value is None:
                continue
            OmegaConf.update(cfg, full_key, value, force_add=True)

    # ------------------------------------------------------------- gpu reclaim
    def _empty_cuda_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _set_encoder_device(self, device: Any) -> None:
        """Move the frozen encoder on/off the GPU and reclaim freed blocks.

        The encoder is unused during warmup (warmup reads pre-seeded sidecar
        embeddings), so parking it on CPU frees its weights until the online phase
        restores it. Frozen + inference-only, so the round trip is numerically inert.
        No-op when there is no encoder (e.g. total_env_steps<=0 builds encoder=None).
        """
        encoder = getattr(self, "encoder", None)
        if encoder is not None:
            encoder.to(device)
        self._empty_cuda_cache()

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

        # Identify that the collected cold-start dump exists BEFORE loading the heavy
        # WM/encoder/classifier. When warmup will seed from offline shards (i.e. no full
        # warmup-ckpt resume), fail fast here instead of paying the model load only to
        # crash in seeding. A full resume needs no seeding, so the check is skipped.
        resume = bool(OmegaConf.select(cfg, "training.resume", default=False))
        need_wm = not (resume and (os.path.exists(self._wm_warmup_ckpt()) or os.path.isdir(self._wm_warmup_hf_dir())))
        need_cls = not (resume and (os.path.exists(self._cls_warmup_ckpt()) or os.path.isdir(self._cls_warmup_hf_dir())))
        if need_wm or need_cls:
            _assert_offline_seed_present(
                data_dir=OmegaConf.select(cfg, "offline_warmup.data_dir"),
                hidden_dir=OmegaConf.select(cfg, "offline_warmup.hidden_dir"),
            )

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
        warmup_replay_epochs = int(
            OmegaConf.select(cfg, "training.warmup_replay_epochs", default=0) or 0
        )
        warmup_replay_max_steps = int(
            OmegaConf.select(cfg, "training.warmup_replay_max_steps", default=0) or 0
        )
        warmup_checkpoint_every = int(
            OmegaConf.select(cfg, "training.warmup_checkpoint_every", default=0) or 0
        )
        bs = int(OmegaConf.select(cfg, "dataloader.batch_size", default=4))
        cls_bs = int(OmegaConf.select(cfg, "training.classifier_batch_size", default=16))
        optim_cfg = OmegaConf.select(cfg, "optim")
        early_neg_stride = int(OmegaConf.select(cfg, "online_rollout.classifier_early_neg_stride", default=8))
        grad_clip = float(OmegaConf.select(optim_cfg, "grad_clip_norm", default=1.0))
        seq_len = int(OmegaConf.select(cfg, "online_rollout.sequence_length", default=24))
        buffer_size = int(OmegaConf.select(cfg, "online_rollout.buffer_size", default=20000))
        replay_capacity_mode = str(
            OmegaConf.select(cfg, "online_rollout.replay_capacity_mode", default="per_task")
        )
        env_task_ids = tuple(int(x) for x in (OmegaConf.select(cfg, "env.task_ids", default=[0]) or [0]))
        default_task_id = OmegaConf.select(cfg, "offline_warmup.task_id", default=None)
        max_seed_eps = OmegaConf.select(cfg, "offline_warmup.max_episodes_per_task", default=None)
        # resume / need_wm / need_cls were computed above (before the heavy build) so the
        # offline-data existence check could fail fast; reuse them here.

        warmup_replay = OnlineReplay(
            capacity=buffer_size,
            sequence_length=seq_len,
            task_ids=env_task_ids,
            capacity_mode=replay_capacity_mode,
            rank=self._rank,
        )
        if need_wm or need_cls:
            data_dir = OmegaConf.select(cfg, "offline_warmup.data_dir")
            hidden_dir = OmegaConf.select(cfg, "offline_warmup.hidden_dir")
            max_seed_label = (
                "all"
                if max_seed_eps is None
                else f"<= {int(max_seed_eps)}/task"
            )
            self._print_pipeline_event(
                "[pipeline][replay] loading offline shards "
                f"data_dir={data_dir} hidden_dir={hidden_dir} "
                f"tasks={list(env_task_ids)} seq_len={seq_len} "
                f"capacity={buffer_size} capacity_mode={replay_capacity_mode} "
                f"episodes={max_seed_label}"
            )
            replay_load_start = time.perf_counter()
            n = seed_replay_from_offline(
                warmup_replay,
                data_dir=data_dir,
                hidden_dir=hidden_dir,
                default_task_id=(int(default_task_id) if default_task_id is not None else None),
                max_episodes_per_task=(
                    int(max_seed_eps) if max_seed_eps is not None else None
                ),
            )
            replay_load_s = time.perf_counter() - replay_load_start
            sampleable_windows = int(warmup_replay.sampleable_window_count())
            self._print_pipeline_event(
                "[pipeline][replay] loaded complete "
                f"episodes={n} transitions={warmup_replay.num_transitions} "
                f"sampleable_windows={sampleable_windows} "
                f"elapsed_s={replay_load_s:.1f}"
            )
            if self.distributed.is_main_process:
                cap_msg = (
                    "all"
                    if max_seed_eps is None
                    else f"<= {int(max_seed_eps)}/task"
                )
                print(
                    f"[pipeline] seeded {n} offline episodes ({cap_msg}), "
                    f"{warmup_replay.num_transitions} transitions, "
                    f"capacity_mode={replay_capacity_mode}",
                    flush=True,
                )
            if n == 0 or warmup_replay.num_transitions == 0:
                raise RuntimeError("offline seeding produced an empty replay buffer")
            classifier_cfg = getattr(_unwrap(self.classifier), "cfg", None)
            default_cls_window = int(
                getattr(
                    self,
                    "_cls_window",
                    OmegaConf.select(cfg, "classifier.window", default=4) or 4,
                )
            )
            cls_window = int(getattr(classifier_cfg, "window", default_cls_window))
            cls_chunk_size = int(getattr(classifier_cfg, "chunk_size", 1))
            classifier_windows = int(
                warmup_replay.classifier_window_count(
                    window=cls_window,
                    chunk_size=cls_chunk_size,
                )
            )
            wm_steps, cls_steps = self._resolve_warmup_steps(
                warmup_replay,
                wm_steps=wm_steps,
                cls_steps=cls_steps,
                replay_epochs=warmup_replay_epochs,
                replay_max_steps=warmup_replay_max_steps,
                wm_batch_size=bs,
                cls_batch_size=cls_bs,
                cls_window=cls_window,
                cls_chunk_size=cls_chunk_size,
            )
            self._print_pipeline_event(
                "[pipeline][warmup] resolved replay warmup "
                f"epochs={warmup_replay_epochs} wm_updates={wm_steps} "
                f"cls_updates={cls_steps} wm_batch={bs} cls_batch={cls_bs} "
                f"classifier_window={cls_window} chunk_size={cls_chunk_size} "
                f"classifier_windows={classifier_windows}"
            )
            # The frozen encoder is idle during warmup — park it off-GPU to reclaim
            # its weights (restored before the online phase below).
            self._set_encoder_device("cpu")
            self._print_pipeline_event(
                "[pipeline][device] encoder parked on cpu for replay warmup"
            )

        if need_wm and need_cls:
            self.console_banner(
                "[1/3] REPLAY WARMUP",
                subtitle=f"wm={wm_steps} cls={cls_steps} learner updates",
            )
            wm_last = self._offline_warmup_wm(
                warmup_replay,
                steps=wm_steps,
                batch_size=bs,
                optim_cfg=optim_cfg,
                checkpoint_every=warmup_checkpoint_every,
                checkpoint_fn=(
                    self._save_wm_warmup if self.distributed.is_main_process else None
                ),
            )
            if self.distributed.is_main_process:
                self._save_wm_warmup()
            cls_last = self._offline_warmup_classifier(
                warmup_replay,
                steps=cls_steps,
                batch_size=cls_bs,
                early_neg_stride=early_neg_stride,
                grad_clip=grad_clip,
                log_step_offset=wm_steps,
                checkpoint_every=warmup_checkpoint_every,
                checkpoint_fn=(
                    self._save_cls_warmup if self.distributed.is_main_process else None
                ),
            )
            if self.distributed.is_main_process:
                self._save_cls_warmup()
                self.console_banner(
                    "[1/3] REPLAY WARMUP",
                    subtitle=f"wm_loss {wm_last:.3f} cls_acc {cls_last:.3f}",
                    done=True,
                )
        elif need_wm:
            self.console_banner("[1/3] WM WARMUP", subtitle=f"{wm_steps} steps")
            wm_last = self._offline_warmup_wm(
                warmup_replay,
                steps=wm_steps,
                batch_size=bs,
                optim_cfg=optim_cfg,
                checkpoint_every=warmup_checkpoint_every,
                checkpoint_fn=(
                    self._save_wm_warmup if self.distributed.is_main_process else None
                ),
            )
            if self.distributed.is_main_process:
                self._save_wm_warmup()
                self.console_banner("[1/3] WM WARMUP", subtitle=f"wm_loss {wm_last:.3f}", done=True)
        if not need_wm:
            if os.path.exists(self._wm_warmup_ckpt()):
                payload = torch.load(self._wm_warmup_ckpt(), map_location="cpu", weights_only=False)
                _unwrap(self.world_model).load_state_dict(payload["world_model"])
            elif os.path.isdir(self._wm_warmup_hf_dir()):
                src = load_module_pretrained(self._wm_warmup_hf_dir())
                _unwrap(self.world_model).load_state_dict(src.state_dict())

        if need_cls and not need_wm:
            self.console_banner("[2/3] CLASSIFIER WARMUP", subtitle=f"{cls_steps} steps")
            cls_last = self._offline_warmup_classifier(
                warmup_replay,
                steps=cls_steps,
                batch_size=cls_bs,
                early_neg_stride=early_neg_stride,
                grad_clip=grad_clip,
                log_step_offset=wm_steps,
                checkpoint_every=warmup_checkpoint_every,
                checkpoint_fn=(
                    self._save_cls_warmup if self.distributed.is_main_process else None
                ),
            )
            if self.distributed.is_main_process:
                self._save_cls_warmup()
                self.console_banner("[2/3] CLASSIFIER WARMUP", subtitle=f"acc {cls_last:.3f}", done=True)
        if not need_cls:
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
        # Restore the encoder onto the device for the online phase that uses it.
        self._set_encoder_device(self.device)
        self._print_pipeline_event(
            f"[pipeline][device] encoder restored to {self.device} for online rollout"
        )
        self.console_banner("[3/3] ONLINE COTRAIN", subtitle=f"{total_env_steps} env steps")
        return self._online_cotrain_loop(cfg)
