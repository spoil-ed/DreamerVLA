"""VLA-only training runner with LIBERO rollout evaluation after each epoch."""

from __future__ import annotations

import copy
import gc
import os
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from diffusers.optimization import get_scheduler
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image
from torch.utils.data import DataLoader
from transformers import GenerationConfig

from dreamervla.dataset import BaseDataset
from dreamervla.runners.base_runner import BaseRunner
from dreamervla.runners.distributed import NopretokenizeSFTDistributedHelper
from dreamervla.runners.eval_metrics import summarize_libero_task_success
from dreamervla.runners.render_device_config import (
    cuda_visible_devices_from_env,
    parse_device_ids,
)
from dreamervla.utils.checkpoint_util import TopKCheckpointManager
from dreamervla.utils.egl_device import apply_libero_render_regime
from dreamervla.utils.ema import EMAHelper
from dreamervla.utils.hf_checkpoint import resolve_hf_checkpoint_dir
from dreamervla.utils.optim import build_optimizer
from dreamervla.utils.paths import checkpoints_path, data_path
from dreamervla.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _select_eval_render_backend(root_cfg: DictConfig, eval_cfg: Any) -> str:
    backend = str(
        OmegaConf.select(
            eval_cfg,
            "render_backend",
            default=OmegaConf.select(root_cfg, "render_backend", default="osmesa"),
        )
    ).strip().lower()
    if backend not in {"egl", "osmesa"}:
        raise ValueError(f"eval.render_backend must be 'egl' or 'osmesa', got {backend!r}")
    return backend


def _eval_render_gpu_pool(root_cfg: DictConfig, eval_cfg: Any, backend: str) -> list[int]:
    if str(backend).strip().lower() != "egl":
        return []
    for key in ("render_gpu_pool", "gpu_pool", "render_devices", "egl_device_pool"):
        devices = parse_device_ids(OmegaConf.select(eval_cfg, key, default=None))
        if devices:
            return devices
    # Auto default: keep mujoco EGL render on a physical GPU DISJOINT from the
    # torch compute device (cuda:0 = first visible). A heavy torch policy and
    # mujoco EGL on ONE physical GPU abort mjr_readPixels after a few hundred
    # renders; both GPUs stay in CUDA_VISIBLE_DEVICES so MUJOCO_EGL_DEVICE_ID
    # remains a valid global EGL id (robosuite asserts it is visible). With a
    # single visible GPU there is no disjoint choice; fall back to it (short
    # evals hold, long ones need >=2 GPUs or render_backend=osmesa).
    visible = cuda_visible_devices_from_env()
    if len(visible) >= 2:
        return visible[1:]
    if visible:
        return visible
    devices = parse_device_ids(OmegaConf.select(root_cfg, "eval.gpus", default=None))
    if devices:
        return devices[1:] if len(devices) >= 2 else devices
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        return list(range(1, count)) if count >= 2 else list(range(count))
    return []


def _eval_render_regime_params(
    root_cfg: DictConfig, eval_cfg: Any
) -> tuple[str, int, list[int]]:
    """Resolve (backend, shard_id, gpu_pool) for the LIBERO render regime.

    Used to hand the render regime to a subprocess env (EvalSubprocEnv) so the
    child applies it via the same shared ``apply_libero_render_regime`` helper,
    instead of setting an EGL context in the eval process next to torch.
    """
    backend = _select_eval_render_backend(root_cfg, eval_cfg)
    shard_id = int(OmegaConf.select(eval_cfg, "render_shard_id", default=0))
    gpu_pool = _eval_render_gpu_pool(root_cfg, eval_cfg, backend)
    if backend == "egl":
        compute = cuda_visible_devices_from_env()
        # mujoco EGL and a heavy torch policy on the SAME physical GPU abort
        # mjr_readPixels under load (NVIDIA EGL driver contention — NOT memory;
        # verified with 63 GB free at crash). The render GPU must be disjoint
        # from the torch compute GPU (cuda:0 == compute[0]). If it cannot be
        # (single visible GPU, or an explicit pool overlapping compute[0]),
        # fall back to osmesa — the only reliable same-GPU path.
        if compute and gpu_pool and compute[0] in gpu_pool:
            print(
                "  [Eval] render_backend=egl but the render GPU overlaps the torch "
                "compute GPU (cuda:0); mujoco EGL + a heavy policy on one physical GPU "
                "abort mjr_readPixels. Falling back to render_backend=osmesa. Give >=2 "
                "GPUs (CUDA_VISIBLE_DEVICES) or an explicit disjoint eval.render_gpu_pool "
                "for real EGL eval.",
                flush=True,
            )
            return "osmesa", shard_id, []
    return backend, shard_id, gpu_pool


def _apply_libero_eval_render_regime(root_cfg: DictConfig, eval_cfg: Any) -> None:
    """Apply the LIBERO render backend before eval imports robosuite/LIBERO envs."""
    backend, shard_id, gpu_pool = _eval_render_regime_params(root_cfg, eval_cfg)
    apply_libero_render_regime(backend, shard_id, gpu_pool)


def build_libero_env_cfg(
    eval_cfg: Any,
    *,
    task_ids: list[int],
    num_episodes: int,
    max_steps: int,
    seed: int,
    resolution: int,
) -> DictConfig:
    """Assemble the LiberoEnv DictConfig from the Hydra eval block."""
    libero_env_cfg = OmegaConf.select(eval_cfg, "libero_env", default=None)
    if libero_env_cfg is None:
        raise ValueError(
            "eval.libero_env config block is required for eval.scheme=rlinf_chunk"
        )
    return OmegaConf.create(
        {
            "task_suite_name": str(
                OmegaConf.select(eval_cfg, "task_suite_name", default="libero_goal")
            ),
            "seed": int(seed),
            "group_size": int(libero_env_cfg.group_size),
            "is_eval": True,
            "use_fixed_reset_state_ids": True,
            "use_ordered_reset_state_ids": True,
            "auto_reset": bool(libero_env_cfg.auto_reset),
            "ignore_terminations": bool(libero_env_cfg.ignore_terminations),
            "max_episode_steps": int(max_steps),
            "reset_wait_steps": int(libero_env_cfg.reset_wait_steps),
            "reset_gripper_open": bool(libero_env_cfg.reset_gripper_open),
            "use_rel_reward": False,
            "use_step_penalty": False,
            "reward_coef": 1.0,
            "task_id_filter": [int(t) for t in task_ids],
            "max_trials_per_task": int(num_episodes),
            "specific_reset_id": None,
            "init_params": {
                "camera_heights": int(resolution),
                "camera_widths": int(resolution),
            },
        }
    )


class _EvalInferResult:
    """Minimal ``infer_fn`` output the rollout core decodes via ``.action_chunk``."""

    __slots__ = ("action_chunk",)

    def __init__(self, action_chunk: Any) -> None:
        self.action_chunk = action_chunk


class _EvalFrameHistoryExtractor:
    """Per-slot frame-history preparer for the parallel LIBERO eval path.

    Faithfully re-implements the sequential eval's ``frame_history``/``padded``
    construction (``evaluate_libero``): append this slot's ``(third_pil,
    wrist_pil)`` each step, truncate to ``history_length``, then pad with the
    oldest available frame until history fills. Returns the exact inputs
    ``_generate_actions`` consumes so a parallel slot is byte-identical to the
    sequential path for the same ``(task, init_state)``.
    """

    def __init__(self, history_length: int) -> None:
        self._history_length = max(1, int(history_length))
        self._frame_history: list[tuple[Image.Image, Image.Image]] = []
        # Optional per-slot OFT base-eval state (None for VLA). The OFT base
        # override's ``_generate_actions`` reads a single shared extractor plus an
        # ``env_step`` from ``self._libero_current_eval_context``; parallel slots
        # step in lockstep, so each slot must carry its OWN extractor and step
        # counter to avoid cross-slot frame-history contamination.
        self._oft_extractor: Any = None
        self._oft_env_step = 0

    def attach_oft_extractor(self, oft_extractor: Any) -> None:
        """Give this slot its own OFT extractor (isolated per-slot history)."""
        self._oft_extractor = oft_extractor

    def reset(self) -> None:
        self._frame_history = []
        # Restart this slot's OFT episode: step counter to 0 (so the OFT override
        # resets the extractor on its first ``env_step==0`` call) and clear the
        # per-slot frame deque in lockstep with the new episode.
        self._oft_env_step = 0
        if self._oft_extractor is not None and hasattr(self._oft_extractor, "reset"):
            self._oft_extractor.reset()

    def prepare(self, record: dict, task_description: str) -> dict:
        third_pil = Image.fromarray(record["third_image"])
        wrist_pil = Image.fromarray(record["wrist_image"])
        self._frame_history.append((third_pil, wrist_pil))
        if len(self._frame_history) > self._history_length:
            self._frame_history = self._frame_history[-self._history_length :]
        padded = [self._frame_history[0]] * (
            self._history_length - len(self._frame_history)
        ) + self._frame_history
        env_step = self._oft_env_step
        self._oft_env_step += 1
        return {
            "padded": padded,
            "state": record["state"],
            "task_description": task_description,
            # Raw LIBERO obs for the OFT base override's _generate_actions,
            # which reads self._libero_current_raw_obs (ignored by VLA).
            "raw_obs": record.get("raw_obs"),
            # Per-slot OFT base-eval state (None/0 for VLA); the parallel
            # _infer_fn threads these onto self before this slot's generate call.
            "oft_extractor": self._oft_extractor,
            "env_step": env_step,
        }


class PretokenizeVLARunner(BaseRunner):
    runner_name = "vla_sft"
    runner_status = "current"
    runner_family = "vla"
    include_keys = ("global_step", "epoch")
    exclude_keys = tuple()
    checkpoint_restore_output_dir = True

    @property
    def default_vla_init_dir(self) -> str:
        return str(checkpoints_path("VLA_model_256", "libero_goal"))

    @property
    def default_output_dir(self) -> str:
        return str(data_path("outputs", "vla", "debug_pretokenize_vla"))

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        if output_dir is None:
            output_dir = str(
                OmegaConf.select(
                    config, "training.out_dir", default=self.default_output_dir
                )
            )
        super().__init__(config, output_dir=output_dir)

        self.distributed = NopretokenizeSFTDistributedHelper.initialize(
            strategy=str(
                OmegaConf.select(config, "training.distributed_strategy", default="ddp")
            ),
            fsdp_mixed_precision=str(
                OmegaConf.select(
                    config, "training.fsdp_mixed_precision", default="bf16"
                )
            ),
            enable_activation_checkpointing=bool(
                OmegaConf.select(
                    config, "training.enable_activation_checkpointing", default=True
                )
            ),
        )
        self.rank = self.distributed.rank
        self.local_rank = self.distributed.local_rank
        self.world_size = self.distributed.world_size
        self.device = self.distributed.resolve_device(str(self.config.trainer.device))
        if self.distributed.is_main_process:
            self.print_config()
        set_seed(int(self.config.seed) + self.rank)
        self.encoder = None
        self.vla_optimizer = None
        self.vla_ema: EMAHelper | None = None

    def _build_trainable_encoder_cfg(self, cfg: DictConfig) -> DictConfig:
        encoder_cfg = self.build_encoder_cfg(cfg)
        with open_dict(encoder_cfg):
            encoder_cfg.model_path = self._resolve_vla_init_path()
            train_encoder_backbone = bool(
                OmegaConf.select(cfg, "training.train_encoder_backbone", default=True)
            )
            encoder_cfg.freeze_backbone = not train_encoder_backbone
        return encoder_cfg

    @staticmethod
    def _set_trainable_encoder_parameters(encoder: Any, patterns: list[str]) -> int:
        if not hasattr(encoder, "named_parameters"):
            return 0
        for _, parameter in encoder.named_parameters():
            parameter.requires_grad = False
        matched = 0
        for name, parameter in encoder.named_parameters():
            if any(pattern in name for pattern in patterns):
                parameter.requires_grad = True
                matched += 1
        return matched

    def _state_dict_for_checkpoint(self, key: str, value: Any) -> dict[str, Any] | None:
        if key == "encoder" and self.encoder is not None:
            with self.distributed.model_state_dict_context(self.encoder.backbone):
                return self.encoder.state_dict()
        if (
            key == "vla_optimizer"
            and self.vla_optimizer is not None
            and self.encoder is not None
        ):
            return self.distributed.optimizer_state_dict(
                self.encoder.backbone, self.vla_optimizer
            )
        return value.state_dict()

    def _save_checkpoint_sidecars(self, path: Path, payload: dict[str, Any]) -> None:
        if self.encoder is None or not bool(
            OmegaConf.select(self.cfg, "checkpoint.save_hf_encoder", default=True)
        ):
            return
        hf_dir = self._hf_dir_for_runner_ckpt(path)
        hf_dir.mkdir(parents=True, exist_ok=True)
        backbone = self.distributed.unwrap_module(self.encoder.backbone)
        if not hasattr(backbone, "save_pretrained"):
            return
        hf_state_dict = self._extract_backbone_state_for_hf(payload)
        try:
            backbone.save_pretrained(
                str(hf_dir),
                state_dict=hf_state_dict,
                safe_serialization=bool(
                    OmegaConf.select(
                        self.cfg, "checkpoint.hf_safe_serialization", default=True
                    )
                ),
            )
        except TypeError:
            try:
                backbone.save_pretrained(str(hf_dir), state_dict=hf_state_dict)
            except TypeError:
                backbone.save_pretrained(str(hf_dir))
        state_path = hf_dir / "dreamervla_runner_state.pt"
        torch.save(
            {
                "global_step": int(self.global_step),
                "epoch": int(self.epoch),
                "source_ckpt": str(path.resolve()),
            },
            state_path,
        )
        if self.distributed.is_main_process:
            print(f"  [Checkpoint] wrote HF VLA checkpoint: {hf_dir}")

    @staticmethod
    def _hf_dir_for_runner_ckpt(path: Path) -> Path:
        if path.suffix:
            return path.with_name(f"{path.stem}_hf")
        return path

    @staticmethod
    def _extract_backbone_state_for_hf(
        payload: dict[str, Any]
    ) -> dict[str, torch.Tensor] | None:
        encoder_state = payload.get("state_dicts", {}).get("encoder")
        if not isinstance(encoder_state, dict):
            return None
        backbone_state: dict[str, torch.Tensor] = {}
        for key, value in encoder_state.items():
            if not isinstance(key, str) or not isinstance(value, torch.Tensor):
                continue
            if key.startswith("backbone.module."):
                backbone_state[key[len("backbone.module.") :]] = value
            elif key.startswith("backbone."):
                backbone_state[key[len("backbone.") :]] = value
        return backbone_state or None

    def load_hf_checkpoint(self, path: str | Path, **_: Any) -> dict[str, Any]:
        if self.encoder is None:
            raise RuntimeError("Cannot load a VLA HF checkpoint before encoder setup.")
        model_dir = resolve_hf_checkpoint_dir(path)
        if self.distributed.is_main_process:
            print(f"Loading VLA HF checkpoint weights from {model_dir}")
        current_model_path = getattr(self.encoder, "model_path", None)
        try:
            self.encoder.model_path = str(model_dir)
            reloaded = type(self.encoder)(
                model_path=str(model_dir),
                tokenizer_path=self.encoder.tokenizer_path,
                text_tokenizer_path=self.encoder.text_tokenizer_path,
                chameleon_vqgan_config=self.encoder.chameleon_vqgan_config,
                chameleon_vqgan_ckpt=self.encoder.chameleon_vqgan_ckpt,
                resolution=self.encoder.resolution,
                action_dim=self.encoder.action_dim,
                time_horizon=self.encoder.time_horizon,
                action_head_type=self.encoder.action_head_type,
                pool=self.encoder.pool,
                freeze_backbone=False,
            ).to(self.device)
            self._load_state_dict_from_checkpoint(
                "encoder", self.encoder, reloaded.state_dict(), strict=True
            )
            del reloaded
        finally:
            if current_model_path is not None:
                self.encoder.model_path = current_model_path
        state_path = model_dir / "dreamervla_runner_state.pt"
        payload: dict[str, Any] = {"hf_checkpoint_dir": str(model_dir)}
        if state_path.is_file():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict):
                self.global_step = int(state.get("global_step", self.global_step))
                self.epoch = int(state.get("epoch", self.epoch))
                payload.update(state)
        return payload

    def _load_state_dict_from_checkpoint(
        self,
        key: str,
        value: Any,
        state_dict: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        if key == "encoder" and self.encoder is not None:
            with self.distributed.model_state_dict_context(self.encoder.backbone):
                value.load_state_dict(state_dict, **kwargs)
            return
        if (
            key == "vla_optimizer"
            and self.vla_optimizer is not None
            and self.encoder is not None
        ):
            self.distributed.load_optimizer_state_dict(
                self.encoder.backbone, self.vla_optimizer, state_dict
            )
            return
        value.load_state_dict(state_dict, **kwargs)

    # ---- Validation loss evaluation (FSDP-compatible) ----

    @torch.no_grad()
    def evaluate_val_loss(
        self, val_dataloader: DataLoader, split_name: str
    ) -> dict[str, float]:
        """Compute VLA loss on a validation set. Runs on all ranks (FSDP-safe)."""
        self.encoder.eval()
        val_losses: list[float] = []
        val_token_losses: list[float] = []
        val_action_losses: list[float] = []

        for batch in val_dataloader:
            has_tokenized = isinstance(batch.get("input_ids"), list) and isinstance(
                batch.get("labels"), list
            )
            if not has_tokenized:
                continue
            vla_loss_dict = self.encoder.compute_action_sft_loss_from_tokenized(
                input_ids_list=batch["input_ids"],
                labels_list=batch["labels"],
            )
            val_losses.append(float(vla_loss_dict["loss"].item()))
            val_token_losses.append(float(vla_loss_dict["token_loss"].item()))
            val_action_losses.append(float(vla_loss_dict["action_loss"].item()))

        self.encoder.train()

        if not val_losses:
            return {}

        count = max(self.distributed.reduce_sum(len(val_losses)), 1.0)
        metrics = {
            f"val_{split_name}_loss": self.distributed.reduce_sum(sum(val_losses))
            / count,
            f"val_{split_name}_token_loss": self.distributed.reduce_sum(
                sum(val_token_losses)
            )
            / count,
            f"val_{split_name}_action_loss": self.distributed.reduce_sum(
                sum(val_action_losses)
            )
            / count,
        }
        if self.distributed.is_main_process:
            print(
                f"  [Val {split_name}] loss={metrics[f'val_{split_name}_loss']:.4f} "
                f"token={metrics[f'val_{split_name}_token_loss']:.4f} "
                f"action={metrics[f'val_{split_name}_action_loss']:.4f}"
            )
        return metrics

    # ---- LIBERO rollout evaluation (single-GPU only) ----

    @torch.no_grad()
    def evaluate_libero(self, epoch: int) -> dict[str, float]:
        """Run LIBERO rollout evaluation (single-process only).

        LIBERO rollout is too slow for inline distributed evaluation.
        This method only works when running in single-GPU (non-FSDP) mode.
        For multi-GPU FSDP training, use the standalone eval script instead.
        """
        if not self.distributed.is_main_process:
            return {}

        eval_cfg = OmegaConf.select(self.cfg, "eval", default=None)
        if eval_cfg is None:
            return {}

        # Skip eval under FSDP — model is sharded, can't do single-rank inference
        if self.distributed.uses_fsdp:
            if epoch == -1:
                print(
                    "  [Eval] Skipping baseline eval under FSDP. Use scripts/eval_libero_vla.sh on saved checkpoints."
                )
            return {}

        _apply_libero_eval_render_regime(self.cfg, eval_cfg)

        from libero.libero import benchmark as libero_benchmark

        from dreamervla.envs import (
            TASK_MAX_STEPS,
            get_libero_dummy_action,
            get_libero_env,
            resolve_libero_eval_protocol,
            save_rollout_video,
            select_libero_action_chunk,
        )
        from dreamervla.envs.libero.utils import build_libero_eval_record

        protocol = resolve_libero_eval_protocol(self.cfg, eval_cfg)
        seed = int(protocol["seed"])
        num_steps_wait = int(protocol["num_steps_wait"])
        np.random.seed(seed)
        task_suite_name = str(
            OmegaConf.select(eval_cfg, "task_suite_name", default="libero_goal")
        )
        num_episodes = int(
            OmegaConf.select(eval_cfg, "num_episodes_per_task", default=3)
        )
        action_steps = int(OmegaConf.select(eval_cfg, "action_steps", default=10))
        resolution = int(OmegaConf.select(self.cfg, "encoder.resolution", default=256))
        # History length must match training (`his` in processed_data_generate_convs.sh).
        # Training builds img_c = [prev_third, prev_wrist, cur_third, cur_wrist] (his=2).
        history_length = int(OmegaConf.select(eval_cfg, "history_length", default=2))
        save_video = bool(OmegaConf.select(eval_cfg, "save_video", default=False))
        video_max_episodes = int(
            OmegaConf.select(eval_cfg, "video_max_episodes", default=1)
        )
        video_dir = os.path.join(self.output_dir, "videos")

        item_processor = self.encoder._build_processor(self.device)

        print(
            f"  [Eval] loading LIBERO benchmark suite '{task_suite_name}' ...",
            flush=True,
        )
        benchmark_dict = libero_benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        total_tasks = int(task_suite.n_tasks)
        task_ids_cfg = OmegaConf.select(eval_cfg, "task_ids", default=None)
        if task_ids_cfg is not None:
            task_ids = [int(task_id) for task_id in task_ids_cfg]
        else:
            task_start = int(OmegaConf.select(eval_cfg, "task_start", default=0))
            max_tasks = OmegaConf.select(eval_cfg, "max_tasks", default=None)
            task_stop = (
                total_tasks
                if max_tasks is None
                else min(total_tasks, task_start + int(max_tasks))
            )
            task_ids = list(range(task_start, task_stop))
        if not task_ids:
            raise ValueError(
                "LIBERO eval selected no tasks; check eval.task_ids/task_start/max_tasks."
            )
        max_steps_cfg = OmegaConf.select(eval_cfg, "max_steps", default=None)
        max_steps = int(
            max_steps_cfg
            if max_steps_cfg is not None
            else TASK_MAX_STEPS.get(task_suite_name, 300)
        )
        print(
            f"  [Eval] suite='{task_suite_name}' tasks={task_ids} "
            f"episodes_per_task={num_episodes} max_steps={max_steps} "
            f"action_steps={action_steps} history_length={history_length} "
            f"seed={seed} num_steps_wait={num_steps_wait}",
            flush=True,
        )

        self.encoder.eval()
        backbone = self.distributed.unwrap_module(self.encoder.backbone)

        num_envs = int(OmegaConf.select(eval_cfg, "num_envs", default=1))
        scheme = str(
            OmegaConf.select(eval_cfg, "scheme", default="rlinf_chunk")
        ).strip().lower()
        if scheme != "rlinf_chunk":
            raise ValueError(
                f"eval.scheme must be 'rlinf_chunk', got {scheme!r}"
            )
        return self._evaluate_libero_rlinf_chunk(
            epoch=epoch,
            eval_cfg=eval_cfg,
            backbone=backbone,
            item_processor=item_processor,
            task_ids=task_ids,
            num_episodes=num_episodes,
            max_steps=max_steps,
            action_steps=action_steps,
            history_length=history_length,
            resolution=resolution,
            seed=seed,
            num_envs=num_envs,
        )

        total_episodes, total_successes = 0, 0
        task_records: list[dict[str, int]] = []
        run_t0 = time.time()

        for task_index, task_id in enumerate(task_ids):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(
                task, resolution=resolution, seed=seed
            )
            n_eps = num_episodes
            print(
                f'  [Eval] >>> Task {task_id} ({task_index + 1}/{len(task_ids)}) start: "{task_description}" '
                f"(episodes={n_eps})",
                flush=True,
            )

            task_successes = 0
            task_t0 = time.time()
            for episode_idx in range(n_eps):
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])

                done = False
                ep_t0 = time.time()
                actions_buffer: list[np.ndarray] = []
                should_record = save_video and total_episodes < video_max_episodes
                rollout_images: list[np.ndarray] = []
                # Frame history buffer: list of (third_view_pil, wrist_pil) oldest→newest.
                # Matches training's `img_history_start_idx = max(0, j - his + 1)` which
                # repeats the first frame until history fills up.
                frame_history: list[tuple[Image.Image, Image.Image]] = []

                steps_taken = 0
                for t in range(max_steps + num_steps_wait):
                    if t < num_steps_wait:
                        obs, _, done, _ = env.step(get_libero_dummy_action())
                        continue

                    # Shared per-step record builder (identical to the parallel
                    # path) to prevent field-by-field drift: build_libero_eval_record
                    # returns get_libero_image(third), get_libero_image(wrist) and the
                    # eef_pos/axisangle/gripper_qpos state concat — byte-identical to
                    # the old inline block.
                    record = build_libero_eval_record(obs, resolution)
                    img = record["third_image"]
                    if should_record:
                        rollout_images.append(img)
                    wrist_img = record["wrist_image"]
                    state = record["state"]

                    third_pil = Image.fromarray(img)
                    wrist_pil = Image.fromarray(wrist_img)
                    frame_history.append((third_pil, wrist_pil))
                    if len(frame_history) > history_length:
                        frame_history = frame_history[-history_length:]

                    if len(actions_buffer) == 0:
                        # Pad with the oldest available frame when history is shorter
                        # than `history_length` (first action_steps steps of episode).
                        padded = [frame_history[0]] * (
                            history_length - len(frame_history)
                        ) + frame_history
                        # Eval subclasses can use the raw simulator observation for
                        # non-VLA inputs (for example pixel DreamerV3 rollout) while
                        # keeping the VLA PIL history path unchanged.
                        self._libero_current_raw_obs = obs
                        self._libero_current_eval_context = {
                            "task_id": int(task_id),
                            "task_index": int(task_index),
                            "episode_idx": int(episode_idx),
                            "env_step": int(steps_taken),
                            "rollout_t": int(t),
                            "task_description": str(task_description),
                        }
                        predicted = self._generate_actions(
                            backbone,
                            item_processor,
                            padded,
                            state,
                            task_description,
                            action_steps,
                        )
                        actions_buffer = select_libero_action_chunk(
                            predicted, action_steps
                        )

                    if len(actions_buffer) == 0:
                        break
                    action = actions_buffer.pop(0)
                    if bool(
                        OmegaConf.select(
                            self.cfg, "eval.empty_cuda_cache_each_step", default=False
                        )
                    ):
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    obs, _, done, _ = env.step(action.tolist())
                    steps_taken = t - num_steps_wait + 1

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                video_path = None
                if should_record and rollout_images:
                    video_path = save_rollout_video(
                        video_dir,
                        rollout_images,
                        total_episodes,
                        bool(done),
                        task_description,
                    )
                total_episodes += 1
                self.console_record_success(bool(done))
                self.console_progress(
                    total_episodes, len(task_ids) * num_episodes, "eval"
                )
                ep_dt = time.time() - ep_t0
                tag = "OK " if done else "FAIL"
                # Reclaim any per-episode GPU/CPU memory leaked by the
                # mujoco offscreen renderer / torch generation cache.
                # Without this the EGL context grows ~500MB per episode and
                # eventually triggers a silent SIGABRT inside the driver.
                import gc as _gc

                _gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gpu_mb = (
                    (torch.cuda.memory_allocated() // (1024 * 1024))
                    if torch.cuda.is_available()
                    else 0
                )
                print(
                    f"  [Eval]   ep {episode_idx + 1}/{n_eps} {tag} "
                    f"steps={steps_taken} time={ep_dt:5.1f}s "
                    f"task_succ={task_successes}/{episode_idx + 1} "
                    f"total_succ={total_successes}/{total_episodes} "
                    f"({total_successes / max(total_episodes, 1):.1%})  "
                    f"gpu_alloc={gpu_mb}MB"
                    f"{' video=' + video_path if video_path else ''}",
                    flush=True,
                )
            env.close()

            rate = task_successes / max(n_eps, 1)
            task_records.append(
                {
                    "task_id": int(task_id),
                    "episodes": int(n_eps),
                    "successes": int(task_successes),
                }
            )
            task_dt = time.time() - task_t0
            print(
                f'  [Eval] <<< Task {task_id} ({task_index + 1}/{len(task_ids)}) done: "{task_description}" '
                f"success={rate:.1%} ({task_successes}/{n_eps}) "
                f"time={task_dt:.1f}s   running_total={total_successes}/{total_episodes} "
                f"({total_successes / max(total_episodes, 1):.1%})",
                flush=True,
            )

        metrics = summarize_libero_task_success(
            task_records,
            episodes_per_task=num_episodes,
        )
        avg_success = float(metrics["eval_success_rate"])
        run_dt = time.time() - run_t0
        print(
            f"  [Eval] Epoch {epoch} task-mean success rate: {avg_success:.1%} "
            f"({total_successes}/{total_episodes}) total_time={run_dt:.1f}s",
            flush=True,
        )
        return metrics

    def _make_parallel_oft_slot_extractor(self) -> Any:
        """Return a fresh per-slot OFT extractor, or None (VLA / non-OFT).

        Overridden by the OFT base eval runner to hand each parallel slot its own
        OFTRolloutHiddenExtractor instance (isolated frame deque + prompt cache).
        """
        return None

    def _evaluate_libero_rlinf_chunk(
        self,
        *,
        epoch: int,
        eval_cfg: Any,
        backbone: Any,
        item_processor: Any,
        task_ids: list[int],
        num_episodes: int,
        max_steps: int,
        action_steps: int,
        history_length: int,
        resolution: int,
        seed: int,
        num_envs: int,
    ) -> dict[str, float]:
        """RLinf LiberoEnv-port eval: N lockstep subprocess envs, one policy
        call per action chunk (the slots path calls ``_generate_actions`` every
        env step and discards queued results), envs kept alive across episodes
        of the same task (RLinf ``is_eval`` reconfigure-only-on-task-change).
        """
        import math

        from dreamervla.envs import select_libero_action_chunk
        from dreamervla.envs.libero.libero_env import LiberoEnv
        from dreamervla.runners.libero_chunk_eval import run_rlinf_chunk_eval

        # Render regime was already applied in-process by evaluate_libero
        # (_apply_libero_eval_render_regime); spawn children inherit it.
        render_backend, _shard_id, _gpu_pool = _eval_render_regime_params(
            self.cfg, eval_cfg
        )

        total_episodes = len(task_ids) * int(num_episodes)
        n_envs = max(1, int(num_envs))

        extractors = [
            _EvalFrameHistoryExtractor(history_length) for _ in range(n_envs)
        ]
        has_oft = False
        for slot_extractor in extractors:
            oft_slot = self._make_parallel_oft_slot_extractor()
            if oft_slot is not None:
                slot_extractor.attach_oft_extractor(oft_slot)
                has_oft = True
        if int(history_length) > 1 and not has_oft:
            raise ValueError(
                "eval.scheme=rlinf_chunk requires eval.history_length==1 for "
                "non-OFT policies. OFT base eval provides per-slot extractor history; "
                "other policies must use history_length=1 until they implement the "
                "same chunk-cadence extractor contract."
            )

        env_cfg = build_libero_env_cfg(
            eval_cfg,
            task_ids=task_ids,
            num_episodes=num_episodes,
            max_steps=max_steps,
            seed=seed,
            resolution=resolution,
        )
        n_chunk_steps = math.ceil(int(max_steps) / int(action_steps))
        num_epochs = math.ceil(total_episodes / n_envs)

        libero_env_ref: list[Any] = [None]

        def _policy_fn(obs: dict) -> np.ndarray:
            chunks = []
            for i in range(n_envs):
                prep = extractors[i].prepare(
                    {
                        "third_image": obs["main_images"][i],
                        "wrist_image": obs["wrist_images"][i],
                        "state": obs["states"][i],
                        "raw_obs": (
                            libero_env_ref[0].current_raw_obs[i]
                            if libero_env_ref[0] is not None
                            and libero_env_ref[0].current_raw_obs is not None
                            else None
                        ),
                    },
                    obs["task_descriptions"][i],
                )
                self._libero_current_raw_obs = prep.get("raw_obs")
                oft_extractor = prep.get("oft_extractor")
                if oft_extractor is not None:
                    self._base_oft_extractor = oft_extractor
                    self._libero_current_eval_context = {
                        "env_step": int(prep.get("env_step", 0))
                    }
                predicted = self._generate_actions(
                    backbone,
                    item_processor,
                    prep["padded"],
                    prep["state"],
                    prep["task_description"],
                    action_steps,
                )
                chunks.append(
                    np.stack(
                        [
                            np.asarray(a, dtype=np.float64)
                            for a in select_libero_action_chunk(
                                predicted, action_steps
                            )
                        ]
                    )
                )
            return np.stack(chunks)

        def _on_epoch_start() -> None:
            for slot_extractor in extractors:
                slot_extractor.reset()

        run_t0 = time.time()
        print(
            f"  [Eval] rlinf_chunk rollout: num_envs={n_envs} "
            f"episodes={total_episodes} chunk_steps={n_chunk_steps} "
            f"epochs={num_epochs} render_backend={render_backend}",
            flush=True,
        )
        with LiberoEnv(env_cfg, num_envs=n_envs) as libero_env:
            libero_env_ref[0] = libero_env
            tally = run_rlinf_chunk_eval(
                libero_env,
                _policy_fn,
                n_chunk_steps=n_chunk_steps,
                num_epochs=num_epochs,
                total_episodes=total_episodes,
                on_epoch_start=_on_epoch_start,
                on_reset=getattr(self, "_on_libero_eval_reset", None),
                on_chunk=getattr(self, "_on_libero_eval_chunk", None),
            )
        metrics = tally.summarize(episodes_per_task=num_episodes)
        avg_success = float(metrics["eval_success_rate"])
        run_dt = time.time() - run_t0
        metrics["eval/env_chunk_steps"] = float(tally.env_chunk_steps)
        metrics["eval/env_action_steps"] = float(tally.env_action_steps)
        metrics["eval/elapsed_seconds"] = float(run_dt)
        metrics["eval/env_chunk_per_s"] = (
            float(tally.env_chunk_steps) / run_dt if run_dt > 0 else 0.0
        )
        metrics["eval/env_action_step_per_s"] = (
            float(tally.env_action_steps) / run_dt if run_dt > 0 else 0.0
        )
        finalize_observer = getattr(
            self,
            "_finalize_libero_eval_observer",
            None,
        )
        if callable(finalize_observer):
            observer_metrics = finalize_observer()
            if not isinstance(observer_metrics, dict):
                raise TypeError(
                    "_finalize_libero_eval_observer() must return a mapping"
                )
            metrics.update(observer_metrics)
        print(
            f"  [Eval] Epoch {epoch} task-mean success rate: {avg_success:.1%} "
            f"(rlinf_chunk num_envs={n_envs}) total_time={run_dt:.1f}s "
            f"env_chunk_per_s={metrics['eval/env_chunk_per_s']:.2f}",
            flush=True,
        )
        return metrics

    def _generate_actions(
        self,
        backbone,
        item_processor,
        frame_history: list[tuple[Image.Image, Image.Image]],
        state: np.ndarray,
        task_description: str,
        action_steps: int,
    ) -> list[np.ndarray]:
        """Tokenize observation, run model generate, decode action tokens.

        `frame_history` is oldest→newest, each element a (third_view_pil, wrist_pil)
        tuple. Flattened to the same [prev_third, prev_wrist, cur_third, cur_wrist]
        ordering used at training time (see dreamervla/preprocess/action_state_model_conv_generation.py:
        `img_c.extend(image_steps[step_idx])` with `img_names=[imgs_third_view, imgs_wrist]`).
        """
        img_c: list[Image.Image] = []
        for third_pil, wrist_pil in frame_history:
            img_c.extend([third_pil, wrist_pil])
        human_val = (
            f"Finish the task: {task_description}."
            + "<|state|>" * 1
            + "<|image|>" * len(img_c)
        )

        conv = {
            "conversations": [{"from": "human", "value": human_val}],
            "image": img_c,
            "action": [],
            "state": [state],
        }
        tokens = item_processor.process_item(conv, training_mode=False)
        if isinstance(tokens, tuple):
            tokens = tokens[0]

        input_ids = torch.tensor(
            tokens, dtype=torch.int64, device=self.device
        ).unsqueeze(0)

        # generate_action_head: generate one token (trigger), then use the
        # action MLP head to predict a full action sequence.  Only needs 1
        # new token so image blocks stay complete.
        generation_config = GenerationConfig(
            max_new_tokens=1,
            max_length=backbone.config.max_position_embeddings,
            temperature=1,
            top_k=None,
            do_sample=False,
            eos_token_id=[8710],
        )

        if hasattr(backbone, "generate_action_head"):
            try:
                predicted = backbone.generate_action_head(input_ids, generation_config)
                actions = self._unnorm_actions(predicted.cpu().float().detach().numpy())
                return [actions[i] for i in range(actions.shape[0])]
            except Exception as e:
                print(f"  [Eval] generate_action_head failed: {e}")

        # Fallback: discrete multi-action generation (needs more tokens)
        generation_config_ma = GenerationConfig(
            max_new_tokens=action_steps * 12,
            max_length=backbone.config.max_position_embeddings,
            temperature=1,
            top_k=None,
            do_sample=False,
            eos_token_id=[8710],
        )
        if hasattr(backbone, "generate_dis_ma"):
            try:
                action_sequences = backbone.generate_dis_ma(
                    input_ids, generation_config_ma
                )
                results = []
                for seq in action_sequences:
                    if isinstance(seq, torch.Tensor):
                        a = seq.cpu().float().detach().numpy()
                    else:
                        a = np.asarray(seq, dtype=np.float32)
                    if a.shape[0] == 7:
                        results.append(self._unnorm_action(a))
                return results
            except Exception as e:
                print(f"  [Eval] generate_dis_ma failed: {e}")

        return []

    @staticmethod
    def _unnorm_action(action: np.ndarray) -> np.ndarray:
        min_values = np.array(
            [-0.9375, -0.9375, -0.9375, -0.24214286, -0.375, -0.36428571, -1.0]
        )
        max_values = np.array([0.9375, 0.9375, 0.9375, 0.34821429, 0.375, 0.375, 1.0])
        if action.shape[0] > 7:
            action = action[:7]
        return (action + 1) / 2 * (max_values - min_values + 1e-8) + min_values

    @staticmethod
    def _unnorm_actions(actions: np.ndarray) -> np.ndarray:
        min_values = np.array(
            [-0.9375, -0.9375, -0.9375, -0.24214286, -0.375, -0.36428571, -1.0]
        )
        max_values = np.array([0.9375, 0.9375, 0.9375, 0.34821429, 0.375, 0.375, 1.0])
        if actions.ndim == 2 and actions.shape[1] > 7:
            actions = actions[:, :7]
        return (actions + 1) / 2 * (max_values - min_values + 1e-8) + min_values

    # ---- main training loop ----

    def run(self) -> list[dict[str, float | str | int]]:
        history: list[dict[str, float | str | int]] = []
        if self.distributed.is_main_process:
            print("VLA Runner begin.")
        cfg = copy.deepcopy(self.cfg)

        dataset: BaseDataset = hydra.utils.instantiate(cfg.dataset)
        assert isinstance(dataset, BaseDataset)
        dataset_action_horizon = getattr(dataset, "action_horizon", None)
        encoder_time_horizon = int(
            OmegaConf.select(cfg, "encoder.time_horizon", default=0) or 0
        )
        if dataset_action_horizon is not None and encoder_time_horizon:
            if int(dataset_action_horizon) != encoder_time_horizon:
                raise ValueError(
                    "VLA dataset action_horizon must match encoder.time_horizon "
                    f"({int(dataset_action_horizon)} != {encoder_time_horizon})."
                )
            if self.distributed.is_main_process:
                print(
                    f"  VLA action horizon: {encoder_time_horizon} "
                    f"(dataset={type(dataset).__name__})"
                )

        train_dataloader = self.make_distributed_dataloader(dataset, cfg.dataloader)

        # ---- Build val dataloaders (optional) ----
        val_dataloaders = self.make_val_dataloaders(cfg)

        encoder_cfg = self._build_trainable_encoder_cfg(cfg)
        self.encoder = hydra.utils.instantiate(encoder_cfg).to(self.device)
        if bool(
            OmegaConf.select(cfg, "training.vla_train_action_head_only", default=False)
        ):
            matched = self._set_trainable_encoder_parameters(
                self.encoder, patterns=["action_head"]
            )
            if matched == 0:
                raise ValueError(
                    "No trainable parameters matched pattern `action_head`."
                )
        self.distributed.wrap_encoder(self.encoder)
        vla_optim_cfg = OmegaConf.select(cfg, "optim.vla")
        if vla_optim_cfg is None:
            raise ValueError("`optim.vla` must be configured.")
        self.vla_optimizer = build_optimizer(self.encoder, vla_optim_cfg)

        # configure ema
        if (
            bool(OmegaConf.select(cfg, "training.use_ema", default=False))
            and self.vla_ema is None
        ):
            self.vla_ema = EMAHelper(
                self.encoder,
                decay=float(OmegaConf.select(cfg, "ema.decay", default=0.9999)),
                update_after_step=int(
                    OmegaConf.select(cfg, "ema.update_after_step", default=0)
                ),
            )

        # resume training
        self.resume(cfg)
        if bool(OmegaConf.select(cfg, "training.resume_advance_epoch", default=False)):
            self.epoch += 1
            self.global_step += 1
            if self.distributed.is_main_process:
                print(
                    f"  [resume] advanced to epoch={self.epoch} global_step={self.global_step}"
                )

        # After optimizer state restore, param_groups may lack `initial_lr`
        # (LambdaLR requires it when last_epoch > -1).
        for pg in self.vla_optimizer.param_groups:
            pg.setdefault("initial_lr", pg["lr"])

        num_epochs_cfg = OmegaConf.select(cfg, "training.num_epochs", default=20)
        num_epochs = 20 if num_epochs_cfg is None else int(num_epochs_cfg)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            str(OmegaConf.select(cfg, "training.lr_scheduler", default="constant")),
            optimizer=self.vla_optimizer,
            num_warmup_steps=int(
                OmegaConf.select(cfg, "training.lr_warmup_steps", default=0)
            ),
            num_training_steps=(len(train_dataloader) * num_epochs)
            // int(cfg.training.gradient_accumulate_every),
            last_epoch=self.global_step - 1,
        )

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        if cfg.training.debug:
            num_epochs = 3
            cfg.training.num_epochs = num_epochs
            cfg.training.max_train_steps = 2
            cfg.training.checkpoint_every = 1

        if self.distributed.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
        self.distributed.barrier()
        train_log_path = os.path.join(self.output_dir, "vla_logs.json.txt")
        train_logger_cm = self.distributed.logger_context(train_log_path)

        if val_dataloaders and self.distributed.is_main_process:
            print(f"  Val dataloaders ready: {list(val_dataloaders.keys())}")

        self.console_banner("TRAINING", subtitle=f"{num_epochs} epochs")
        try:
            with train_logger_cm as train_json_logger:
                reached_max_steps = False
                while self.epoch < num_epochs:
                    self.set_dataloader_epoch(train_dataloader, self.epoch)

                    step_log: dict[str, float | str | int] = {}
                    train_vla_losses: list[float] = []
                    train_vla_token_losses: list[float] = []
                    train_vla_action_losses: list[float] = []

                    self.encoder.train()
                    for batch_idx, batch in enumerate(train_dataloader):
                        has_tokenized = isinstance(
                            batch.get("input_ids"), list
                        ) and isinstance(batch.get("labels"), list)
                        if not has_tokenized:
                            continue

                        vla_loss_dict = (
                            self.encoder.compute_action_sft_loss_from_tokenized(
                                input_ids_list=batch["input_ids"],
                                labels_list=batch["labels"],
                                token_loss_coef=float(
                                    OmegaConf.select(
                                        cfg,
                                        "training.vla_token_loss_coef",
                                        default=1.0,
                                    )
                                ),
                                action_loss_coef=float(
                                    OmegaConf.select(
                                        cfg,
                                        "training.vla_action_loss_coef",
                                        default=1.0,
                                    )
                                ),
                            )
                        )
                        vla_raw_loss = vla_loss_dict["loss"]
                        vla_loss = (
                            vla_raw_loss / cfg.training.gradient_accumulate_every
                        )
                        vla_loss.backward()

                        grad_clip_norm = cfg.optim.get("grad_clip_norm")
                        if grad_clip_norm is not None:
                            self.distributed.clip_grad_norm(
                                self.encoder.backbone, float(grad_clip_norm)
                            )

                        self.vla_optimizer.step()
                        self.vla_optimizer.zero_grad(
                            set_to_none=bool(
                                cfg.optim.get("zero_grad_set_to_none", True)
                            )
                        )
                        lr_scheduler.step()

                        # update ema
                        if self.vla_ema is not None:
                            self.vla_ema.step(self.encoder)

                        train_vla_losses.append(float(vla_raw_loss.item()))
                        train_vla_token_losses.append(
                            float(vla_loss_dict["token_loss"].item())
                        )
                        train_vla_action_losses.append(
                            float(vla_loss_dict["action_loss"].item())
                        )

                        local_step_metrics = {
                            "train_vla_loss": float(vla_raw_loss.item()),
                            "train_vla_token_loss": float(
                                vla_loss_dict["token_loss"].item()
                            ),
                            "train_vla_action_loss": float(
                                vla_loss_dict["action_loss"].item()
                            ),
                            "lr": float(lr_scheduler.get_last_lr()[0]),
                        }
                        reduced = self.distributed.reduce_mean_dict(
                            local_step_metrics
                        )
                        step_log = {
                            **reduced,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                        }
                        self.console_progress(
                            int(self.global_step),
                            len(train_dataloader) * num_epochs,
                            "train",
                        )

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            train_json_logger.log(step_log)
                            self.log_metrics(step_log, step=self.global_step)
                            self.global_step += 1

                        if (
                            cfg.training.max_train_steps is not None
                            and batch_idx >= (cfg.training.max_train_steps - 1)
                        ):
                            reached_max_steps = True
                            break

                    if not train_vla_losses:
                        self.global_step += 1
                        self.epoch += 1
                        continue

                    vla_count = max(
                        self.distributed.reduce_sum(len(train_vla_losses)), 1.0
                    )
                    step_log["train_vla_loss"] = (
                        self.distributed.reduce_sum(sum(train_vla_losses)) / vla_count
                    )
                    step_log["train_vla_token_loss"] = (
                        self.distributed.reduce_sum(sum(train_vla_token_losses))
                        / vla_count
                    )
                    step_log["train_vla_action_loss"] = (
                        self.distributed.reduce_sum(sum(train_vla_action_losses))
                        / vla_count
                    )

                    self.console_metrics(
                        f"train · epoch {self.epoch}",
                        {
                            "train/vla_loss": float(step_log["train_vla_loss"]),
                            "train/vla_token_loss": float(step_log["train_vla_token_loss"]),
                            "train/vla_action_loss": float(step_log["train_vla_action_loss"]),
                            "train/lr": float(step_log.get("lr", 0.0)),
                        },
                    )

                    # ---- Validation loss eval at end of epoch ----
                    eval_every = int(
                        OmegaConf.select(cfg, "eval.eval_every", default=1)
                    )
                    if val_dataloaders and (self.epoch % eval_every) == 0:
                        for split_name, val_dl in val_dataloaders.items():
                            val_metrics = self.evaluate_val_loss(val_dl, split_name)
                            step_log.update(val_metrics)

                    train_json_logger.log(step_log)
                    self.log_metrics(step_log, step=self.global_step)

                    if (self.epoch % cfg.training.checkpoint_every) == 0:
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        metric_dict = {
                            key.replace("/", "_"): value
                            for key, value in step_log.items()
                        }
                        topk_ckpt_path = None
                        if self.distributed.is_main_process:
                            topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        topk_ckpt_path = self.distributed.broadcast_object(
                            topk_ckpt_path
                        )
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)

                    self.global_step += 1
                    self.epoch += 1
                    if reached_max_steps:
                        break
                self.console_banner("TRAINING", done=True)
        finally:
            self.distributed.barrier()
            self.distributed.cleanup()

        return history


__all__ = ["PretokenizeVLARunner"]
