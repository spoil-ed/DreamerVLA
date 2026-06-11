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
import tqdm
from diffusers.optimization import get_scheduler
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image
from torch.utils.data import DataLoader
from transformers import GenerationConfig

from dreamer_vla.dataset import BaseDataset
from dreamer_vla.runners.base_runner import BaseRunner
from dreamer_vla.trainer import NopretokenizeSFTDistributedHelper
from dreamer_vla.utils.checkpoint_util import TopKCheckpointManager
from dreamer_vla.utils.ema import EMAHelper
from dreamer_vla.utils.optim import build_optimizer
from dreamer_vla.utils.paths import checkpoints_path, data_path
from dreamer_vla.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PretokenizeVLARunner(BaseRunner):
    runner_name = "vla_sft"
    runner_status = "current"
    runner_family = "vla"
    include_keys = ("global_step", "epoch")
    exclude_keys = tuple()
    checkpoint_restore_output_dir = True
    default_vla_init_dir = str(checkpoints_path("VLA_model_256", "libero_goal"))
    default_output_dir = str(data_path("outputs", "vla", "debug_pretokenize_vla"))

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

        from libero.libero import benchmark as libero_benchmark

        from dreamer_vla.envs import (
            TASK_MAX_STEPS,
            get_libero_dummy_action,
            get_libero_env,
            get_libero_image,
            quat2axisangle,
            resolve_libero_eval_protocol,
            save_rollout_video,
            select_libero_action_chunk,
        )

        protocol = resolve_libero_eval_protocol(self.cfg, eval_cfg)
        seed = int(protocol["seed"])
        num_steps_wait = int(protocol["num_steps_wait"])
        np.random.seed(seed)
        task_suite_name = str(
            OmegaConf.select(eval_cfg, "task_suite_name", default="libero_goal")
        )
        num_episodes = int(
            OmegaConf.select(eval_cfg, "num_episodes_per_task", default=50)
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

        total_episodes, total_successes = 0, 0
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

                    img = get_libero_image(obs, resolution)
                    if should_record:
                        rollout_images.append(img)
                    wrist_img = get_libero_image(
                        obs, resolution, "robot0_eye_in_hand_image"
                    )
                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )

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
            task_dt = time.time() - task_t0
            print(
                f'  [Eval] <<< Task {task_id} ({task_index + 1}/{len(task_ids)}) done: "{task_description}" '
                f"success={rate:.1%} ({task_successes}/{n_eps}) "
                f"time={task_dt:.1f}s   running_total={total_successes}/{total_episodes} "
                f"({total_successes / max(total_episodes, 1):.1%})",
                flush=True,
            )

        avg_success = total_successes / max(total_episodes, 1)
        run_dt = time.time() - run_t0
        metrics = {
            "eval_success_rate": avg_success,
            "eval_total_episodes": float(total_episodes),
            "eval_total_successes": float(total_successes),
            "results/total_success_rate": avg_success,
            "results/total_episodes": float(total_episodes),
            "results/total_successes": float(total_successes),
        }
        print(
            f"  [Eval] Epoch {epoch} overall success rate: {avg_success:.1%} "
            f"({total_successes}/{total_episodes}) total_time={run_dt:.1f}s",
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
        ordering used at training time (see dreamer_vla/preprocess/action_state_model_conv_generation.py:
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

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            str(OmegaConf.select(cfg, "training.lr_scheduler", default="constant")),
            optimizer=self.vla_optimizer,
            num_warmup_steps=int(
                OmegaConf.select(cfg, "training.lr_warmup_steps", default=0)
            ),
            num_training_steps=(len(train_dataloader) * int(cfg.training.num_epochs))
            // int(cfg.training.gradient_accumulate_every),
            last_epoch=self.global_step - 1,
        )

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        if cfg.training.debug:
            cfg.training.num_epochs = 3
            cfg.training.max_train_steps = 2
            cfg.training.checkpoint_every = 1

        if self.distributed.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
        self.distributed.barrier()
        train_log_path = os.path.join(self.output_dir, "vla_logs.json.txt")
        train_logger_cm = self.distributed.logger_context(train_log_path)

        if val_dataloaders and self.distributed.is_main_process:
            print(f"  Val dataloaders ready: {list(val_dataloaders.keys())}")

        try:
            with train_logger_cm as train_json_logger:
                reached_max_steps = False
                for _local_epoch_idx in range(cfg.training.num_epochs - self.epoch):
                    self.set_dataloader_epoch(train_dataloader, self.epoch)

                    step_log: dict[str, float | str | int] = {}
                    train_vla_losses: list[float] = []
                    train_vla_token_losses: list[float] = []
                    train_vla_action_losses: list[float] = []

                    self.encoder.train()
                    with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Training epoch {self.epoch}",
                        disable=not self.distributed.is_main_process,
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
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
                            tepoch.set_postfix(
                                refresh=False, vla=float(step_log["train_vla_loss"])
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
        finally:
            self.distributed.barrier()
            self.distributed.cleanup()

        return history


__all__ = ["PretokenizeVLARunner"]
