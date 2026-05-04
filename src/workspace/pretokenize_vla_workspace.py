"""VLA-only training workspace with LIBERO rollout evaluation after each epoch."""
from __future__ import annotations

import contextlib
import copy
import os
import pathlib
import pickle
import time
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

from src.dataloader import BaseDataset
from src.trainer import NopretokenizeSFTDistributedHelper
from src.utils.checkpoint_util import TopKCheckpointManager
from src.utils.ema import EMAHelper
from src.utils.optim import build_optimizer
from src.utils.seed import set_seed
from src.workspace.base_workspace import BaseWorkspace


class PretokenizeVLAWorkspace(BaseWorkspace):
    include_keys = ("global_step", "epoch")
    exclude_keys = tuple()
    default_vla_init_dir = "/home/user01/liops/workspace/DreamerVLA/data/ckpts/VLA_model_256/libero_10"
    default_output_dir = "/home/user01/liops/workspace/DreamerVLA/data/outputs/vla/debug_pretokenize_vla"

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        if output_dir is None:
            output_dir = str(OmegaConf.select(config, "training.out_dir", default=self.default_output_dir))
        super().__init__(config, output_dir=output_dir)

        self.distributed = NopretokenizeSFTDistributedHelper.initialize(
            strategy=str(OmegaConf.select(config, "training.distributed_strategy", default="ddp")),
            fsdp_mixed_precision=str(OmegaConf.select(config, "training.fsdp_mixed_precision", default="bf16")),
            enable_activation_checkpointing=bool(
                OmegaConf.select(config, "training.enable_activation_checkpointing", default=True)
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

    # ---- path helpers ----

    def _resolve_vla_init_path(self) -> str:
        configured = OmegaConf.select(self.cfg, "init.vla_ckpt_path")
        candidate = (
            pathlib.Path(str(configured)).expanduser().resolve()
            if configured is not None
            else pathlib.Path(self.default_vla_init_dir)
        )
        if candidate.is_dir():
            if (candidate / "config.json").is_file():
                return str(candidate)
            for subdir in sorted(path for path in candidate.iterdir() if path.is_dir()):
                if (subdir / "config.json").is_file():
                    return str(subdir.resolve())
        return str(candidate.resolve())

    def build_encoder_cfg(self, cfg: DictConfig) -> DictConfig:
        encoder_cfg = copy.deepcopy(cfg.encoder)
        init_model_path = OmegaConf.select(cfg, "init.vla_ckpt_path")
        if init_model_path is not None and OmegaConf.select(encoder_cfg, "model_path") is None:
            encoder_cfg.model_path = str(init_model_path)
        return encoder_cfg

    def _build_trainable_encoder_cfg(self, cfg: DictConfig) -> DictConfig:
        encoder_cfg = self.build_encoder_cfg(cfg)
        with open_dict(encoder_cfg):
            encoder_cfg.model_path = self._resolve_vla_init_path()
            train_encoder_backbone = bool(OmegaConf.select(cfg, "training.train_encoder_backbone", default=True))
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

    # ---- checkpoint ----

    def save_checkpoint(
        self,
        path: str | pathlib.Path | None = None,
        tag: str = "latest",
        exclude_keys: tuple[str, ...] | None = None,
        include_keys: tuple[str, ...] | None = None,
    ) -> str:
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        path = pathlib.Path(path)

        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir",)

        if not self.distributed.requires_collective_checkpointing and not self.distributed.is_main_process:
            return str(path.absolute())

        payload = {"cfg": self.cfg, "state_dicts": {}, "pickles": {}}
        for key, value in self.__dict__.items():
            if key in exclude_keys:
                continue
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                state_dict = self._state_dict_for_checkpoint(key, value)
                if self.distributed.is_main_process and state_dict is not None:
                    payload["state_dicts"][key] = _copy_to_cpu(state_dict)
            elif key in include_keys and self.distributed.is_main_process:
                payload["pickles"][key] = pickle.dumps(value)

        if self.distributed.is_main_process:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path)
        return str(path.absolute())

    def load_payload(
        self,
        payload: dict[str, Any],
        exclude_keys: tuple[str, ...] | None = None,
        include_keys: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = tuple(payload["pickles"].keys())

        for key, value in payload["state_dicts"].items():
            if key in exclude_keys or key not in self.__dict__:
                continue
            self._load_state_dict_from_checkpoint(key, self.__dict__[key], value, **kwargs)

        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = pickle.loads(payload["pickles"][key])

    def _state_dict_for_checkpoint(self, key: str, value: Any) -> dict[str, Any] | None:
        if key == "encoder" and self.encoder is not None:
            with self.distributed.model_state_dict_context(self.encoder.backbone):
                return self.encoder.state_dict()
        if key == "vla_optimizer" and self.vla_optimizer is not None and self.encoder is not None:
            return self.distributed.optimizer_state_dict(self.encoder.backbone, self.vla_optimizer)
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
        if key == "vla_optimizer" and self.vla_optimizer is not None and self.encoder is not None:
            self.distributed.load_optimizer_state_dict(self.encoder.backbone, self.vla_optimizer, state_dict)
            return
        value.load_state_dict(state_dict, **kwargs)

    # ---- Validation loss evaluation (FSDP-compatible) ----

    @torch.no_grad()
    def evaluate_val_loss(self, val_dataloader: DataLoader, split_name: str) -> dict[str, float]:
        """Compute VLA loss on a validation set. Runs on all ranks (FSDP-safe)."""
        self.encoder.eval()
        val_losses: list[float] = []
        val_token_losses: list[float] = []
        val_action_losses: list[float] = []

        for batch in val_dataloader:
            has_tokenized = isinstance(batch.get("input_ids"), list) and isinstance(batch.get("labels"), list)
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
            f"val_{split_name}_loss": self.distributed.reduce_sum(sum(val_losses)) / count,
            f"val_{split_name}_token_loss": self.distributed.reduce_sum(sum(val_token_losses)) / count,
            f"val_{split_name}_action_loss": self.distributed.reduce_sum(sum(val_action_losses)) / count,
        }
        if self.distributed.is_main_process:
            print(f"  [Val {split_name}] loss={metrics[f'val_{split_name}_loss']:.4f} "
                  f"token={metrics[f'val_{split_name}_token_loss']:.4f} "
                  f"action={metrics[f'val_{split_name}_action_loss']:.4f}")
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
                print("  [Eval] Skipping baseline eval under FSDP. Use scripts/eval_libero.sh on saved checkpoints.")
            return {}

        from libero.libero import benchmark as libero_benchmark
        from src.env import get_libero_env, get_libero_dummy_action, get_libero_image, quat2axisangle, TASK_MAX_STEPS

        task_suite_name = str(OmegaConf.select(eval_cfg, "task_suite_name", default="libero_goal"))
        num_episodes = int(OmegaConf.select(eval_cfg, "num_episodes_per_task", default=10))
        action_steps = int(OmegaConf.select(eval_cfg, "action_steps", default=10))
        resolution = int(OmegaConf.select(self.cfg, "encoder.resolution", default=256))
        # History length must match training (`his` in processed_data_generate_convs.sh).
        # Training builds img_c = [prev_third, prev_wrist, cur_third, cur_wrist] (his=2).
        history_length = int(OmegaConf.select(eval_cfg, "history_length", default=2))

        item_processor = self.encoder._build_processor(self.device)

        print(f"  [Eval] loading LIBERO benchmark suite '{task_suite_name}' ...", flush=True)
        benchmark_dict = libero_benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        num_tasks = task_suite.n_tasks
        max_tasks = OmegaConf.select(eval_cfg, "max_tasks", default=None)
        if max_tasks is not None:
            num_tasks = min(num_tasks, int(max_tasks))
        max_steps = int(OmegaConf.select(eval_cfg, "max_steps", default=TASK_MAX_STEPS.get(task_suite_name, 300)))
        print(
            f"  [Eval] suite='{task_suite_name}' num_tasks={num_tasks} "
            f"episodes_per_task={num_episodes} max_steps={max_steps} "
            f"action_steps={action_steps} history_length={history_length}",
            flush=True,
        )

        self.encoder.eval()
        backbone = self.distributed.unwrap_module(self.encoder.backbone)

        total_episodes, total_successes = 0, 0
        run_t0 = time.time()

        for task_id in range(num_tasks):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, resolution=resolution)
            n_eps = min(num_episodes, len(initial_states))
            print(
                f"  [Eval] >>> Task {task_id+1}/{num_tasks} start: \"{task_description}\" "
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
                # Frame history buffer: list of (third_view_pil, wrist_pil) oldest→newest.
                # Matches training's `img_history_start_idx = max(0, j - his + 1)` which
                # repeats the first frame until history fills up.
                frame_history: list[tuple[Image.Image, Image.Image]] = []

                steps_taken = 0
                for t in range(max_steps + 10):
                    if t < 10:
                        obs, _, done, _ = env.step(get_libero_dummy_action())
                        continue

                    img = get_libero_image(obs, resolution)
                    wrist_img = get_libero_image(obs, resolution, "robot0_eye_in_hand_image")
                    state = np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    )

                    third_pil = Image.fromarray(img)
                    wrist_pil = Image.fromarray(wrist_img)
                    frame_history.append((third_pil, wrist_pil))
                    if len(frame_history) > history_length:
                        frame_history = frame_history[-history_length:]

                    if len(actions_buffer) == 0:
                        # Pad with the oldest available frame when history is shorter
                        # than `history_length` (first action_steps steps of episode).
                        padded = [frame_history[0]] * (history_length - len(frame_history)) + frame_history
                        predicted = self._generate_actions(
                            backbone, item_processor,
                            padded, state, task_description, action_steps,
                        )
                        actions_buffer = predicted

                    if len(actions_buffer) == 0:
                        break
                    action = actions_buffer.pop(0)
                    obs, _, done, _ = env.step(action.tolist())
                    steps_taken = t - 9

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

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
                gpu_mb = (torch.cuda.memory_allocated() // (1024 * 1024)) if torch.cuda.is_available() else 0
                print(
                    f"  [Eval]   ep {episode_idx+1}/{n_eps} {tag} "
                    f"steps={steps_taken} time={ep_dt:5.1f}s "
                    f"task_succ={task_successes}/{episode_idx+1} "
                    f"total_succ={total_successes}/{total_episodes} "
                    f"({total_successes / max(total_episodes,1):.1%})  "
                    f"gpu_alloc={gpu_mb}MB",
                    flush=True,
                )
            env.close()

            rate = task_successes / max(n_eps, 1)
            task_dt = time.time() - task_t0
            print(
                f"  [Eval] <<< Task {task_id+1}/{num_tasks} done: \"{task_description}\" "
                f"success={rate:.1%} ({task_successes}/{n_eps}) "
                f"time={task_dt:.1f}s   running_total={total_successes}/{total_episodes} "
                f"({total_successes / max(total_episodes,1):.1%})",
                flush=True,
            )

        avg_success = total_successes / max(total_episodes, 1)
        run_dt = time.time() - run_t0
        metrics = {
            "eval_success_rate": avg_success,
            "eval_total_episodes": float(total_episodes),
            "eval_total_successes": float(total_successes),
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
        ordering used at training time (see src/preprocess/action_state_model_conv_generation.py:
        `img_c.extend(image_steps[step_idx])` with `img_names=[imgs_third_view, imgs_wrist]`).
        """
        img_c: list[Image.Image] = []
        for third_pil, wrist_pil in frame_history:
            img_c.extend([third_pil, wrist_pil])
        human_val = f"Finish the task: {task_description}." + "<|state|>" * 1 + "<|image|>" * len(img_c)

        conv = {
            "conversations": [{"from": "human", "value": human_val}],
            "image": img_c,
            "action": [],
            "state": [state],
        }
        tokens = item_processor.process_item(conv, training_mode=False)
        if isinstance(tokens, tuple):
            tokens = tokens[0]

        input_ids = torch.tensor(tokens, dtype=torch.int64, device=self.device).unsqueeze(0)

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
                action_sequences = backbone.generate_dis_ma(input_ids, generation_config_ma)
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
        min_values = np.array([-0.9375, -0.9375, -0.9375, -0.24214286, -0.375, -0.36428571, -1.0])
        max_values = np.array([0.9375, 0.9375, 0.9375, 0.34821429, 0.375, 0.375, 1.0])
        if action.shape[0] > 7:
            action = action[:7]
        return (action + 1) / 2 * (max_values - min_values + 1e-8) + min_values

    @staticmethod
    def _unnorm_actions(actions: np.ndarray) -> np.ndarray:
        min_values = np.array([-0.9375, -0.9375, -0.9375, -0.24214286, -0.375, -0.36428571, -1.0])
        max_values = np.array([0.9375, 0.9375, 0.9375, 0.34821429, 0.375, 0.375, 1.0])
        if actions.ndim == 2 and actions.shape[1] > 7:
            actions = actions[:, :7]
        return (actions + 1) / 2 * (max_values - min_values + 1e-8) + min_values

    # ---- main training loop ----

    def run(self) -> list[dict[str, float | str | int]]:
        history: list[dict[str, float | str | int]] = []
        if self.distributed.is_main_process:
            print("VLA Workspace begin.")
        cfg = copy.deepcopy(self.cfg)

        dataset: BaseDataset = hydra.utils.instantiate(cfg.dataset)
        assert isinstance(dataset, BaseDataset)

        dataloader_kwargs = dict(cfg.dataloader)
        sampler = self.distributed.maybe_make_sampler(
            dataset,
            shuffle=bool(dataloader_kwargs.get("shuffle", True)),
            drop_last=bool(dataloader_kwargs.get("drop_last", False)),
        )
        if sampler is not None:
            dataloader_kwargs["shuffle"] = False
            dataloader_kwargs["sampler"] = sampler
        collate_fn = getattr(dataset, "collate_fn", None)
        if callable(collate_fn):
            dataloader_kwargs["collate_fn"] = collate_fn
        train_dataloader = DataLoader(dataset, **dataloader_kwargs)

        # ---- Build val dataloaders (optional) ----
        val_dataloaders: dict[str, DataLoader] = {}
        for split_name in ("val_ind", "val_ood"):
            val_cfg_key = f"dataset_{split_name}"
            val_ds_cfg = OmegaConf.select(cfg, val_cfg_key, default=None)
            if val_ds_cfg is None:
                continue
            val_ds = hydra.utils.instantiate(val_ds_cfg)
            val_dl_kwargs = dict(cfg.dataloader)
            val_dl_kwargs["shuffle"] = False
            val_dl_kwargs["drop_last"] = False
            val_sampler = self.distributed.maybe_make_sampler(val_ds, shuffle=False, drop_last=False)
            if val_sampler is not None:
                val_dl_kwargs["sampler"] = val_sampler
            val_collate = getattr(val_ds, "collate_fn", None)
            if callable(val_collate):
                val_dl_kwargs["collate_fn"] = val_collate
            val_dataloaders[split_name] = DataLoader(val_ds, **val_dl_kwargs)

        encoder_cfg = self._build_trainable_encoder_cfg(cfg)
        self.encoder = hydra.utils.instantiate(encoder_cfg).to(self.device)
        if bool(OmegaConf.select(cfg, "training.vla_train_action_head_only", default=False)):
            matched = self._set_trainable_encoder_parameters(self.encoder, patterns=["action_head"])
            if matched == 0:
                raise ValueError("No trainable parameters matched pattern `action_head`.")
        self.distributed.wrap_encoder(self.encoder)
        vla_optim_cfg = OmegaConf.select(cfg, "optim.vla")
        if vla_optim_cfg is None:
            raise ValueError("`optim.vla` must be configured.")
        self.vla_optimizer = build_optimizer(self.encoder, vla_optim_cfg)

        # configure ema
        if bool(OmegaConf.select(cfg, "training.use_ema", default=False)) and self.vla_ema is None:
            self.vla_ema = EMAHelper(
                self.encoder,
                decay=float(OmegaConf.select(cfg, "ema.decay", default=0.9999)),
                update_after_step=int(OmegaConf.select(cfg, "ema.update_after_step", default=0)),
            )

        # resume training
        self.resume(cfg)

        # After optimizer state restore, param_groups may lack `initial_lr`
        # (LambdaLR requires it when last_epoch > -1).
        for pg in self.vla_optimizer.param_groups:
            pg.setdefault("initial_lr", pg["lr"])

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            str(OmegaConf.select(cfg, "training.lr_scheduler", default="constant")),
            optimizer=self.vla_optimizer,
            num_warmup_steps=int(OmegaConf.select(cfg, "training.lr_warmup_steps", default=0)),
            num_training_steps=(
                len(train_dataloader) * int(cfg.training.num_epochs)
            ) // int(cfg.training.gradient_accumulate_every),
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
                    if sampler is not None:
                        sampler.set_epoch(self.epoch)

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
                            has_tokenized = isinstance(batch.get("input_ids"), list) and isinstance(
                                batch.get("labels"), list
                            )
                            if not has_tokenized:
                                continue

                            vla_loss_dict = self.encoder.compute_action_sft_loss_from_tokenized(
                                input_ids_list=batch["input_ids"],
                                labels_list=batch["labels"],
                                token_loss_coef=float(
                                    OmegaConf.select(cfg, "training.vla_token_loss_coef", default=1.0)
                                ),
                                action_loss_coef=float(
                                    OmegaConf.select(cfg, "training.vla_action_loss_coef", default=1.0)
                                ),
                            )
                            vla_raw_loss = vla_loss_dict["loss"]
                            vla_loss = vla_raw_loss / cfg.training.gradient_accumulate_every
                            vla_loss.backward()

                            grad_clip_norm = cfg.optim.get("grad_clip_norm")
                            if grad_clip_norm is not None:
                                self.distributed.clip_grad_norm(self.encoder.backbone, float(grad_clip_norm))

                            self.vla_optimizer.step()
                            self.vla_optimizer.zero_grad(
                                set_to_none=bool(cfg.optim.get("zero_grad_set_to_none", True))
                            )
                            lr_scheduler.step()

                            # update ema
                            if self.vla_ema is not None:
                                self.vla_ema.step(self.encoder)

                            train_vla_losses.append(float(vla_raw_loss.item()))
                            train_vla_token_losses.append(float(vla_loss_dict["token_loss"].item()))
                            train_vla_action_losses.append(float(vla_loss_dict["action_loss"].item()))

                            local_step_metrics = {
                                "train_vla_loss": float(vla_raw_loss.item()),
                                "train_vla_token_loss": float(vla_loss_dict["token_loss"].item()),
                                "train_vla_action_loss": float(vla_loss_dict["action_loss"].item()),
                                "lr": float(lr_scheduler.get_last_lr()[0]),
                            }
                            reduced = self.distributed.reduce_mean_dict(local_step_metrics)
                            step_log = {**reduced, "global_step": self.global_step, "epoch": self.epoch}
                            tepoch.set_postfix(refresh=False, vla=float(step_log["train_vla_loss"]))

                            is_last_batch = batch_idx == (len(train_dataloader) - 1)
                            if not is_last_batch:
                                train_json_logger.log(step_log)
                                self.global_step += 1

                            if cfg.training.max_train_steps is not None and batch_idx >= (
                                cfg.training.max_train_steps - 1
                            ):
                                reached_max_steps = True
                                break

                    if not train_vla_losses:
                        self.global_step += 1
                        self.epoch += 1
                        continue

                    vla_count = max(self.distributed.reduce_sum(len(train_vla_losses)), 1.0)
                    step_log["train_vla_loss"] = self.distributed.reduce_sum(sum(train_vla_losses)) / vla_count
                    step_log["train_vla_token_loss"] = (
                        self.distributed.reduce_sum(sum(train_vla_token_losses)) / vla_count
                    )
                    step_log["train_vla_action_loss"] = (
                        self.distributed.reduce_sum(sum(train_vla_action_losses)) / vla_count
                    )

                    # ---- Validation loss eval at end of epoch ----
                    eval_every = int(OmegaConf.select(cfg, "eval.eval_every", default=1))
                    if val_dataloaders and (self.epoch % eval_every) == 0:
                        for split_name, val_dl in val_dataloaders.items():
                            val_metrics = self.evaluate_val_loss(val_dl, split_name)
                            step_log.update(val_metrics)

                    train_json_logger.log(step_log)

                    if (self.epoch % cfg.training.checkpoint_every) == 0:
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        metric_dict = {key.replace("/", "_"): value for key, value in step_log.items()}
                        topk_ckpt_path = None
                        if self.distributed.is_main_process:
                            topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        topk_ckpt_path = self.distributed.broadcast_object(topk_ckpt_path)
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


__all__ = ["PretokenizeVLAWorkspace"]


def _copy_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _copy_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_to_cpu(item) for item in value]
    return copy.deepcopy(value)
