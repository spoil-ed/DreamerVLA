"""World-model-only training workspace (TSSM)."""
from __future__ import annotations

import copy
import os
import pathlib
import pickle
from typing import Any

import hydra
import torch
import tqdm
from diffusers.optimization import get_scheduler
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from src.dataloader import BaseDataset
from src.trainer import NopretokenizeSFTDistributedHelper
from src.utils.checkpoint_util import TopKCheckpointManager
from src.utils.ema import EMAHelper
from src.utils.optim import build_optimizer
from src.utils.seed import set_seed
from src.workspace.base_workspace import BaseWorkspace


class PretokenizeWMWorkspace(BaseWorkspace):
    include_keys = ("global_step", "epoch")
    exclude_keys = tuple()
    default_vla_init_dir = "/home/user01/yuxinglei/workspace/DreamerVLA/data/ckpts/VLA_model_256/libero_10"
    default_output_dir = "/home/user01/yuxinglei/workspace/DreamerVLA/data/outputs/debug_pretokenize_wm"

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
        self.encoder = None  # frozen encoder for obs embedding
        self.world_model = None
        self.world_model_optimizer = None
        self.world_model_ema: EMAHelper | None = None
        self.image_visualizer = None  # WorldModelImageVisualizer, main-process only

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

    def _build_frozen_encoder_cfg(self, cfg: DictConfig) -> DictConfig:
        encoder_cfg = copy.deepcopy(cfg.encoder)
        init_model_path = OmegaConf.select(cfg, "init.vla_ckpt_path")
        if init_model_path is not None and OmegaConf.select(encoder_cfg, "model_path") is None:
            encoder_cfg.model_path = str(init_model_path)
        with open_dict(encoder_cfg):
            encoder_cfg.model_path = self._resolve_vla_init_path()
            encoder_cfg.freeze_backbone = True
        return encoder_cfg

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
            if key == "encoder":
                continue  # encoder is frozen, no need to save
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
            if key in exclude_keys or key not in self.__dict__ or self.__dict__[key] is None:
                continue
            self._load_state_dict_from_checkpoint(key, self.__dict__[key], value, **kwargs)

        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = pickle.loads(payload["pickles"][key])

    def _state_dict_for_checkpoint(self, key: str, value: Any) -> dict[str, Any] | None:
        if key == "world_model" and self.world_model is not None:
            with self.distributed.model_state_dict_context(self.world_model):
                return self.world_model.state_dict()
        if key == "world_model_optimizer" and self.world_model_optimizer is not None and self.world_model is not None:
            return self.distributed.optimizer_state_dict(self.world_model, self.world_model_optimizer)
        return value.state_dict()

    def _preresume_world_model_weights(self, cfg: DictConfig) -> None:
        """Load world_model weights into the UNWRAPPED nn.Module (before FSDP).

        FSDP wrap uses ``sync_module_states=True`` which broadcasts rank 0's
        params to all ranks during wrap construction — so only rank 0 needs
        to load the ckpt here.  The subsequent `self.resume(cfg)` will still
        load optimizer / global_step / epoch, but will skip the world_model
        state_dict (see `_load_state_dict_from_checkpoint`).
        """
        if not bool(OmegaConf.select(cfg, "training.resume", default=False)):
            return
        ckpt_path = self.get_checkpoint_path()
        if not ckpt_path.is_file():
            if self.distributed.is_main_process:
                print(f"[pre-resume] ckpt not found at {ckpt_path}; skipping pre-load.")
            return

        if self.distributed.is_main_process:
            print(f"[pre-resume] loading world_model weights from {ckpt_path} ...")
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            wm_sd = payload["state_dicts"].get("world_model")
            if wm_sd is None:
                print("[pre-resume] no world_model in ckpt; skipping.")
            else:
                # Cast to target dtype before loading (model is already in bf16).
                target_dtype = next(self.world_model.parameters()).dtype
                wm_sd = {
                    k: (v.to(dtype=target_dtype) if torch.is_floating_point(v) else v)
                    for k, v in wm_sd.items()
                }
                missing, unexpected = self.world_model.load_state_dict(wm_sd, strict=False)
                print(f"[pre-resume] loaded world_model: {len(wm_sd)} tensors, "
                      f"missing={len(missing)}, unexpected={len(unexpected)}")
                if missing:
                    print(f"[pre-resume] missing (first 5): {missing[:5]}")
                if unexpected:
                    print(f"[pre-resume] unexpected (first 5): {unexpected[:5]}")
            del payload
        # All ranks wait for rank 0 to finish loading so FSDP wrap sees the
        # fully loaded state before sync_module_states broadcasts.
        self.distributed.barrier()

    def _load_state_dict_from_checkpoint(
        self,
        key: str,
        value: Any,
        state_dict: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        if key in ("world_model", "_unwrapped_world_model") and self.world_model is not None:
            # World-model weights are loaded BEFORE FSDP wrap via
            # `_preresume_world_model_weights` (called from run()), so that
            # FSDP's `sync_module_states=True` can broadcast them to all ranks.
            # Loading via FSDP `load_state_dict` under `use_orig_params=True`
            # does not work with a monolithic torch.save ckpt (shape mismatch
            # between 2D ckpt tensors and 1D flat-sharded param views).
            # `_unwrapped_world_model` is just a reference to the same params
            # as `world_model` (set before FSDP wrap), so it's redundant too.
            if self.distributed.is_main_process:
                print(f"[resume] skipping {key} state_dict (WM pre-loaded before FSDP wrap).")
            return
        if key == "world_model_optimizer" and self.world_model_optimizer is not None and self.world_model is not None:
            self.distributed.load_optimizer_state_dict(self.world_model, self.world_model_optimizer, state_dict)
            return
        value.load_state_dict(state_dict, **kwargs)

    # ---- validation ----

    @torch.no_grad()
    def evaluate_val_loss(self, val_dataloader: DataLoader, split_name: str) -> dict[str, float]:
        self.world_model.eval()
        val_losses: list[float] = []
        val_transition_losses: list[float] = []
        val_kl_losses: list[float] = []

        for batch in val_dataloader:
            wm_batch = self._build_world_model_batch(batch)
            if wm_batch is None:
                continue
            wm_loss_dict = self.world_model(wm_batch)
            val_losses.append(float(wm_loss_dict["loss"].item()))
            val_transition_losses.append(float(wm_loss_dict["transition_loss"].item()))
            val_kl_losses.append(float(wm_loss_dict["kl_loss"].item()))

        self.world_model.train()

        if not val_losses:
            return {}

        count = max(self.distributed.reduce_sum(len(val_losses)), 1.0)
        metrics = {
            f"val_{split_name}_wm_loss": self.distributed.reduce_sum(sum(val_losses)) / count,
            f"val_{split_name}_wm_transition_loss": self.distributed.reduce_sum(sum(val_transition_losses)) / count,
            f"val_{split_name}_wm_kl_loss": self.distributed.reduce_sum(sum(val_kl_losses)) / count,
        }
        if self.distributed.is_main_process:
            print(
                f"  [Val {split_name}] wm={metrics[f'val_{split_name}_wm_loss']:.4f} "
                f"tr={metrics[f'val_{split_name}_wm_transition_loss']:.4f} "
                f"kl={metrics[f'val_{split_name}_wm_kl_loss']:.4f}"
            )
        return metrics

    # ---- world model batch building ----

    def _build_world_model_batch(self, batch: dict[str, Any]) -> dict[str, Any] | None:
        wm = self.world_model
        wm_inner = getattr(self, "_unwrapped_world_model", None) or wm
        spatial_codec = bool(getattr(wm_inner, "spatial_codec", False))
        io_mode = str(getattr(wm_inner, "io_mode", "hidden"))
        need_image_hiddens = bool(
            getattr(wm_inner, "image_decoder_enabled", False)
            and getattr(wm_inner, "image_decoder_loss_coef", 0.0) > 0.0
        ) if wm is not None else False
        # Under spatial_codec the per-image-token hiddens are themselves the
        # obs_embedding, so we don't need the separate MLP-decoder target.
        if spatial_codec:
            need_image_hiddens = False

        if (
            "obs_embedding" not in batch
            and isinstance(batch.get("wm_obs_input_ids"), list)
            and isinstance(batch.get("wm_next_obs_input_ids"), list)
        ):
            if io_mode == "token":
                # Token mode: skip the frozen Chameleon forward entirely.
                # Just extract the third-view image BPE ids from each sample.
                obs_embedding = self._extract_image_bpe_ids(batch["wm_obs_input_ids"])
                next_obs_embedding = self._extract_image_bpe_ids(batch["wm_next_obs_input_ids"])
                batch["obs_embedding"] = obs_embedding
                batch["next_obs_embedding"] = next_obs_embedding
            elif self.encoder is not None:
                obs_embedding, _, _ = self._encode_hidden_from_tokenized(
                    batch["wm_obs_input_ids"],
                    return_image_hiddens=False,
                    return_image_token_ids=False,
                    per_token_embedding=spatial_codec,
                )
                (
                    next_obs_embedding,
                    next_image_hiddens,
                    next_image_token_ids,
                ) = self._encode_hidden_from_tokenized(
                    batch["wm_next_obs_input_ids"],
                    return_image_hiddens=need_image_hiddens,
                    return_image_token_ids=spatial_codec,
                    per_token_embedding=spatial_codec,
                )
                batch["obs_embedding"] = obs_embedding
                batch["next_obs_embedding"] = next_obs_embedding
                if next_image_hiddens is not None:
                    batch["next_obs_image_hiddens"] = next_image_hiddens
                if next_image_token_ids is not None:
                    batch["next_obs_image_token_ids"] = next_image_token_ids

        wm_batch: dict[str, Any] = {}
        for key in (
            "obs_embedding", "next_obs_embedding", "action", "action_mask",
            "reward", "next_obs_image_hiddens", "next_obs_image_token_ids",
        ):
            value = batch.get(key)
            if value is not None:
                wm_batch[key] = value

        required = ("obs_embedding", "next_obs_embedding", "action")
        if not all(isinstance(wm_batch.get(key), torch.Tensor) for key in required):
            return None

        for key in (
            "obs_embedding", "next_obs_embedding", "action", "action_mask",
            "reward", "next_obs_image_hiddens", "next_obs_image_token_ids",
        ):
            value = wm_batch.get(key)
            if isinstance(value, torch.Tensor):
                wm_batch[key] = value.to(self.device)
        return wm_batch

    def _extract_image_bpe_ids(
        self, input_ids_list: list[list[int]],
    ) -> torch.Tensor:
        """Token-mode helper: pull the third-view image block's BPE ids out of
        each sample's input_ids, without invoking the frozen encoder.

        Returns [B, n_img_tok] long tensor of image BPE ids.
        """
        from src.utils.wm_image_viz import extract_image_blocks

        wm_inner = getattr(self, "_unwrapped_world_model", None) or self.world_model
        n_img_tok = int(getattr(wm_inner, "n_image_tokens", 256))
        which_block = int(OmegaConf.select(self.cfg, "viz.which_block", default=-2))
        img_bpe = self._get_image_bpe_set()

        if not input_ids_list:
            return torch.zeros((0, n_img_tok), dtype=torch.long, device=self.device)

        rows: list[list[int]] = []
        for idx, seq in enumerate(input_ids_list):
            blocks = extract_image_blocks(list(seq))
            if not blocks:
                raise ValueError(f"sample {idx}: no image block found in tokens")
            bidx = which_block if which_block >= 0 else len(blocks) + which_block
            if not (0 <= bidx < len(blocks)):
                raise ValueError(
                    f"sample {idx}: which_block={which_block} out of range (have {len(blocks)} blocks)"
                )
            _start, _end, block_ids = blocks[bidx]
            tok_ids = [int(tok) for tok in block_ids if int(tok) in img_bpe]
            if len(tok_ids) != n_img_tok:
                raise ValueError(
                    f"sample {idx}: block has {len(tok_ids)} image tokens, expected {n_img_tok}"
                )
            rows.append(tok_ids)
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    def _get_image_bpe_set(self) -> set[int]:
        cached = getattr(self, "_image_bpe_set_cache", None)
        if cached is not None:
            return cached
        vocab_mapping = self.encoder.backbone.model.vocabulary_mapping
        self._image_bpe_set_cache = set(vocab_mapping.bpe2img.keys())
        return self._image_bpe_set_cache

    def _encode_hidden_from_tokenized(
        self,
        input_ids_list: list[list[int]],
        return_image_hiddens: bool = False,
        return_image_token_ids: bool = False,
        per_token_embedding: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Run the frozen VLA backbone and package frame embeddings.

        Returns a 3-tuple ``(embedding, image_hiddens, image_token_ids)``:
          - ``embedding``
              * `per_token_embedding=False` (route-0): pooled  `[B, hidden_dim]`.
              * `per_token_embedding=True`  (route-B): per-image-token hiddens
                `[B, n_img_tok, hidden_dim]`, ready to feed into the WM's
                learnable conv stem.
          - ``image_hiddens``: optional per-image-token hiddens (legacy MLP
            image_decoder target).
          - ``image_token_ids``: optional `[B, n_img_tok]` bpe ids of the
            target image block (used for route-B CE loss).
        """
        if self.encoder is None:
            raise ValueError("Encoder is required for token-level world-model conditioning.")
        if not input_ids_list:
            wm_inner = getattr(self, "_unwrapped_world_model", None) or self.world_model
            in_ch = int(getattr(wm_inner, "in_channels", 0) or 0)
            hidden_dim = int(OmegaConf.select(self.cfg, "world_model.hidden_dim", default=1))
            if per_token_embedding and in_ch > 0:
                n_img_tok = int(getattr(wm_inner, "n_image_tokens", 256))
                empty = torch.zeros((0, n_img_tok, in_ch), device=self.device, dtype=torch.float32)
            else:
                empty = torch.zeros((0, hidden_dim), device=self.device, dtype=torch.float32)
            return empty, None, None
        labels_list = [[-100] * len(example) for example in input_ids_list]
        lengths = [len(example) for example in input_ids_list]
        with torch.no_grad():
            _, _, _, hidden_states, _, _, _ = self.encoder.backbone(
                input_ids=input_ids_list,
                labels=labels_list,
                training=True,
                output_hidden_states=True,
                att_mask=False,
            )
        attention_mask = torch.zeros(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        for idx, length in enumerate(lengths):
            if length > 0:
                attention_mask[idx, :length] = True

        # Compute per-image-token hiddens / token ids lazily — needed by both
        # the route-B primary embedding and the legacy image_hiddens target.
        need_per_token = per_token_embedding or return_image_hiddens or return_image_token_ids

        per_token_hiddens: torch.Tensor | None = None
        per_token_ids: torch.Tensor | None = None
        if need_per_token:
            from src.utils.wm_image_viz import extract_image_blocks
            wm_inner = getattr(self, "_unwrapped_world_model", None) or self.world_model
            n_img_tok = int(getattr(wm_inner, "n_image_tokens", 256))
            which_block = int(OmegaConf.select(self.cfg, "viz.which_block", default=-2))
            img_bpe = self._get_image_bpe_set()
            hidden_samples: list[torch.Tensor] = []
            id_samples: list[torch.Tensor] = []
            for idx, seq in enumerate(input_ids_list):
                blocks = extract_image_blocks(list(seq))
                if not blocks:
                    raise ValueError(f"sample {idx}: no image block found in tokens")
                bidx = which_block if which_block >= 0 else len(blocks) + which_block
                if not (0 <= bidx < len(blocks)):
                    raise ValueError(
                        f"sample {idx}: which_block={which_block} out of range (have {len(blocks)} blocks)"
                    )
                start, _end, block_ids = blocks[bidx]
                positions = [
                    start + off for off, tok in enumerate(block_ids) if int(tok) in img_bpe
                ]
                tok_ids = [tok for tok in block_ids if int(tok) in img_bpe]
                if len(positions) != n_img_tok:
                    raise ValueError(
                        f"sample {idx}: block has {len(positions)} image tokens, expected {n_img_tok}"
                    )
                pos_t = torch.tensor(positions, device=hidden_states.device)
                hidden_samples.append(hidden_states[idx].index_select(0, pos_t))
                id_samples.append(torch.tensor(tok_ids, dtype=torch.long))
            per_token_hiddens = torch.stack(hidden_samples, dim=0).float().detach()
            per_token_ids = torch.stack(id_samples, dim=0)

        if per_token_embedding:
            embedding = per_token_hiddens
        else:
            weights = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
            embedding = (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            embedding = embedding.float().detach()

        image_hiddens = per_token_hiddens if return_image_hiddens else None
        image_token_ids = per_token_ids if return_image_token_ids else None
        return embedding, image_hiddens, image_token_ids

    # ---- image visualisation ----

    def _maybe_build_image_visualizer(self, cfg: DictConfig) -> None:
        viz_cfg = OmegaConf.select(cfg, "viz")
        if viz_cfg is None or not bool(OmegaConf.select(viz_cfg, "enabled", default=False)):
            return
        if not self.distributed.is_main_process:
            return
        if self.encoder is None:
            return
        vqgan_cfg = OmegaConf.select(cfg, "encoder.chameleon_vqgan_config")
        vqgan_ckpt = OmegaConf.select(cfg, "encoder.chameleon_vqgan_ckpt")
        if vqgan_cfg is None or vqgan_ckpt is None:
            if self.distributed.is_main_process:
                print("[viz] encoder.chameleon_vqgan_{config,ckpt} not set; skipping image viz.")
            return
        try:
            from src.utils.wm_image_viz import WorldModelImageVisualizer
            self.image_visualizer = WorldModelImageVisualizer(
                vqgan_config_path=str(vqgan_cfg),
                vqgan_ckpt_path=str(vqgan_ckpt),
                encoder=self.encoder,
                device=self.device,
                which_block=int(OmegaConf.select(viz_cfg, "which_block", default=-2)),
            )
            print(f"[viz] image visualiser ready (which_block={self.image_visualizer.which_block}).")
        except Exception as exc:
            print(f"[viz] failed to build image visualiser, disabling: {exc}")
            self.image_visualizer = None

    def _maybe_log_images(
        self,
        cfg: DictConfig,
        batch: dict[str, Any],
    ) -> None:
        viz_cfg = OmegaConf.select(cfg, "viz")
        every = int(OmegaConf.select(viz_cfg, "every_n_steps", default=500))
        if every <= 0 or (self.global_step % every) != 0:
            return  # all ranks return together — no collective needed

        # summon_full_params is collective, so ALL ranks must enter the same
        # context. Non-main ranks enter but do nothing inside.
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if isinstance(self.world_model, FSDP):
            summon_ctx = FSDP.summon_full_params(
                self.world_model, recurse=True, rank0_only=True, writeback=False,
            )
        else:
            import contextlib as _ctx
            summon_ctx = _ctx.nullcontext()

        with summon_ctx:
            if self.image_visualizer is None or not self.distributed.is_main_process:
                return
            obs_ids = batch.get("wm_obs_input_ids")
            nxt_ids = batch.get("wm_next_obs_input_ids")
            action = batch.get("action")
            if not isinstance(obs_ids, list) or not isinstance(nxt_ids, list) or action is None:
                return

            num_samples = int(OmegaConf.select(viz_cfg, "num_samples", default=4))
            out_dir = pathlib.Path(self.output_dir) / "viz"
            tag = f"step{self.global_step:07d}"
            world_model = getattr(self, "_unwrapped_world_model", self.world_model)
            try:
                paths = self.image_visualizer.visualize_batch(
                    world_model=world_model,
                    wm_obs_input_ids=obs_ids,
                    wm_next_obs_input_ids=nxt_ids,
                    action=action if isinstance(action, torch.Tensor) else torch.as_tensor(action),
                    out_dir=out_dir,
                    tag=tag,
                    num_samples=num_samples,
                )
                if paths:
                    print(f"[viz] step {self.global_step}: wrote {len(paths)} panel(s) under {out_dir}")
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(f"[viz] step {self.global_step}: visualisation failed: {exc}")

    # ---- main training loop ----

    def run(self) -> list[dict[str, float | str | int]]:
        history: list[dict[str, float | str | int]] = []
        if self.distributed.is_main_process:
            print("WM Workspace begin.")
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

        # configure validation dataset
        val_dataloaders: dict[str, DataLoader] = {}
        for split_name in ("val_ind", "val_ood"):
            val_ds_cfg = OmegaConf.select(cfg, f"dataset_{split_name}", default=None)
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

        # Frozen encoder for obs embedding extraction
        encoder_cfg = OmegaConf.select(cfg, "encoder")
        if encoder_cfg is not None:
            encoder_cfg = self._build_frozen_encoder_cfg(cfg)
            self.encoder = hydra.utils.instantiate(encoder_cfg).to(self.device)
            # Freeze all encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        # Trainable world model
        world_model_cfg = OmegaConf.select(cfg, "world_model")
        if world_model_cfg is None:
            raise ValueError("`world_model` config is required for WM workspace.")

        world_model_hidden_dim = self.infer_hidden_dim_from_dataset(dataset)
        if world_model_hidden_dim is None:
            world_model_hidden_dim = self.infer_hidden_dim_from_encoder(self.encoder)

        # Auto-fill num_image_tokens_vocab from the encoder's vocab mapping if
        # the user enabled io_mode='token' but did not set it explicitly.
        instantiate_kwargs: dict[str, Any] = {}
        if world_model_hidden_dim is not None:
            instantiate_kwargs["hidden_dim"] = world_model_hidden_dim
        if (
            str(OmegaConf.select(world_model_cfg, "io_mode", default="hidden")) == "token"
            and OmegaConf.select(world_model_cfg, "num_image_tokens_vocab") is None
            and self.encoder is not None
        ):
            vocab_mapping = self.encoder.backbone.model.vocabulary_mapping
            instantiate_kwargs["num_image_tokens_vocab"] = len(vocab_mapping.bpe2img)

        self.world_model = hydra.utils.instantiate(
            world_model_cfg, **instantiate_kwargs
        ).to(self.device)

        

        # Debug overrides: small epochs / frequent viz / frequent ckpt so the
        # pipeline can be exercised end-to-end in a minute.  Applied early
        # (before lr_scheduler uses num_epochs) so all downstream code sees
        # the debug values consistently.
        if bool(OmegaConf.select(cfg, "training.debug", default=False)):
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 5
            cfg.training.checkpoint_every = 1
            if OmegaConf.select(cfg, "viz") is not None:
                cfg.viz.every_n_steps = 1
                cfg.viz.num_samples = 2
            if self.distributed.is_main_process:
                print(
                    "[debug] training.debug=True → "
                    f"num_epochs={cfg.training.num_epochs}, "
                    f"max_train_steps={cfg.training.max_train_steps}, "
                    f"viz.every_n_steps={OmegaConf.select(cfg, 'viz.every_n_steps')}"
                )

        # Under route-B (spatial_codec) the WM computes a CE loss over
        # image-token logits.  Attach the image-vocab mapping and (hidden mode
        # only) a reference to the frozen LLM lm_head.  Token mode produces
        # logits directly from its own decoder, so lm_head is not needed.
        if getattr(self.world_model, "spatial_codec", False) and self.encoder is not None:
            try:
                lm_head = self.encoder.backbone.lm_head
                vocab_mapping = self.encoder.backbone.model.vocabulary_mapping
                image_token_bpe_ids = torch.tensor(
                    sorted(vocab_mapping.bpe2img.keys()), dtype=torch.long,
                )
                full_vocab_size = int(lm_head.weight.shape[0])
                wm_io_mode = str(getattr(self.world_model, "io_mode", "hidden"))
                self.world_model.attach_lm_head(
                    lm_head if wm_io_mode == "hidden" else None,
                    image_token_bpe_ids,
                    full_vocab_size=full_vocab_size,
                )
                if self.distributed.is_main_process:
                    tag = "lm_head" if wm_io_mode == "hidden" else "vocab (token mode, no lm_head)"
                    print(
                        f"[wm] attached {tag} for CE loss "
                        f"(image_vocab={image_token_bpe_ids.numel()}, full_vocab={full_vocab_size})"
                    )
            except Exception as exc:
                if self.distributed.is_main_process:
                    print(f"[wm] failed to attach lm_head — CE loss will stay at 0: {exc}")

        # Ensure uniform dtype before FSDP wrapping.  The transition backbone is
        # loaded in bfloat16 while the remaining heads default to float32; FSDP
        # requires all parameters to share a single dtype before it can build its
        # flat-parameter handle.  Cast the entire module to bfloat16 here so the
        # FSDP MixedPrecision (bf16) policy sees a consistent starting dtype.
        fsdp_precision = str(OmegaConf.select(cfg, "training.fsdp_mixed_precision", default="bf16"))
        _precision_to_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        _target_dtype = _precision_to_dtype.get(fsdp_precision, torch.bfloat16)
        self.world_model = self.world_model.to(dtype=_target_dtype)

        # Pre-resume: load world-model weights into the UNWRAPPED module so
        # FSDP's `sync_module_states=True` can broadcast them to all ranks
        # during wrap.  We cannot load weights through FSDP `load_state_dict`
        # under `use_orig_params=True` because the flat 1D sharded views don't
        # match 2D dict tensors.  Only rank 0 reads the ckpt; barrier ensures
        # all ranks wait before entering FSDP wrap.
        self._preresume_world_model_weights(cfg)

        world_optim_cfg = OmegaConf.select(cfg, "optim.world_model")
        if world_optim_cfg is None:
            raise ValueError("`optim.world_model` must be configured.")

        # Build the pixel-decoding image visualiser on main process, before
        # FSDP wraps the world model (so we still have a plain nn.Module
        # reference with predict_next_hidden). The visualiser keeps its own
        # unwrapped handle — no weight copy, just the reference.
        self._maybe_build_image_visualizer(cfg)
        self._unwrapped_world_model = self.world_model

        self.world_model = self.distributed.wrap_trainable_module(self.world_model)
        self.world_model_optimizer = build_optimizer(self.world_model, world_optim_cfg)

        # configure ema
        if bool(OmegaConf.select(cfg, "training.use_ema", default=False)) and self.world_model_ema is None:
            self.world_model_ema = EMAHelper(
                self.world_model,
                decay=float(OmegaConf.select(cfg, "ema.decay", default=0.9999)),
                update_after_step=int(OmegaConf.select(cfg, "ema.update_after_step", default=0)),
            )

        # resume training
        self.resume(cfg)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            str(OmegaConf.select(cfg, "training.lr_scheduler", default="constant")),
            optimizer=self.world_model_optimizer,
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

        if self.distributed.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
        self.distributed.barrier()
        train_log_path = os.path.join(self.output_dir, "wm_logs.json.txt")
        train_logger_cm = self.distributed.logger_context(train_log_path)

        try:
            with train_logger_cm as train_json_logger:
                reached_max_steps = False
                for _local_epoch_idx in range(cfg.training.num_epochs):
                    if sampler is not None:
                        sampler.set_epoch(self.epoch)

                    step_log: dict[str, float | str | int] = {}
                    train_wm_losses: list[float] = []
                    train_wm_transition_losses: list[float] = []
                    train_wm_kl_losses: list[float] = []
                    train_wm_reward_losses: list[float] = []

                    self.world_model.train()
                    with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Training epoch {self.epoch}",
                        disable=not self.distributed.is_main_process,
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            wm_batch = self._build_world_model_batch(batch)
                            if wm_batch is None:
                                continue

                            wm_loss_dict = self.world_model(wm_batch)
                            wm_raw_loss = wm_loss_dict["loss"]
                            wm_loss = wm_raw_loss / cfg.training.gradient_accumulate_every
                            wm_loss.backward()

                            grad_clip_norm = cfg.optim.get("grad_clip_norm")
                            if grad_clip_norm is not None:
                                pre_clip_grad_norm = float(
                                    self.distributed.clip_grad_norm(
                                        self.world_model, float(grad_clip_norm),
                                    )
                                )
                            else:
                                pre_clip_grad_norm = float("nan")

                            self.world_model_optimizer.step()
                            self.world_model_optimizer.zero_grad(
                                set_to_none=bool(cfg.optim.get("zero_grad_set_to_none", True))
                            )
                            lr_scheduler.step()

                            # update ema
                            if self.world_model_ema is not None:
                                self.world_model_ema.step(self.world_model)

                            train_wm_losses.append(float(wm_raw_loss.item()))
                            train_wm_transition_losses.append(float(wm_loss_dict["transition_loss"].item()))
                            train_wm_kl_losses.append(float(wm_loss_dict["kl_loss"].item()))
                            if "reward_loss" in wm_loss_dict:
                                train_wm_reward_losses.append(float(wm_loss_dict["reward_loss"].item()))

                            def _pick(key: str) -> float:
                                v = wm_loss_dict.get(key)
                                return float(v.item()) if isinstance(v, torch.Tensor) else float("nan")

                            local_step_metrics = {
                                "train_wm_loss": float(wm_raw_loss.item()),
                                "train_wm_transition_loss": _pick("transition_loss"),
                                "train_wm_kl_loss": _pick("kl_loss"),
                                "train_wm_dyn_kl": _pick("dyn_kl"),
                                "train_wm_rep_kl": _pick("rep_kl"),
                                "train_wm_reward_loss": _pick("reward_loss"),
                                "train_wm_image_recon_ce_loss":  _pick("image_recon_ce_loss"),
                                "train_wm_image_recon_mse_loss": _pick("image_recon_mse_loss"),
                                "train_wm_image_decoder_loss":   _pick("image_decoder_loss"),
                                "train_wm_image_recon_accuracy": _pick("image_recon_accuracy"),
                                "train_wm_pred_entropy":         _pick("pred_entropy"),
                                "train_wm_pred_unique_tokens":   _pick("pred_unique_tokens"),
                                "train_wm_gt_unique_tokens":     _pick("gt_unique_tokens"),
                                "train_wm_grad_norm": pre_clip_grad_norm,
                                "lr": float(lr_scheduler.get_last_lr()[0]),
                            }
                            reduced = self.distributed.reduce_mean_dict(local_step_metrics)
                            step_log = {**reduced, "global_step": self.global_step, "epoch": self.epoch}
                            tepoch.set_postfix(
                                refresh=False,
                                wm=float(step_log["train_wm_loss"]),
                                kl=float(step_log["train_wm_kl_loss"]),
                                ce=float(step_log.get("train_wm_image_recon_ce_loss", float("nan"))),
                                acc=float(step_log.get("train_wm_image_recon_accuracy", float("nan"))),
                                uniq=float(step_log.get("train_wm_pred_unique_tokens", float("nan"))),
                            )

                            self._maybe_log_images(cfg, batch)

                            is_last_batch = batch_idx == (len(train_dataloader) - 1)
                            if not is_last_batch:
                                train_json_logger.log(step_log)
                                self.global_step += 1

                            if cfg.training.max_train_steps is not None and batch_idx >= (
                                cfg.training.max_train_steps - 1
                            ):
                                reached_max_steps = True
                                break

                    if not train_wm_losses:
                        self.global_step += 1
                        self.epoch += 1
                        continue

                    wm_count = max(self.distributed.reduce_sum(len(train_wm_losses)), 1.0)
                    step_log["train_wm_loss"] = self.distributed.reduce_sum(sum(train_wm_losses)) / wm_count
                    step_log["train_wm_transition_loss"] = (
                        self.distributed.reduce_sum(sum(train_wm_transition_losses)) / wm_count
                    )
                    step_log["train_wm_kl_loss"] = self.distributed.reduce_sum(sum(train_wm_kl_losses)) / wm_count
                    if train_wm_reward_losses:
                        step_log["train_wm_reward_loss"] = (
                            self.distributed.reduce_sum(sum(train_wm_reward_losses)) / wm_count
                        )

                    # run validation
                    eval_every = int(OmegaConf.select(cfg, "eval.eval_every", default=1))
                    if val_dataloaders and (self.epoch % eval_every) == 0:
                        for split_name, val_dl in val_dataloaders.items():
                            step_log.update(self.evaluate_val_loss(val_dl, split_name))

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


__all__ = ["PretokenizeWMWorkspace"]


def _copy_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _copy_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_to_cpu(item) for item in value]
    return copy.deepcopy(value)
