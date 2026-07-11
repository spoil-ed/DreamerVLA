from __future__ import annotations

import contextlib
import functools
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from dreamervla.utils.json_logger import JsonLogger


def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    """Return the trainable module behind DDP/FSDP, or the input unchanged."""

    return module.module if isinstance(module, (DDP, FSDP)) else module


class _NullJsonLogger:
    def __enter__(self) -> _NullJsonLogger:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    def log(self, data: dict[str, object]) -> None:
        return None


@dataclass
class NopretokenizeSFTDistributedHelper:
    rank: int
    local_rank: int
    world_size: int
    strategy: str
    fsdp_mixed_precision: str
    enable_activation_checkpointing: bool

    @classmethod
    def initialize(
        cls,
        strategy: str = "ddp",
        fsdp_mixed_precision: str = "bf16",
        enable_activation_checkpointing: bool = True,
        nccl_timeout_seconds: int | None = None,
    ) -> NopretokenizeSFTDistributedHelper:
        normalized_strategy = str(strategy).lower()
        if normalized_strategy not in {"ddp", "fsdp"}:
            raise ValueError(f"Unsupported distributed strategy: {strategy}")

        if dist.is_available() and not dist.is_initialized():
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if world_size > 1:
                if torch.cuda.is_available():
                    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
                init_kwargs: dict[str, Any] = {"backend": "nccl"}
                if nccl_timeout_seconds is not None:
                    init_kwargs["timeout"] = timedelta(seconds=int(nccl_timeout_seconds))
                dist.init_process_group(**init_kwargs)

        rank = (
            int(dist.get_rank()) if dist.is_available() and dist.is_initialized() else 0
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = (
            int(dist.get_world_size())
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            strategy=normalized_strategy,
            fsdp_mixed_precision=str(fsdp_mixed_precision).lower(),
            enable_activation_checkpointing=bool(enable_activation_checkpointing),
        )

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def uses_fsdp(self) -> bool:
        return self.is_distributed and self.strategy == "fsdp"

    @property
    def requires_collective_checkpointing(self) -> bool:
        return self.uses_fsdp

    def resolve_device(self, configured_device: str) -> torch.device:
        smallest_cuda_device = self._smallest_visible_cuda_device()
        if configured_device == "auto":
            if self.is_distributed and torch.cuda.is_available():
                return torch.device(f"cuda:{self.local_rank}")
            return (
                smallest_cuda_device
                if smallest_cuda_device is not None
                else torch.device("cpu")
            )
        if (
            configured_device.startswith("cuda")
            and self.is_distributed
            and torch.cuda.is_available()
        ):
            return torch.device(f"cuda:{self.local_rank}")
        if configured_device.startswith("cuda"):
            if smallest_cuda_device is not None:
                return smallest_cuda_device
            return torch.device("cpu")
        return torch.device(configured_device)

    def maybe_make_sampler(
        self, dataset: Any, shuffle: bool, drop_last: bool
    ) -> DistributedSampler | None:
        if not self.is_distributed:
            return None
        return DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def wrap_encoder(self, encoder: Any) -> None:
        if not self.is_distributed or encoder is None:
            return
        if self.uses_fsdp:
            encoder.backbone = self._wrap_with_fsdp(encoder.backbone)
            return
        encoder.backbone = self._wrap_module_with_ddp(encoder.backbone)

    def wrap_world_model(self, world_model: Any) -> None:
        if not self.is_distributed or world_model is None:
            return
        if self.uses_fsdp:
            return
        for name, child in world_model.named_children():
            if isinstance(child, torch.nn.Module) and any(
                p.requires_grad for p in child.parameters()
            ):
                setattr(world_model, name, self._wrap_module_with_ddp(child))

    def wrap_trainable_module(
        self,
        module: Any,
        *,
        find_unused_parameters: bool | None = None,
        broadcast_buffers: bool | None = None,
    ) -> Any:
        if not self.is_distributed or module is None:
            return module
        if self.uses_fsdp:
            return self._wrap_with_fsdp(module)
        if not any(p.requires_grad for p in module.parameters()):
            return module
        return self._wrap_module_with_ddp(
            module,
            find_unused_parameters=find_unused_parameters,
            broadcast_buffers=broadcast_buffers,
        )

    def unwrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
        return unwrap_module(module)

    def clip_grad_norm_tensor(
        self, module: torch.nn.Module, max_norm: float
    ) -> torch.Tensor:
        """Clip gradients without forcing a device-to-host synchronization."""

        if isinstance(module, FSDP):
            grad_norm = module.clip_grad_norm_(float(max_norm))
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                module.parameters(), float(max_norm)
            )
        if isinstance(grad_norm, torch.Tensor):
            return grad_norm
        return torch.tensor(float(grad_norm), device=self._reduce_device())

    def clip_grad_norm(self, module: torch.nn.Module, max_norm: float) -> float:
        grad_norm = self.clip_grad_norm_tensor(module, max_norm)
        return float(
            grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        )

    def logger_context(self, path: str) -> JsonLogger | _NullJsonLogger:
        return JsonLogger(path) if self.is_main_process else _NullJsonLogger()

    def reduce_mean(self, value: float | int | torch.Tensor) -> float:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(
                device=self._reduce_device(), dtype=torch.float32
            )
        else:
            tensor = torch.tensor(
                float(value), device=self._reduce_device(), dtype=torch.float32
            )
        if self.is_distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= float(self.world_size)
        return float(tensor.item())

    def reduce_sum(self, value: float | int | torch.Tensor) -> float:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(
                device=self._reduce_device(), dtype=torch.float32
            )
        else:
            tensor = torch.tensor(
                float(value), device=self._reduce_device(), dtype=torch.float32
            )
        if self.is_distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item())

    def reduce_mean_dict(
        self, metrics: dict[str, float | int | torch.Tensor]
    ) -> dict[str, float]:
        keys = list(metrics.keys())
        if not keys:
            return {}
        # Stack into one float32 tensor (matching per-key ``reduce_mean``'s
        # float32 round-trip) so a single all_reduce + single D2H replaces one
        # collective + ``.item()`` per key, while staying numerically identical.
        device = self._reduce_device()
        values = torch.stack(
            [
                value.detach().to(device=device, dtype=torch.float32).reshape(())
                if isinstance(value, torch.Tensor)
                else torch.tensor(float(value), device=device, dtype=torch.float32)
                for value in (metrics[key] for key in keys)
            ]
        )
        if self.is_distributed:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values /= float(self.world_size)
        return dict(zip(keys, values.detach().cpu().tolist(), strict=True))

    def broadcast_object(self, value: Any) -> Any:
        if not self.is_distributed:
            return value
        object_list = [value]
        dist.broadcast_object_list(object_list, src=0)
        return object_list[0]

    def model_state_dict_context(
        self,
        module: torch.nn.Module,
        rank0_only: bool = True,
    ) -> contextlib.AbstractContextManager[None]:
        """FSDP FULL_STATE_DICT context.

        - save (``rank0_only=True``, default): only rank 0 receives the gathered
          full dict; other ranks get empty dicts — efficient for writing.
        - load (``rank0_only=False``): every rank must provide the full dict,
          because all ranks read the ckpt file; FSDP scatters per-rank shards.
        """
        if not isinstance(module, FSDP):
            return contextlib.nullcontext()
        return FSDP.state_dict_type(
            module,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(
                offload_to_cpu=True, rank0_only=rank0_only
            ),
            optim_state_dict_config=FullOptimStateDictConfig(
                offload_to_cpu=True, rank0_only=rank0_only
            ),
        )

    def optimizer_state_dict(
        self, module: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> dict[str, Any]:
        if not isinstance(module, FSDP):
            return optimizer.state_dict()
        with self.model_state_dict_context(module):
            return FSDP.optim_state_dict(module, optimizer)

    def load_optimizer_state_dict(
        self,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        state_dict: dict[str, Any],
    ) -> None:
        if not isinstance(module, FSDP):
            optimizer.load_state_dict(state_dict)
            return
        with self.model_state_dict_context(module, rank0_only=False):
            converted_state_dict = FSDP.optim_state_dict_to_load(
                module, optimizer, state_dict
            )
        optimizer.load_state_dict(converted_state_dict)

    def _wrap_module_with_ddp(
        self,
        module: torch.nn.Module,
        *,
        find_unused_parameters: bool | None = None,
        broadcast_buffers: bool | None = None,
    ) -> DDP:
        return DDP(
            module,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            broadcast_buffers=(
                False if broadcast_buffers is None else bool(broadcast_buffers)
            ),
            find_unused_parameters=(
                False
                if find_unused_parameters is None
                else bool(find_unused_parameters)
            ),
        )

    def _wrap_with_fsdp(self, module: torch.nn.Module) -> FSDP:
        wrap_modules = []
        if hasattr(module, "get_fsdp_wrap_module_list"):
            wrap_modules = list(module.get_fsdp_wrap_module_list())

        checkpoint_modules = []
        if self.enable_activation_checkpointing and hasattr(
            module, "get_checkpointing_wrap_module_list"
        ):
            checkpoint_modules = list(module.get_checkpointing_wrap_module_list())

        fsdp_module = FSDP(
            module,
            auto_wrap_policy=functools.partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda submodule: submodule in wrap_modules,
            ),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=self._build_fsdp_mixed_precision(),
            device_id=torch.cuda.current_device(),
            sync_module_states=True,
            limit_all_gathers=True,
            use_orig_params=True,
        )

        if checkpoint_modules:
            non_reentrant_wrapper = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            apply_activation_checkpointing(
                fsdp_module,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=lambda submodule: submodule in checkpoint_modules,
            )

        torch.cuda.synchronize()
        return fsdp_module

    def _build_fsdp_mixed_precision(self) -> MixedPrecision:
        dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }.get(self.fsdp_mixed_precision)
        if dtype is None:
            raise ValueError(
                f"Unsupported FSDP mixed precision: {self.fsdp_mixed_precision}"
            )
        return MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        )

    def _reduce_device(self) -> torch.device:
        if self.is_distributed and torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")

    def _smallest_visible_cuda_device(self) -> torch.device | None:
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return None
        # Use the smallest logical CUDA index visible to the current process.
        return torch.device("cuda:0")

    def barrier(self) -> None:
        if dist.is_available() and dist.is_initialized():
            if torch.cuda.is_available():
                dist.barrier(device_ids=[self.local_rank])
            else:
                dist.barrier()

    def cleanup(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


__all__ = ["NopretokenizeSFTDistributedHelper"]
