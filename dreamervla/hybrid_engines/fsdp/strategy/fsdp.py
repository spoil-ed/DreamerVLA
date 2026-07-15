"""FSDP1 + no-shard strategies."""

from __future__ import annotations

from functools import partial

import torch
import torch.distributed as dist

from dreamervla.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase


class NoShardStrategy(FSDPStrategyBase):
    """``none``/``ddp``: no parameter sharding (DDP handled outside this helper)."""

    def fsdp_version(self) -> str:
        return "none"

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return self._apply_checkpointing(model)


class FSDPStrategy(FSDPStrategyBase):
    """Classic ``torch.distributed.fsdp.FullyShardedDataParallel`` wrapping."""

    def fsdp_version(self) -> str:
        return "fsdp1"

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        self._apply_checkpointing(model)
        self.ensure_process_group()
        if not (dist.is_available() and dist.is_initialized()):
            return model

        from torch.distributed.fsdp import (
            BackwardPrefetch,
            CPUOffload,
            FullyShardedDataParallel,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import _module_wrap_policy

        mixed_precision = None
        if self.param_dtype is not torch.float32:
            mixed_precision = MixedPrecision(
                param_dtype=self.param_dtype,
                reduce_dtype=self.param_dtype,
                buffer_dtype=self.param_dtype,
            )
        sharding = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
        }.get(self.sharding_strategy)
        if sharding is None:
            raise ValueError(
                "sharding_strategy must be full_shard, shard_grad_op, or no_shard; "
                f"got {self.sharding_strategy!r}"
            )
        backward_prefetch = {
            None: None,
            "none": None,
            "backward_pre": BackwardPrefetch.BACKWARD_PRE,
            "backward_post": BackwardPrefetch.BACKWARD_POST,
        }.get(self.backward_prefetch, ...)
        if backward_prefetch is ...:
            raise ValueError(
                "backward_prefetch must be backward_pre, backward_post, or none; "
                f"got {self.backward_prefetch!r}"
            )
        wrap_classes_getter = getattr(model, "fsdp_wrap_module_classes", None)
        wrap_classes = tuple(wrap_classes_getter()) if callable(wrap_classes_getter) else ()
        if not wrap_classes and self.require_layer_wrap:
            raise TypeError(
                "FSDP1 requires policy.fsdp_wrap_module_classes() so the model is "
                "sharded at transformer/vision block boundaries"
            )
        auto_wrap_policy = (
            partial(_module_wrap_policy, module_classes=set(wrap_classes)) if wrap_classes else None
        )
        first_parameter = next(model.parameters(), None)
        device_id = (
            first_parameter.device
            if first_parameter is not None and first_parameter.device.type == "cuda"
            else None
        )
        return FullyShardedDataParallel(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=device_id,
            sharding_strategy=sharding,
            cpu_offload=CPUOffload(offload_params=bool(self.cpu_offload)),
            mixed_precision=mixed_precision,
            forward_prefetch=self.forward_prefetch,
            backward_prefetch=backward_prefetch,
            limit_all_gathers=self.limit_all_gathers,
            use_orig_params=bool(self.use_orig_params),
            sync_module_states=bool(self.sync_module_states),
        )

    _FSDP_CACHE_ATTRS = (
        "_mp_shard",
        "_full_param_padded",
        "_full_prec_full_param_padded",
        "_unsharded_flat_param_for_skipped_views",
    )
    _FSDP_GRAD_ATTRS = ("_saved_grad_shard", "_cpu_grad")

    @staticmethod
    def _iter_fsdp_handles(model: torch.nn.Module) -> list[object]:
        handles: list[object] = []
        seen: set[int] = set()
        for module in model.modules():
            candidates = [getattr(module, "_handle", None)]
            candidates.extend(getattr(module, "_all_handles", None) or ())
            for handle in candidates:
                if handle is not None and id(handle) not in seen:
                    seen.add(id(handle))
                    handles.append(handle)
        return handles

    @staticmethod
    def _move_tensor(
        tensor: torch.Tensor | None,
        device: torch.device | str,
    ) -> torch.Tensor | None:
        if tensor is None or tensor.device == torch.device(device):
            return tensor
        return tensor.to(device, non_blocking=True)

    @staticmethod
    def _free_tensor_storage(tensor: torch.Tensor | None) -> None:
        if tensor is None:
            return
        try:
            storage = tensor.untyped_storage()
            if storage.size() > 0:
                storage.resize_(0)
        except Exception:
            pass

    @staticmethod
    def _rebind_handle_views(handle: object) -> None:
        if getattr(handle, "_use_orig_params", False):
            handle._use_sharded_views()
        elif getattr(handle, "uses_sharded_strategy", False):
            handle._use_sharded_views()
        else:
            handle._use_unsharded_views(as_params=False)

    @torch.no_grad()
    def _move_param_and_grad(
        self,
        model: torch.nn.Module,
        device: torch.device | str,
        *,
        move_grad: bool,
    ) -> None:
        for handle in self._iter_fsdp_handles(model):
            flat_param = handle.flat_param
            if hasattr(flat_param, "_local_shard"):
                flat_param._local_shard = self._move_tensor(flat_param._local_shard, device)
            flat_param.data = self._move_tensor(flat_param.data, device)
            if hasattr(flat_param, "_local_shard"):
                flat_param._local_shard = flat_param.data
            if move_grad:
                flat_param.grad = self._move_tensor(flat_param.grad, device)
                for name in self._FSDP_GRAD_ATTRS:
                    if hasattr(flat_param, name):
                        setattr(
                            flat_param, name, self._move_tensor(getattr(flat_param, name), device)
                        )
            for name in self._FSDP_CACHE_ATTRS:
                if hasattr(flat_param, name):
                    self._free_tensor_storage(getattr(flat_param, name))
            self._rebind_handle_views(handle)
        for parameter in model.parameters():
            parameter.data = self._move_tensor(parameter.data, device)
            if move_grad:
                parameter.grad = self._move_tensor(parameter.grad, device)
        for buffer in model.buffers():
            buffer.data = self._move_tensor(buffer.data, device)
        self.clear_memory()

    def offload_param_and_grad(
        self,
        model: torch.nn.Module,
        offload_grad: bool,
    ) -> None:
        self._move_param_and_grad(model, "cpu", move_grad=offload_grad)

    def onload_param_and_grad(
        self,
        model: torch.nn.Module,
        device: torch.device,
        onload_grad: bool,
    ) -> None:
        self._move_param_and_grad(model, device, move_grad=onload_grad)
