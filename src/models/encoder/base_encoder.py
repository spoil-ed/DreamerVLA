import torch

from .protocol import EncoderInputBatch, build_encoder_input_batch


class BaseEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def encode(self, obs: dict[str, object]) -> torch.Tensor:
        raise NotImplementedError

    def prepare_inputs(
        self,
        obs: dict[str, object],
        *,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        meta: list[dict[str, object]] | None = None,
    ) -> EncoderInputBatch:
        return build_encoder_input_batch(obs, action=action, action_mask=action_mask, meta=meta)
