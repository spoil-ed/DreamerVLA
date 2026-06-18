"""OpenVLA-OFT rollout bundle for the generic RolloutInferenceWorker."""

from __future__ import annotations

from typing import Any


class OFTRolloutBundle:
    """Wrap OFT batched decoding and per-env hidden extraction."""

    def __init__(
        self,
        policy_cfg: dict[str, Any],
        unnorm_key: str,
        image_keys: list[str],
        history: int,
        rotate_images_180: bool = True,
        center_crop: bool = True,
        device: str = "cuda",
    ) -> None:
        from dreamervla.runners import oft_collect_common
        from dreamervla.runners import rollout_hidden_extractor as rhe

        gpu = 0 if str(device).startswith("cuda") else -1
        cfg = dict(policy_cfg)
        cfg.setdefault("unnorm_key", str(unnorm_key))
        cfg.setdefault("_rank", 0)
        self._policy = oft_collect_common.load_policy(cfg, gpu)
        self._decoder = rhe.OFTBatchedDecoder(self._policy, str(unnorm_key))
        self._unnorm_key = str(unnorm_key)
        self._image_keys = list(image_keys)
        self._history = int(history)
        self._rotate = bool(rotate_images_180)
        self._center_crop = bool(center_crop)

    def predict_batch(self, preps: list[dict[str, Any]]) -> Any:
        return self._decoder.predict_batch(preps)

    def make_extractor(self) -> Any:
        from dreamervla.runners import rollout_hidden_extractor as rhe

        return rhe.OFTRolloutHiddenExtractor(
            self._policy,
            image_keys=self._image_keys,
            history=self._history,
            rotate_images_180=self._rotate,
            center_crop=self._center_crop,
            unnorm_key=self._unnorm_key,
        )
