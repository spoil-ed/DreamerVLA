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
        obs_hidden_source: str = "hidden_token",
        expected_action_head_type: str | None = None,
        expected_include_state: bool | None = None,
        device: str = "cuda",
    ) -> None:
        from dreamervla.runtime import oft_collect as oft_collect_common
        from dreamervla.runtime import rollout_hidden_extractor as rhe

        device_ref = _device_ref_from_device(device)
        cfg = dict(policy_cfg)
        cfg.setdefault("unnorm_key", str(unnorm_key))
        cfg.setdefault("_rank", 0)
        selected_image_keys = _select_image_keys_for_policy(
            image_keys,
            cfg.get("num_images_in_input", 1),
        )
        if int(history) != 1:
            raise ValueError("OpenVLA-OFT hidden-token mainline requires history=1")
        if str(obs_hidden_source) != "hidden_token":
            raise ValueError(
                "OpenVLA-OFT rollout observations require "
                "obs_hidden_source='hidden_token'"
            )
        self._policy = oft_collect_common.load_policy(cfg, device_ref)
        if expected_action_head_type is not None:
            cfg["expected_action_head_type"] = str(expected_action_head_type)
        if expected_include_state is not None:
            cfg["expected_include_state"] = bool(expected_include_state)
        if "expected_action_head_type" in cfg and "expected_include_state" in cfg:
            oft_collect_common.assert_policy_mode_matches(cfg)
        self._device = str(device)
        self._decoder = rhe.OFTBatchedDecoder(
            self._policy,
            str(unnorm_key),
            obs_hidden_source=obs_hidden_source,
            image_keys=selected_image_keys,
        )
        self._unnorm_key = str(unnorm_key)
        self._image_keys = selected_image_keys
        self._history = int(history)
        self._rotate = bool(rotate_images_180)
        self._center_crop = bool(center_crop)
        self._obs_hidden_source = str(obs_hidden_source)

    def predict_batch(self, preps: list[dict[str, Any]]) -> Any:
        return self._decoder.predict_batch(preps)

    def to(self, device: str) -> OFTRolloutBundle:
        self._device = str(device)
        if hasattr(self._policy, "to"):
            self._policy.to(device)
        return self

    def make_extractor(self) -> Any:
        from dreamervla.runtime import rollout_hidden_extractor as rhe

        return rhe.OFTRolloutHiddenExtractor(
            self._policy,
            image_keys=self._image_keys,
            history=self._history,
            rotate_images_180=self._rotate,
            center_crop=self._center_crop,
            unnorm_key=self._unnorm_key,
            obs_hidden_source=self._obs_hidden_source,
        )


def _device_ref_from_device(device: str) -> int | str:
    value = str(device).strip()
    if not value.startswith("cuda"):
        return value or "cpu"
    if ":" not in value:
        return 0
    return int(value.split(":", 1)[1])


def _select_image_keys_for_policy(
    image_keys: list[str],
    num_images_in_input: object,
) -> list[str]:
    """Return the only camera key admitted by the one-image mainline."""

    keys = [str(key) for key in image_keys]
    count = int(num_images_in_input)
    if count != 1:
        raise ValueError(
            "OpenVLA-OFT hidden-token mainline requires num_images_in_input=1, "
            f"got {count}"
        )
    if keys != ["agentview_rgb"]:
        raise ValueError(
            "OpenVLA-OFT hidden-token mainline requires exactly one image key "
            f"'agentview_rgb', got {keys!r}"
        )
    return keys
