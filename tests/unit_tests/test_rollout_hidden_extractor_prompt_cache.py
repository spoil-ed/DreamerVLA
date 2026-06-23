"""Prompt-tokenization cache for OFTRolloutHiddenExtractor.

The prompt is invariant for a fixed ``task_description`` while the image changes
every rollout step.  ``processor(prompt, img)`` does BOTH text tokenization
(invariant) and image preprocessing (per-step).  The extractor must:

  * tokenize the prompt at most ONCE per ``task_description`` (cache hit reuses it),
  * re-run the image branch every step,
  * return ``input_ids`` / ``attention_mask`` / ``pixel_values`` numerically
    IDENTICAL (atol=0) to calling ``processor(prompt, img)`` directly.

``prepare`` itself needs the vendored OpenVLA-OFT tree (for image preprocessing),
so these tests exercise the factored, model-free pieces directly:
``_prompt_text_inputs`` (the cache) and ``_view_pixel_values`` (the image branch).
A fake processor (no model / no GPU) whose tokenizer counts calls and whose
tensors are deterministic functions of their inputs makes the contract verifiable
in isolation.
"""

from __future__ import annotations

import numpy as np
import torch

from dreamervla.runners.rollout_hidden_extractor import OFTRolloutHiddenExtractor


# ── fake processor mirroring PrismaticProcessor.__call__ structure ───────────
#   __call__(text, images) returns a dict with:
#     pixel_values = image_processor(images)["pixel_values"]   (image-only)
#     input_ids/attention_mask = tokenizer(text)               (text-only)
class _FakeImageProcessor:
    def __call__(self, images, return_tensors=None):
        # Deterministic, image-dependent; small integers so a bf16 cast is exact.
        arr = np.asarray(images, dtype=np.float32)
        val = float(int(arr.sum()) % 7)
        return {"pixel_values": torch.full((1, 6, 4, 4), val, dtype=torch.float32)}


class _FakeTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text, **_):
        self.calls += 1
        # Deterministic, text-dependent token ids (length grows with the prompt).
        ids = torch.tensor([[1] + [ord(c) % 97 for c in text]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class _FakeProcessor:
    def __init__(self) -> None:
        self.image_processor = _FakeImageProcessor()
        self.tokenizer = _FakeTokenizer()

    def __call__(self, text, images, **_):
        pixel_values = self.image_processor(images)["pixel_values"]
        text_inputs = self.tokenizer(text)
        return {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "pixel_values": pixel_values,
        }


class _StubPolicy:
    use_proprio = False

    def __init__(self, processor) -> None:
        self.processor = processor
        self.vla = torch.nn.Linear(1, 1)  # only needs .parameters() for a device


def _make_extractor(processor):
    return OFTRolloutHiddenExtractor(
        _StubPolicy(processor),
        image_keys=["agentview_rgb", "eye_in_hand_rgb"],
        history=2,
        rotate_images_180=False,
        center_crop=False,
        unnorm_key="dummy",
    )


def _img(seed: int) -> np.ndarray:
    return np.random.RandomState(seed).randint(0, 256, (8, 8, 3), dtype=np.uint8)


def _prompt(task: str) -> str:
    return f"In: What action should the robot take to {task.lower()}?\nOut:"


# ── cache contract ───────────────────────────────────────────────────────────


def test_prompt_tokenized_once_per_task() -> None:
    """N tokenize lookups with the SAME task_description tokenize the prompt ONCE."""
    proc = _FakeProcessor()
    ext = _make_extractor(proc)
    ext.reset()
    task = "open the middle drawer of the cabinet"

    for i in range(5):
        ext._prompt_text_inputs(proc, _prompt(task), task, _img(i))

    assert proc.tokenizer.calls == 1, (
        f"prompt must be tokenized once per task, got {proc.tokenizer.calls}"
    )


def test_cached_text_identical_to_direct_call() -> None:
    """Cached input_ids/attention_mask == a direct processor(prompt, img) call (atol=0)."""
    proc = _FakeProcessor()
    ext = _make_extractor(proc)
    ext.reset()
    task = "put the bowl on the plate"
    prompt = _prompt(task)

    ext._prompt_text_inputs(proc, prompt, task, _img(0))  # warm the cache
    ids, mask = ext._prompt_text_inputs(proc, prompt, task, _img(1))  # hit

    # Reference: the text branch of a direct call (a fresh, uncached processor).
    ref = _FakeProcessor()(prompt, _img(99))
    torch.testing.assert_close(ids.cpu(), ref["input_ids"], atol=0, rtol=0)
    torch.testing.assert_close(mask.cpu(), ref["attention_mask"], atol=0, rtol=0)


def test_view_pixel_values_identical_to_direct_call() -> None:
    """Per-view image branch == processor(prompt, img)["pixel_values"] (atol=0)."""
    proc = _FakeProcessor()
    ext = _make_extractor(proc)
    prompt = _prompt("anything")
    for seed in (0, 1, 2):
        img = _img(seed)
        got = ext._view_pixel_values(proc, img)
        ref = proc(prompt, img)["pixel_values"]
        torch.testing.assert_close(got.cpu(), ref, atol=0, rtol=0)


def test_cache_refreshes_on_task_change() -> None:
    """A new task_description re-tokenizes and yields different text tensors."""
    proc = _FakeProcessor()
    ext = _make_extractor(proc)
    ext.reset()

    ids_a, _ = ext._prompt_text_inputs(
        proc, _prompt("pick up the bowl"), "pick up the bowl", _img(0)
    )
    assert proc.tokenizer.calls == 1
    ids_b, _ = ext._prompt_text_inputs(
        proc, _prompt("open the top drawer"), "open the top drawer", _img(0)
    )
    assert proc.tokenizer.calls == 2, "different task must re-tokenize"

    assert ids_a.shape != ids_b.shape or not torch.equal(
        ids_a.cpu(), ids_b.cpu()
    ), "different task must produce different input_ids"
