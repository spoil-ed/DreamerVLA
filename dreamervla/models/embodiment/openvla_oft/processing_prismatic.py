# Copyright 2025 The DreamerVLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from typing import ClassVar

import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor as PrismaticImageProcessorOriginal,
)
from prismatic.extern.hf.processing_prismatic import (
    PrismaticProcessor as PrismaticProcessorOriginal,
)
from transformers.image_processing_utils import BatchFeature
from transformers.tokenization_utils import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import TensorType


class PrismaticImageProcessor(PrismaticImageProcessorOriginal):
    def apply_transform(self, img: torch.Tensor) -> torch.Tensor:
        if self.tvf_do_letterbox:
            raise NotImplementedError("Letterbox padding is not yet supported!")

        imgs_t = []
        batch_size = img.shape[0]
        img = img.reshape(-1, *img.shape[2:])

        for idx in range(len(self.input_sizes)):
            img_idx = TVF.resize(img, **self.tvf_resize_params[idx])
            img_idx = TVF.center_crop(img_idx, **self.tvf_crop_params[idx])
            if isinstance(img_idx, Image.Image):
                img_idx = TVF.to_tensor(img_idx)
            img_idx = img_idx / 255.0
            img_idx = TVF.normalize(img_idx, **self.tvf_normalize_params[idx])
            imgs_t.append(img_idx)

        img_t = torch.cat(imgs_t, dim=1)
        return img_t.reshape(batch_size, -1, *img_t.shape[1:])

    def preprocess(
        self,
        images: torch.Tensor,
        return_tensors: str | TensorType | None = None,
        **_: str,
    ) -> BatchFeature:
        return BatchFeature(
            data={"pixel_values": self.apply_transform(images)},
            tensor_type=return_tensors,
        )

    def __call__(self, images: torch.Tensor, **kwargs) -> BatchFeature:
        return self.preprocess(images, **kwargs)


class PrismaticProcessor(PrismaticProcessorOriginal):
    attributes: ClassVar[list[str]] = ["image_processor", "tokenizer"]
    image_processor_class: str = "AutoImageProcessor"
    tokenizer_class: str = "AutoTokenizer"

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        images: torch.Tensor,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy | None = None,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = TensorType.PYTORCH,
    ) -> BatchFeature:
        assert self.tokenizer.padding_side == "left", (
            "Required: Init tokenizer with padding_side='left'"
        )
        pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
        text_inputs = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        self._normalize_left_padded_bos(text_inputs)
        if pixel_values.shape[0] != text_inputs.input_ids.shape[0]:
            raise ValueError("Batch is malformed; expected same number of images and text inputs!")
        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})

    def _normalize_left_padded_bos(self, text_inputs) -> None:
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]
        first_nonzero_indices = torch.argmax(attention_mask, dim=1).unsqueeze(1)
        assert torch.all(input_ids.gather(1, first_nonzero_indices) == self.tokenizer.bos_token_id)
        assert torch.all(input_ids[:, -1] != self.tokenizer.pad_token_id)
        input_ids.scatter_(1, first_nonzero_indices, self.tokenizer.pad_token_id)
        attention_mask.scatter_(1, first_nonzero_indices, 0)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        attention_mask[:, 0] = 1


class MultiInputPrismaticProcessor(PrismaticProcessor):
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        images: dict[str, torch.Tensor],
        proprio_states: torch.Tensor,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy | None = None,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = TensorType.PYTORCH,
    ) -> BatchFeature:
        all_pixel_values = [
            self.image_processor(image, return_tensors=return_tensors)["pixel_values"]
            for image in images.values()
        ]
        input_pixel_values = torch.cat(all_pixel_values, dim=1)
        text_inputs = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        self._normalize_left_padded_bos(text_inputs)
        if input_pixel_values.shape[0] != text_inputs.input_ids.shape[0]:
            raise ValueError("Batch is malformed; expected same number of images and text inputs!")
        return BatchFeature(data={**text_inputs, "pixel_values": input_pixel_values})
