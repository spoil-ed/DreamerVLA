from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

from dreamer_vla.models.chameleon_model.chameleon_vae_ori.image_tokenizer import ImageTokenizer
from dreamer_vla.models.chameleon_model.chameleon_vae_ori.vocab import (
    VocabInfo,
    VocabTranslation,
)
from dreamer_vla.models.encoder.rynnvla_image_ops import (
    generate_crop_size_list,
    var_center_crop,
)
from dreamer_vla.utils.conversation import Conversation

logger = logging.getLogger(__name__)


class RynnVLATokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.bos_id = self.tokenizer.bos_token_id
        if self.bos_id is None:
            self.bos_id = self.tokenizer.eos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self._probe_tokenizer_style()

    def encode(self, s: str, bos: bool, eos: bool) -> list[int]:
        tokens = self.tokenizer.encode(s, truncation=False, add_special_tokens=False)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def encode_segment(self, s: str):
        s = s.lstrip(" ")
        if self.need_space_before_segment:
            return self.encode(" " + s, bos=False, eos=False)
        return self.encode(s, bos=False, eos=False)

    def encode_wo_prefix_space(self, s: str):
        if self.need_space_before_segment:
            return self.encode(s, bos=False, eos=False)
        prefixes = ["@", "\n", "\\", "=", ">", "`"]
        for prefix in prefixes:
            prefix_tokens = self.encode(prefix, bos=False, eos=False)
            cat_tokens = self.encode(prefix + s, bos=False, eos=False)
            if cat_tokens[: len(prefix_tokens)] == prefix_tokens:
                return cat_tokens[len(prefix_tokens) :]
        raise NotImplementedError(
            f"Unable to tokenize segment without prefix space: {s!r}"
        )

    def _probe_tokenizer_style(self):
        sentence1 = self.encode("Hi my darling", bos=False, eos=False)
        sentence2 = self.encode("my darling", bos=False, eos=False)
        if sentence1[-len(sentence2) :] == sentence2:
            self.need_space_before_segment = False
        else:
            sentence3 = self.encode(" my darling", bos=False, eos=False)
            assert sentence1[-len(sentence3) :] == sentence3
            self.need_space_before_segment = True


class MMConvItemProcessor:
    def __init__(
        self,
        transform: dict[str, Callable[[Any], dict]],
        media_symbols: list[str],
        tokenizer_path: str,
        conv_template=Conversation,
    ):
        self.transform = transform
        self.media_symbols = media_symbols
        self.tokenizer = RynnVLATokenizer(tokenizer_path)
        self.tokenizer.tokenizer.add_tokens(media_symbols)
        self.d_media_symbol2token = {}
        self.d_media_token2symbol = {}
        for media_symbol in media_symbols:
            tokenized_symbol = self.tokenizer.encode(media_symbol, bos=False, eos=False)
            assert len(tokenized_symbol) == 1
            self.d_media_symbol2token[media_symbol] = tokenized_symbol[0]
            self.d_media_token2symbol[tokenized_symbol[0]] = media_symbol
        self.conv_template = conv_template
        self.implicit_at_beginning = False

    def collect_and_process_media(self, data_item):
        d_media = {}
        for media_symbol in self.media_symbols:
            key = media_symbol.lstrip("<|").rstrip("|>")
            l_media = data_item.get(media_symbol, data_item.get(key, []))
            if not isinstance(l_media, list):
                l_media = [l_media]
            d_media[media_symbol] = []
            for media in l_media:
                media = self.transform[media_symbol](media)
                media["type"] = media_symbol
                d_media[media_symbol].append(media)
        return d_media

    @staticmethod
    def insert_implicit_media_symbol_in_q1(conv_list, d_media):
        conv_list = copy.deepcopy(conv_list)
        for media_symbol, l_media in d_media.items():
            media_symbol_count = "".join(
                [entry["value"] for entry in conv_list if entry["value"] is not None]
            ).count(media_symbol)
            if media_symbol_count == 0:
                conv_list[0]["value"] = (media_symbol + " ") * len(l_media) + conv_list[
                    0
                ]["value"]
            else:
                assert media_symbol_count == len(l_media)
        return conv_list

    def add_speaker_and_signal(self, source):
        conv = self.conv_template()
        for i, sentence in enumerate(source):
            role = conv.roles[0] if i % 2 == 0 else conv.roles[1]
            conv.append_message(role, sentence["value"])
        processed = conv.process()
        return processed["conv"], processed["pieces"]

    def replace_media_token_with_media(self, tokens, labels, d_media):
        d_media_counter = {key: 0 for key in d_media}
        for i, token in enumerate(tokens):
            if token in self.d_media_token2symbol:
                media_symbol = self.d_media_token2symbol[token]
                media = d_media[media_symbol][d_media_counter[media_symbol]]
                d_media_counter[media_symbol] += 1
                tokens[i] = media
                media["to_predict"] = labels[i] > 0
        return tokens, labels

    def process_item(self, data_item: dict, training_mode=False):
        d_media = self.collect_and_process_media(data_item)
        source = self.insert_implicit_media_symbol_in_q1(
            data_item["conversations"], d_media
        )
        conversation, pieces = self.add_speaker_and_signal(source)
        tokens = self.tokenizer.encode(conversation, bos=True, eos=False)
        labels = [-100 for _ in tokens]

        check_pos = 0
        for i, piece in enumerate(pieces):
            tokenized_value = (
                self.tokenizer.encode(piece["data"], bos=(i == 0), eos=False)
                if i == 0
                else self.tokenizer.encode_wo_prefix_space(piece["data"])
            )
            assert (
                tokens[check_pos : check_pos + len(tokenized_value)] == tokenized_value
            )
            if piece["predict"]:
                labels[check_pos : check_pos + len(tokenized_value)] = tokenized_value
            check_pos += len(tokenized_value)

        tokens, labels = self.replace_media_token_with_media(tokens, labels, d_media)

        flattened_tokens = []
        flattened_labels = []
        for token_or_media, ori_label in zip(tokens, labels, strict=True):
            if isinstance(token_or_media, int):
                flattened_tokens.append(token_or_media)
                flattened_labels.append(ori_label)
            else:
                flattened_tokens += token_or_media["input_ids"]
                if ori_label <= 0:
                    flattened_labels += [-100] * len(token_or_media["input_ids"])
                else:
                    flattened_labels += token_or_media["labels"]
        return flattened_tokens, flattened_labels


class FlexARItemProcessorActionState(MMConvItemProcessor):
    image_start_token = "<racm3:break>"
    image_end_token = "<eoss>"
    new_line_token = "<reserved08799>"
    action_start_token = "<reserved10000>"
    action_end_token = "<reserved15000>"
    state_start_token = "<reserved15500>"
    state_end_token = "<reserved16000>"

    def __init__(
        self,
        tokenizer_path: str,
        text_tokenizer_path: str,
        vqgan_cfg_path: str,
        vqgan_ckpt_path: str,
        target_size: int = 256,
        device: str = "cuda",
    ):
        super().__init__(
            {
                "<|image|>": self.process_image,
                "<|action|>": self.process_action,
                "<|state|>": self.process_state,
            },
            ["<|image|>", "<|action|>", "<|state|>"],
            tokenizer_path=tokenizer_path,
            conv_template=Conversation,
        )
        self.patch_size = 32
        self.crop_size_list = generate_crop_size_list(
            (target_size // self.patch_size) ** 2, self.patch_size
        )
        self.device = device
        self.chameleon_ori_vocab = VocabInfo(
            json.load(open(text_tokenizer_path, encoding="utf8"))["model"]["vocab"]
        )
        self.chameleon_ori_translation = VocabTranslation(
            self.chameleon_ori_vocab, device=device
        )
        self.chameleon_ori_image_tokenizer = ImageTokenizer(
            cfg_path=vqgan_cfg_path,
            ckpt_path=vqgan_ckpt_path,
            device=device,
        )
        self.n_bins = 256
        self.bins = np.linspace(-1, 1, self.n_bins)

    def token2id(self, token: str) -> int:
        return self.tokenizer.tokenizer.vocab[token]

    @staticmethod
    def get_n_grids_token(n_grids):
        return f"<reserved{8800 + n_grids:05d}>"

    @torch.no_grad()
    def process_image(self, image) -> dict:
        if isinstance(image, Image.Image):
            pass
        elif isinstance(image, list):
            image = Image.fromarray(np.array(image).astype(np.uint8))
        else:
            raise TypeError(f"Unsupported image input: {type(image)!r}")
        image = var_center_crop(image, crop_size_list=self.crop_size_list)
        w_grids, h_grids = (
            image.size[0] // self.patch_size,
            image.size[1] // self.patch_size,
        )
        image_toks = self.chameleon_ori_translation.convert_img2bp2(
            self.chameleon_ori_image_tokenizer.img_tokens_from_pil(image)
        ).view(-1)
        full_image_toks = image_toks.reshape(image.size[1] // 16, image.size[0] // 16)
        new_line_id = self.token2id(self.new_line_token)
        full_image_toks = torch.cat(
            (
                full_image_toks,
                torch.ones(
                    image.size[1] // 16,
                    1,
                    device=full_image_toks.device,
                    dtype=full_image_toks.dtype,
                )
                * new_line_id,
            ),
            dim=1,
        ).flatten()
        result_toks = [
            self.token2id(self.image_start_token),
            self.token2id(self.get_n_grids_token(h_grids)),
            self.token2id(self.get_n_grids_token(w_grids)),
            *full_image_toks.tolist(),
            self.token2id(self.image_end_token),
        ]
        return {"input_ids": result_toks, "labels": result_toks}

    @torch.no_grad()
    def process_action(self, action) -> dict:
        action = np.asarray(action)
        norm_action = self.norm_action(action)
        discretized_action = (
            np.digitize(norm_action, self.bins)
            + self.token2id(self.action_start_token)
            + 1
        )
        result_toks = [
            self.token2id(self.action_start_token),
            *discretized_action.tolist(),
            self.token2id(self.action_end_token),
        ]
        return {"input_ids": result_toks, "labels": result_toks}

    @torch.no_grad()
    def process_state(self, state) -> dict:
        state = np.asarray(state)
        norm_state = self.norm_state(state)
        discretized_state = (
            np.digitize(norm_state, self.bins)
            + self.token2id(self.state_start_token)
            + 1
        )
        result_toks = [
            self.token2id(self.state_start_token),
            *discretized_state.tolist(),
            self.token2id(self.state_end_token),
        ]
        return {"input_ids": result_toks, "labels": result_toks}

    @staticmethod
    def norm_action(action):
        min_values = np.array(
            [-0.9375, -0.9375, -0.9375, -0.24214286, -0.375, -0.36428571, -1.0]
        )
        max_values = np.array([0.9375, 0.9375, 0.9375, 0.34821429, 0.375, 0.375, 1.0])
        norm_action = 2 * (action - min_values) / (max_values - min_values + 1e-8) - 1
        return np.clip(norm_action, a_min=-1, a_max=1)

    @staticmethod
    def norm_state(state):
        min_values = np.array(
            [
                -0.4827807,
                -0.3309336,
                0.00812818,
                1.00279467,
                -3.63125079,
                -1.84273835,
                -0.00545302,
                -0.04201502,
            ]
        )
        max_values = np.array(
            [
                2.10313803e-01,
                3.90426440e-01,
                1.47277813e00,
                3.72486417e00,
                3.56188956e00,
                1.38632160e00,
                4.23214189e-02,
                1.31260958e-03,
            ]
        )
        norm_state = 2 * (state - min_values) / (max_values - min_values + 1e-8) - 1
        return np.clip(norm_state, a_min=-1, a_max=1)


__all__ = ["Conversation", "FlexARItemProcessorActionState"]
