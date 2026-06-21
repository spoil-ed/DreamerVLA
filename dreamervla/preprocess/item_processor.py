# ruff: noqa: E402
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from dreamervla.constants import DEFAULT_ACTION_TOKEN_ID
from dreamervla.models.embodiment.chameleon_model import chameleon_vae_ori
from dreamervla.models.encoder.rynnvla_image_ops import (
    generate_crop_size_list,
    var_center_crop,
)
from dreamervla.preprocess.conversation import Conversation
from dreamervla.preprocess.paths import DEFAULT_CHAMELEON_TOKENIZER_DIR, DEFAULT_TOKENIZER_PATH
from dreamervla.preprocess.xllmx.data.data_reader import read_general
from dreamervla.preprocess.xllmx.data.item_processor import MMConvItemProcessor

logger = logging.getLogger(__name__)


class _FlexARItemProcessorBase(MMConvItemProcessor):
    """Shared helpers identical across every FlexAR item processor variant.

    Holds only the methods whose bodies are byte-identical across all four
    concrete processors (``get_n_grids_token``, ``token2id``, ``process_item``).
    Variant-specific pieces (``__init__``, ``process_image``, ``decode_image``,
    action/state handling and per-class norm constants) stay on each subclass.
    """

    @staticmethod
    def get_n_grids_token(n_grids):
        return f"<reserved{8800 + n_grids:05d}>"

    def token2id(self, token: str) -> int:
        return self.tokenizer.tokenizer.vocab[token]

    def process_item(self, item, training_mode=False, out_flatten=True):
        if not out_flatten:
            return super().process_item(item, training_mode=training_mode)

        if training_mode:
            tokens, labels = super().process_item(item, training_mode=training_mode)
            input_tokens_item = []
            modified_labels_item = []
            for token_or_media, ori_label in zip(tokens, labels, strict=True):
                if isinstance(token_or_media, int):
                    token = token_or_media
                    input_tokens_item.append(token)
                    modified_labels_item.append(ori_label)
                else:
                    input_tokens_item += token_or_media["input_ids"]
                    if ori_label <= 0:  # in the prompt part
                        modified_labels_item += [-100] * len(
                            token_or_media["input_ids"]
                        )
                    else:
                        modified_labels_item += token_or_media["labels"]

            return input_tokens_item, modified_labels_item
        else:
            tokens = super().process_item(item, training_mode=training_mode)
            input_tokens_item = []
            for token_or_media in tokens:
                if isinstance(token_or_media, int):
                    input_tokens_item.append(token_or_media)
                else:
                    input_tokens_item += token_or_media["input_ids"]

            return input_tokens_item


class FlexARItemProcessorAction(_FlexARItemProcessorBase):
    image_start_token = (
        "<racm3:break>"  # fixed tokens for start and end, so can hardcode
    )
    image_end_token = "<eoss>"
    full_sub_sep_token = "<reserved08796>"
    sub_sub_sep_token = "<reserved08797>"
    sub_skip_token = "<reserved08798>"
    new_line_token = "<reserved08799>"

    action_start_token = "<reserved10000>"
    action_end_token = "<reserved15000>"

    def __init__(
        self,
        tokenizer=str(DEFAULT_TOKENIZER_PATH),
        conv_template=Conversation,
        target_size=512,
        device="cuda",
    ):
        super().__init__(
            {
                "<|image|>": self.process_image,
                "<|action|>": self.process_action,
            },
            ["<|image|>", "<|action|>"],
            tokenizer,
            conv_template,
        )

        self.patch_size = 32
        self.crop_size_list = generate_crop_size_list(
            (target_size // self.patch_size) ** 2, self.patch_size
        )
        logger.info("List of crop sizes:")
        for i in range(0, len(self.crop_size_list), 6):
            logger.info(
                " "
                + "".join(
                    [f"{f'{w} x {h}':14s}" for w, h in self.crop_size_list[i : i + 6]]
                )
            )

        #  todo
        #  currently still use the original image tokenizer provided by Meta rather than transformers
        #  because the transformers implementation does not contain the vae decoder
        self.chameleon_ori_vocab = chameleon_vae_ori.VocabInfo(
            json.load(
                open(
                    DEFAULT_CHAMELEON_TOKENIZER_DIR / "text_tokenizer.json",
                    encoding="utf8",
                )
            )["model"]["vocab"]
        )
        self.chameleon_ori_translation = chameleon_vae_ori.VocabTranslation(
            self.chameleon_ori_vocab, device=device
        )
        self.chameleon_ori_image_tokenizer = chameleon_vae_ori.ImageTokenizer(
            cfg_path=str(DEFAULT_CHAMELEON_TOKENIZER_DIR / "vqgan.yaml"),
            ckpt_path=str(DEFAULT_CHAMELEON_TOKENIZER_DIR / "vqgan.ckpt"),
            device=device,
        )

        self.n_bins, self.min_action, self.max_action = 256, -1, 1

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(self.min_action, self.max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

    @torch.no_grad()
    def process_image(self, image) -> dict:
        if isinstance(image, Image.Image):
            pass
        elif isinstance(image, list):
            image = Image.fromarray(np.array(image).astype(np.uint8))
        else:
            image = Image.open(read_general(image))
            # new_size = (320, 224)
            # image = image.resize(new_size)

        # import pdb; pdb.set_trace()

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
        if isinstance(action, str):
            action = np.load(action)
        action = np.array(action)
        # action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
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
        # print(action, norm_action, discretized_action, result_toks)
        # import pdb; pdb.set_trace()
        return {"input_ids": result_toks, "labels": result_toks}

    def decode_token_ids_to_actions(self, dis_action):
        bins = np.linspace(-1, 1, 256)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        discretized_actions = dis_action - 1 - DEFAULT_ACTION_TOKEN_ID
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=bin_centers.shape[0] - 1
        )
        return bin_centers[discretized_actions]

    def norm_action(self, action):
        min_values = np.array(
            [-0.9375, -0.9375, -0.9375, -0.32571429, -0.375, -0.375, -1.0]
        )
        max_values = np.array([0.9375, 0.9375, 0.9375, 0.375, 0.375, 0.375, 1.0])
        # min_values = np.array([-0.06998102, -0.0699713, 0, 0])
        # max_values = np.array([0.06991026, 0.06998004, 4, 4])
        norm_action = 2 * (action - min_values) / (max_values - min_values + 1e-8) - 1
        norm_action = np.clip(norm_action, a_min=-1, a_max=1)

        return norm_action

    def decode_image(self, tokens: list[int]) -> Image.Image:
        # print('0', tokens, len(tokens))
        if tokens[0] == self.token2id(self.image_start_token):
            tokens = tokens[1:]
        if tokens[-1] == self.token2id(self.image_end_token):
            tokens = tokens[:-1]
        # print('1', tokens, len(tokens))

        h_grids, w_grids = tokens[0] - 8804, tokens[1] - 8804
        tokens = tokens[2:]
        # print('2', tokens, len(tokens))
        h_latent_dim, w_latent_dim = h_grids * 2, w_grids * 2

        for i in range(len(tokens)):
            if (i + 1) % (w_latent_dim + 1) != 0:
                tokens[i] = self.chameleon_ori_translation.bpe2img[tokens[i]]

        assert len(tokens) == h_latent_dim * (w_latent_dim + 1)
        tokens = torch.tensor(tokens, dtype=torch.int64).cuda()

        tokens = tokens.view(h_latent_dim, w_latent_dim + 1)[:, :-1].flatten()

        return self.chameleon_ori_image_tokenizer.pil_from_img_toks(
            tokens, h_latent_dim, w_latent_dim
        )


class FlexARItemProcessorActionState(_FlexARItemProcessorBase):
    image_start_token = (
        "<racm3:break>"  # fixed tokens for start and end, so can hardcode
    )
    image_end_token = "<eoss>"
    full_sub_sep_token = "<reserved08796>"
    sub_sub_sep_token = "<reserved08797>"
    sub_skip_token = "<reserved08798>"
    new_line_token = "<reserved08799>"

    action_start_token = "<reserved10000>"
    action_end_token = "<reserved15000>"

    state_start_token = "<reserved15500>"
    state_end_token = "<reserved16000>"

    def __init__(
        self,
        tokenizer=str(DEFAULT_TOKENIZER_PATH),
        conv_template=Conversation,
        target_size=512,
        device="cuda",
    ):
        super().__init__(
            {
                "<|image|>": self.process_image,
                "<|action|>": self.process_action,
                "<|state|>": self.process_state,
            },
            ["<|image|>", "<|action|>", "<|state|>"],
            tokenizer,
            conv_template,
        )

        self.patch_size = 32
        self.crop_size_list = generate_crop_size_list(
            (target_size // self.patch_size) ** 2, self.patch_size
        )
        logger.info("List of crop sizes:")
        for i in range(0, len(self.crop_size_list), 6):
            logger.info(
                " "
                + "".join(
                    [f"{f'{w} x {h}':14s}" for w, h in self.crop_size_list[i : i + 6]]
                )
            )

        #  todo
        #  currently still use the original image tokenizer provided by Meta rather than transformers
        #  because the transformers implementation does not contain the vae decoder
        self.chameleon_ori_vocab = chameleon_vae_ori.VocabInfo(
            json.load(
                open(
                    DEFAULT_CHAMELEON_TOKENIZER_DIR / "text_tokenizer.json",
                    encoding="utf8",
                )
            )["model"]["vocab"]
        )
        self.chameleon_ori_translation = chameleon_vae_ori.VocabTranslation(
            self.chameleon_ori_vocab, device=device
        )
        self.chameleon_ori_image_tokenizer = chameleon_vae_ori.ImageTokenizer(
            cfg_path=str(DEFAULT_CHAMELEON_TOKENIZER_DIR / "vqgan.yaml"),
            ckpt_path=str(DEFAULT_CHAMELEON_TOKENIZER_DIR / "vqgan.ckpt"),
            device=device,
        )

        self.n_bins, self.min_action, self.max_action = 256, -1, 1

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(self.min_action, self.max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.device = device

    @torch.no_grad()
    def process_image(self, image) -> dict:
        # print('1: ', image.shape, type(image))
        # print(np.array(image).astype(np.uint8))
        # image = Image.fromarray(np.array(image).astype(np.uint8))
        # print(image.size, image)
        # new_size = (512, 512)
        # new_size = (256, 256)
        # image = image.resize(new_size)
        # print('2: ', image.size)
        # print(image)

        if isinstance(image, Image.Image):
            pass
        elif isinstance(image, list):
            image = Image.fromarray(np.array(image).astype(np.uint8))
        else:
            image = Image.open(read_general(image))
            new_size = (256, 256)
            # new_size = (512, 512)
            image = image.resize(new_size)

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
        if isinstance(action, str):
            action = np.load(action)
        action = np.array(action)
        # action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
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
        # print(action, norm_action, discretized_action, result_toks)
        # import pdb; pdb.set_trace()
        return {"input_ids": result_toks, "labels": result_toks}

    @torch.no_grad()
    def process_state(self, state) -> dict:
        if isinstance(state, str):
            state = np.load(state)
        state = np.array(state)
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
        # print(state, norm_state, discretized_state, result_toks)
        # import pdb; pdb.set_trace()
        return {"input_ids": result_toks, "labels": result_toks}

    def norm_action(self, action):
        # spatial, object, goal, 10   no_ops
        min_values = np.array(
            [-0.9375, -0.9375, -0.9375, -0.24214286, -0.375, -0.36428571, -1.0]
        )
        max_values = np.array([0.9375, 0.9375, 0.9375, 0.34821429, 0.375, 0.375, 1.0])

        norm_action = 2 * (action - min_values) / (max_values - min_values + 1e-8) - 1
        norm_action = np.clip(norm_action, a_min=-1, a_max=1)

        return norm_action

    def norm_state(self, state):
        # spatial, object, goal, 10   no_ops
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
        norm_state = np.clip(norm_state, a_min=-1, a_max=1)

        return norm_state

    def decode_image(self, tokens: list[int]) -> Image.Image:
        if tokens[0] == self.token2id(self.image_start_token):
            tokens = tokens[1:]
        if tokens[-1] == self.token2id(self.image_end_token):
            tokens = tokens[:-1]

        h_grids, w_grids = tokens[0] - 8804, tokens[1] - 8804
        tokens = tokens[2:]
        h_latent_dim, w_latent_dim = h_grids * 2, w_grids * 2

        for i in range(len(tokens)):
            if (i + 1) % (w_latent_dim + 1) != 0:
                tokens[i] = self.chameleon_ori_translation.bpe2img[tokens[i]]

        assert len(tokens) == h_latent_dim * (w_latent_dim + 1)
        tokens = torch.tensor(tokens, dtype=torch.int64).to(self.device)

        tokens = tokens.view(h_latent_dim, w_latent_dim + 1)[:, :-1].flatten()

        return self.chameleon_ori_image_tokenizer.pil_from_img_toks(
            tokens, h_latent_dim, w_latent_dim
        )


FlexARItemProcessor_Action = FlexARItemProcessorAction
FlexARItemProcessor_Action_State = FlexARItemProcessorActionState


__all__ = [
    "FlexARItemProcessorAction",
    "FlexARItemProcessorActionState",
    "FlexARItemProcessor_Action",
    "FlexARItemProcessor_Action_State",
]
