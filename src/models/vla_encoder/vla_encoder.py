import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from abc import ABC, abstractmethod
from PIL import Image
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRANSFORMERS_SRC = PROJECT_ROOT.parent / "WMPO-main" / "transformers-openvla-oft" / "src"
if str(TRANSFORMERS_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSFORMERS_SRC))

from models.chameleon_model import chameleon_vae_ori
from models.chameleon_model.chameleon import RynnVLAForActionPrediction

def _convert_to_rgb(image):
    return image.convert('RGB')

class BaseEncoder(nn.Module, ABC):
    @abstractmethod
    def encode(self, obs, text):
        pass

class RynnVLAEncoder(BaseEncoder):
    def __init__(self,
                 pretrained_policy_ckpt,
                 precision,
                 device,
                 configs,
                 condition_frame_num):
        super().__init__()

        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        self.model = RynnVLAForActionPrediction.from_pretrained(
            pretrained_policy_ckpt,
            torch_dtype=self.dtype,
            device_map=device,
        )

        self.patch_size = 32

        self.condition_frame_num = condition_frame_num

        # define tokenizer
        self.image_start_token = "<racm3:break>"  # fixed tokens for start and end, so can hardcode
        self.image_end_token = "<eoss>"
        self.full_sub_sep_token = "<reserved08796>"
        self.sub_sub_sep_token = "<reserved08797>"
        self.sub_skip_token = "<reserved08798>"
        self.new_line_token = "<reserved08799>"

        self.action_token_id = 65536
        self.wrist_start_token_id = 65537
        self.wrist_end_token_id = 65538
        self.state_token_id = 65539

        self.chameleon_ori_vocab = chameleon_vae_ori.VocabInfo(
            json.load(
                open(PROJECT_ROOT / "pretrained_models" / "Chameleon" / "original_tokenizers" / "text_tokenizer.json", encoding="utf8")
            )["model"]["vocab"]
        )
        self.chameleon_ori_translation = chameleon_vae_ori.VocabTranslation(self.chameleon_ori_vocab, device=self.model.model.vqmodel.device)
        self.chameleon_ori_image_tokenizer = chameleon_vae_ori.ImageTokenizer(
            cfg_path=str(PROJECT_ROOT / "pretrained_models" / "Chameleon" / "original_tokenizers" / "vqgan.yaml"),
            ckpt_path=str(PROJECT_ROOT / "pretrained_models" / "Chameleon" / "original_tokenizers" / "vqgan.ckpt"),
            device=self.model.model.vqmodel.device,
        )

        self.text_tokenizer = AutoTokenizer.from_pretrained(
            str(PROJECT_ROOT / "pretrained_models" / "Chameleon"),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.text_tokenizer.add_tokens(["<|image|>"])
        self.text_vocab = self.text_tokenizer.get_vocab()
        self.text_bos_id = self.text_tokenizer.bos_token_id
        if self.text_bos_id is None:
            self.text_bos_id = self.text_tokenizer.eos_token_id
        self.text_eos_id = self.text_tokenizer.eos_token_id

        self.action_tokenizer = None
        actionvae_config = configs.get('actionvae_config')
        actionvae_pretrained_path = configs.get('actionvae_pretrained_path')
        if actionvae_config is not None:
            from models.actionvae_model import ActionVAE

            self.action_tokenizer = ActionVAE(actionvae_config)
            if actionvae_pretrained_path:
                ckpt_path = Path(actionvae_pretrained_path)
                if not ckpt_path.is_absolute():
                    ckpt_path = PROJECT_ROOT / ckpt_path
                if ckpt_path.exists():
                    checkpoint = torch.load(ckpt_path, map_location="cpu")
                    self.action_tokenizer.load_state_dict(checkpoint, strict=True)
            self.action_tokenizer = self.action_tokenizer.to(self.model.device)
            self.action_tokenizer.eval()


        self.image_size = int(configs['train_dataset']['img_size'])

        try:
            self.scale_p = configs['train_dataset']['scale_p']
        except:
            pass

        self.use_rel_action = configs['train_dataset']['use_rel_action']

        self.min_max_norm = configs['train_dataset']['min_max_norm']
        self.mean_std_norm = configs['train_dataset']['mean_std_norm']

        # action normalization params
        data_path = configs['train_dataset'].get('data_path')
        action_stats = None
        if data_path:
            stats_path = Path(data_path)
            if not stats_path.is_absolute():
                stats_path = PROJECT_ROOT / stats_path
            if stats_path.exists():
                with open(stats_path, "r") as f:
                    action_stats = json.load(f)
        if action_stats is None:
            state_dim = int(getattr(self.model.config, "state_dim", configs.get("state_dim", 6)))
            action_dim = int(configs.get('action_dim', 6))
            action_stats = {
                'rel_min_action': [0.0] * action_dim,
                'rel_max_action': [1.0] * action_dim,
                'rel_mean_action': [0.0] * action_dim,
                'rel_std_action': [1.0] * action_dim,
                'min_action': [0.0] * action_dim,
                'max_action': [1.0] * action_dim,
                'mean_action': [0.0] * action_dim,
                'std_action': [1.0] * action_dim,
                'mean_state': [0.0] * state_dim,
                'std_state': [1.0] * state_dim,
            }

        if self.use_rel_action:
            self.min_action = np.array(action_stats['rel_min_action'])
            self.max_action = np.array(action_stats['rel_max_action'])
            self.mean_action = np.array(action_stats['rel_mean_action'])
            self.std_action = np.array(action_stats['rel_std_action'])
        else:
            self.min_action = np.array(action_stats['min_action'])
            self.max_action = np.array(action_stats['max_action'])
            self.mean_action = np.array(action_stats['mean_action'])
            self.std_action = np.array(action_stats['std_action'])

        self.mean_state = np.array(action_stats['mean_state'])
        self.std_state = np.array(action_stats['std_state'])

        self.repeat_lang_tokens = configs.get('repeat_lang_tokens', 1)
        self.language_first = configs.get('language_first', True)
        self.predict_actions_forward = configs.get('predict_actions_forward', False)
        self.use_transformer_final_hidden_states = bool(
            configs.get('use_transformer_final_hidden_states', False)
        )

        self.action_chunk_size = configs.get('action_chunk_size', 20)
        self.action_dim = configs.get('action_dim', 6)

    def image_transforms(self, image):
        image = _convert_to_rgb(image)
        image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        image = np.asarray(image, dtype=np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image

    def token2id(self, token):
        return self.text_vocab[token]

    @staticmethod
    def get_n_grids_token(n_grids):
        return f"<reserved{8800 + n_grids:05d}>"

    @torch.no_grad()
    def quantize_images(self, image, start_token_id, end_token_id):
        patch_size = 32
        h_grids, w_grids = image.size()[1] // patch_size, image.size()[2] // patch_size

        crop_h = h_grids * patch_size
        crop_w = w_grids * patch_size

        image = image[:, :crop_h, :crop_w].unsqueeze(0)

        image_toks = self.chameleon_ori_translation.convert_img2bp2(
            self.chameleon_ori_image_tokenizer.img_tokens_from_tensor(image.to(self.model.device))
        ).view(-1)

        full_image_toks = image_toks.reshape(image.size(2) // 16, image.size(3) // 16)
        new_line_id = self.token2id(self.new_line_token)

        full_image_toks = torch.cat(
            (
                full_image_toks,
                torch.ones(image.size(2) // 16, 1, device=full_image_toks.device, dtype=full_image_toks.dtype)
                * new_line_id,
            ),
            dim=1,
        ).flatten()

        result_toks = [
            start_token_id,
            self.token2id(self.get_n_grids_token(h_grids)),
            self.token2id(self.get_n_grids_token(w_grids)),
            *full_image_toks.tolist(),
            end_token_id,
        ]

        return result_toks

    def get_condition_tokens(self, text, image, wrist_image=None):
        text_tokens = self.text_tokenizer.encode(
            text,
            truncation=False,
            add_special_tokens=False,
        )
        if self.text_bos_id is not None:
            text_tokens = [self.text_bos_id] + text_tokens

        image_tokens = self.quantize_images(image, start_token_id=self.token2id(self.image_start_token), end_token_id=self.token2id(self.image_end_token))

        if wrist_image is None:
            return text_tokens, image_tokens
        else:
            wrist_image_tokens = self.quantize_images(wrist_image, start_token_id=self.wrist_start_token_id, end_token_id=self.wrist_end_token_id)

            return text_tokens, image_tokens, wrist_image_tokens

    def encode(self, obs, text):
        # RGB observation
        image = Image.fromarray(obs['rgb_obs']['rgb_static'])

        image = self.image_transforms(image).to(self.model.device).to(torch.float32)


        wrist_image = Image.fromarray(obs['rgb_obs']['wrist_static'])

        wrist_image = self.image_transforms(wrist_image).to(self.model.device).to(torch.float32)

        text_tokens, image_tokens, wrist_image_tokens = self.get_condition_tokens(text, image, wrist_image)

        condition_tokens = text_tokens * self.repeat_lang_tokens + image_tokens + wrist_image_tokens + [self.state_token_id]
        input_ids = torch.tensor(
            condition_tokens,
            dtype=torch.int64,
            device=self.model.device,
        ).unsqueeze(0)

        normed_state = (obs['state'].copy() - self.mean_state) / self.std_state
        state_embeds = torch.tensor(
            normed_state,
            device=self.model.state_projection.weight.device,
            dtype=self.model.state_projection.weight.dtype,
        ).unsqueeze(0)

        projected_state_embeds = self.model.state_projection(state_embeds)
        inputs_embeds = self.model.prepare_inputs_for_transformer_model(
            input_ids.clone(),
            projected_state_embeds,
        )

        if not self.use_transformer_final_hidden_states:
            return inputs_embeds

        transformer_outputs = self.model.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            return_dict=True,
        )
        return transformer_outputs.last_hidden_state
