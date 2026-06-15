from __future__ import annotations

from transformers import AutoTokenizer


class Tokenizer:
    """Tokenizer adapter used by the legacy XLLMX preprocessing helpers."""

    def __init__(self, model_path: str) -> None:
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

    def encode_wo_prefix_space(self, s: str) -> list[int]:
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

    def _probe_tokenizer_style(self) -> None:
        sentence1 = self.encode("Hi my darling", bos=False, eos=False)
        sentence2 = self.encode("my darling", bos=False, eos=False)
        if sentence1[-len(sentence2) :] == sentence2:
            self.need_space_before_segment = False
            return

        sentence3 = self.encode(" my darling", bos=False, eos=False)
        assert sentence1[-len(sentence3) :] == sentence3
        self.need_space_before_segment = True


__all__ = ["Tokenizer"]
