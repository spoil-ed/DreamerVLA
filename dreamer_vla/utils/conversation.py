"""Conversation template used by RynnVLA tokenization paths."""

from __future__ import annotations

from collections.abc import Sequence


class Conversation:
    """Minimal two-role conversation template for RynnVLA-style prompts."""

    sep_token = "<reserved08706>"
    roles = ["Human", "Assistant"]

    def __init__(self, messages: Sequence[Sequence[str | None]] | None = None) -> None:
        self.messages = [list(message) for message in messages] if messages else []

    def process(self) -> dict[str, list[dict[str, bool | str]] | str]:
        conv = ""
        pieces: list[dict[str, bool | str]] = []
        for idx, (role, message) in enumerate(self.messages):
            if message is None:
                assert idx == len(self.messages) - 1 and role == self.roles[1], (
                    "only last assistant message can be None"
                )
                continue
            turn = message + self.sep_token
            conv += turn
            pieces.append({"data": turn, "predict": role == self.roles[1]})
        return {"conv": conv, "pieces": pieces}

    def get_prompt(self) -> str:
        """Return the serialized prompt text."""
        return str(self.process()["conv"])

    def append_message(self, role: str, message: str | None) -> None:
        self.messages.append([role, message])

    def copy(self) -> Conversation:
        """Return a shallow copy of the conversation messages."""
        return Conversation(messages=self.messages)

    def load_qas(self, qas: Sequence[Sequence[str | None]]) -> None:
        """Load question/answer pairs into the template."""
        self.messages = []
        for question, answer in qas:
            self.append_message(self.roles[0], question)
            self.append_message(self.roles[1], answer)
