from __future__ import annotations

from abc import ABC, abstractmethod


class PromptBuilder(ABC):
    def __init__(self, model_family: str, system_prompt: str | None = None) -> None:
        self.model_family = model_family
        self.system_prompt = system_prompt

    @abstractmethod
    def add_turn(self, role: str, message: str) -> str: ...

    @abstractmethod
    def get_potential_prompt(self, user_msg: str) -> str: ...

    @abstractmethod
    def get_prompt(self) -> str: ...


class PurePromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: str | None = None) -> None:
        super().__init__(model_family, system_prompt)
        self.bos, self.eos = "<s>", "</s>"
        self.prompt = ""
        self.turn_count = 0

    def add_turn(self, role: str, message: str) -> str:
        if (self.turn_count % 2 == 0 and role != "human") or (
            self.turn_count % 2 == 1 and role != "gpt"
        ):
            raise AssertionError(f"Unexpected role {role!r} at turn {self.turn_count}")
        message = message.replace("<image>", "").strip()
        if self.turn_count % 2 == 0:
            wrapped = f"In: {message}\nOut: "
        else:
            wrapped = f"{message if message != '' else ' '}{self.eos}"
        self.prompt += wrapped
        self.turn_count += 1
        return wrapped

    def get_potential_prompt(self, user_msg: str) -> str:
        return (self.prompt + f"In: {user_msg}\nOut: ").removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        return self.prompt.removeprefix(self.bos).rstrip()


__all__ = ["PromptBuilder", "PurePromptBuilder"]
