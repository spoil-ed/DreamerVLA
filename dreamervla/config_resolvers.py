from __future__ import annotations

from omegaconf import OmegaConf


def _dvla_mul(*values: object) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def register_dreamervla_resolvers() -> None:
    """Register project-local OmegaConf resolvers used by Hydra YAML."""
    OmegaConf.register_new_resolver("dvla_mul", _dvla_mul, replace=True)
