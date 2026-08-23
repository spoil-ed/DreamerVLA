from __future__ import annotations

from omegaconf import OmegaConf


def _dvla_mul(*values: object) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _dvla_div(value: object, *divisors: object) -> int:
    result = int(value)
    divisor = _dvla_mul(*divisors)
    if divisor == 0:
        raise ValueError("dvla_div divisor must not be zero")
    if result % divisor != 0:
        raise ValueError(f"dvla_div requires exact division, got {result} / {divisor}")
    return result // divisor


def register_dreamervla_resolvers() -> None:
    """Register project-local OmegaConf resolvers used by Hydra YAML."""
    OmegaConf.register_new_resolver("dvla_mul", _dvla_mul, replace=True)
    OmegaConf.register_new_resolver("dvla_div", _dvla_div, replace=True)
