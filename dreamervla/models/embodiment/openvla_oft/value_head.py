# Copyright 2025 The DreamerVLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import torch.nn as nn


class ValueHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes=(512, 128),
        output_dim: int = 1,
        activation: str = "gelu",
        bias_last: bool = False,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim
        if activation.lower() == "relu":
            act = nn.ReLU
        elif activation.lower() == "gelu":
            act = nn.GELU
        elif activation.lower() == "tanh":
            act = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim, bias=bias_last))
        self.mlp = nn.Sequential(*layers)
        self._init_weights(activation.lower())

    def _init_weights(self, nonlinearity="relu"):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                if module is self.mlp[-1]:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    nn.init.kaiming_normal_(
                        module.weight, mode="fan_out", nonlinearity=nonlinearity
                    )
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.mlp(x)
