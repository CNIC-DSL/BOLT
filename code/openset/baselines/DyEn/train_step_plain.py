from typing import Dict, Any

import torch
from torch import nn


class TrainStep:
    @staticmethod
    def _model_forward(model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        loss = model(inputs, mode="train")
        return loss

    def step(self, model: nn.Module, inputs: Dict[str, Any]) -> float:

        loss = self._model_forward(model, inputs)
        loss.backward()

        total_loss = loss.item()
        return total_loss
