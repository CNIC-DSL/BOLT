from typing import Dict, Any

import torch
from torch import nn


class FreeLB:
    def __init__(
        self,
        adv_k: int,
        adv_lr: float,
        adv_init_mag: float,
        adv_max_norm: float,
        base_model: str = "bert",
        adv_norm_type: str = "l2",
    ):
        self.adv_k = adv_k
        self.adv_lr = adv_lr
        self.adv_max_norm = adv_max_norm
        self.adv_init_mag = adv_init_mag
        self.adv_norm_type = adv_norm_type
        self.base_model = base_model
        assert self.adv_norm_type == "l2"

        self.random_states = {}

    @staticmethod
    def _model_forward(model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        loss = model(inputs, mode="train")
        return loss

    def _save_random_state(self) -> None:
        self.random_states = {
            "cuda_cur_state": torch.cuda.get_rng_state(),
            "cur_state": torch.get_rng_state(),
        }

    def _load_random_state(self) -> None:
        torch.cuda.set_rng_state(self.random_states["cuda_cur_state"])
        torch.set_rng_state(self.random_states["cur_state"])

    def step(self, model: nn.Module, inputs: Dict[str, Any]) -> float:

        self._save_random_state()

        loss = self._model_forward(model, inputs)

        loss: torch.Tensor = torch.div(loss, (1 + self.adv_k))
        loss.backward()

        total_loss = loss.item()

        input_ids = inputs["input_ids"]

        tmp_embeds_init = (
            getattr(model, self.base_model)
            .embeddings.word_embeddings(input_ids)
            .clone()
            .detach()
        )

        if self.adv_init_mag > 0:
            input_mask: torch.Tensor = inputs["attention_mask"]
            input_lengths = torch.sum(input_mask, 1)

            delta = torch.zeros_like(tmp_embeds_init).uniform_(
                -1, 1
            ) * input_mask.unsqueeze(2)

            dims = input_lengths * tmp_embeds_init.size(-1)
            mag = self.adv_init_mag / torch.sqrt(dims)

            delta = delta * mag.view(-1, 1, 1)
        else:
            delta = torch.zeros_like(tmp_embeds_init)

        for i in range(self.adv_k):

            delta.requires_grad_()

            embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(
                input_ids
            )

            inputs["inputs_embeds"] = delta + embeds_init
            inputs["input_ids"] = None

            self._load_random_state()

            loss = self._model_forward(model, inputs)

            loss: torch.Tensor = torch.div(loss, (1 + self.adv_k))
            loss.backward()

            total_loss += loss.item()

            delta_grad = delta.grad.clone().detach()
            delta = delta.clone().detach()

            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(
                -1, 1, 1
            )
            denorm = torch.clamp(denorm, min=1e-8)

            delta = delta + self.adv_lr * delta_grad / denorm

            if self.adv_max_norm > 0:
                delta_norm = torch.norm(
                    delta.view(delta.size(0), -1).float(), p=2, dim=1
                )
                exceed_mask = torch.gt(delta_norm, self.adv_max_norm).to(embeds_init)
                reweights = (
                    self.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)
                ).view(-1, 1, 1)
                delta = delta * reweights

        return total_loss
