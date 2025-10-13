import torch
from transformers import Trainer
from torch import nn
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from transformers.trainer import _is_peft_model, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from src.pytorch_ood.loss import VOSRegLoss, NPOSRegLoss
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

class RegTrainer(Trainer):
    def __init__(self, reg_loss,num_labels, device,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_loss  = reg_loss
        if reg_loss == 'vos':
            phi = torch.nn.Linear(1, 2).to(device)
            weights_energy = torch.nn.Linear(num_labels, 1).to(device)
            torch.nn.init.uniform_(weights_energy.weight)
            reg_loss_fn = VOSRegLoss(
                logistic_regression=phi,
                weights_energy=weights_energy,
                alpha=0.1,
                device=device,
            )
            self.reg_loss_fn = reg_loss_fn
        elif reg_loss == 'npo':  # fix here
            phi = torch.nn.Linear(self.model.config.hidden_size, 1).to(device)
            self.reg_loss_fn = NPOSRegLoss(
                phi=phi,
                sigma=0.5,
                k=1,
                p=5,
                device=device
            )
        else:
            self.reg_loss_fn = None

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Args:
            model (`nn.Module`):
                The model to compute the loss for.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The input data for the model.
            return_outputs (`bool`, *optional*, defaults to `False`):
                Whether to return the model outputs along with the loss.
            num_items_in_batch (Optional[torch.Tensor], *optional*):
                The number of items in the batch. If num_items_in_batch is not passed,

        Returns:
            The loss of the model along with its output if return_outputs was set to True

        Subclass and override for custom behavior. If you are not using `num_items_in_batch` when computing your loss,
        make sure to overwrite `self.model_accepts_loss_kwargs` to `False`. Otherwise, the loss calculationg might be slightly inacuknown_cls_ratio when performing gradient accumulation.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        inputs['output_hidden_states'] = True
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
                
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        if self.reg_loss == 'npo':    
            loss += self.reg_loss_fn(outputs.hidden_states[-1][:,-1], inputs['labels'])
        
        if self.reg_loss == 'vos':    
            loss += self.reg_loss_fn(outputs.logits, inputs['labels'])

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        # del outputs.hidden_states
        # outputs.__delitem__('hidden_states') 
        inputs['output_hidden_states'] = False

        new_output = SequenceClassifierOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values
        )

        return (loss, new_output) if return_outputs else loss