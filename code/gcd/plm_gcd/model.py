import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
from utils import get_best_checkpoint, create_model
from peft import get_peft_model, LoraConfig, TaskType


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = args.backbone
        checkpoint_path = get_best_checkpoint(args.checkpoint_path)
        if checkpoint_path is None:
            checkpoint_path = args.model_path

        base_model = create_model(
            model_path=args.model_path, num_labels=args.num_labels
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
        self.model.eval()
        self.model.config.pad_token_id = tokenizer.pad_token_id
        self.model.config.eos_token_id = tokenizer.eos_token_id

        if hasattr(self.model.base_model, "bert"):
            self.llm = self.model.base_model.bert
            self.fc = self.model.base_model.classifier
        elif hasattr(self.model.base_model, "roberta"):
            self.llm = self.model.base_model.roberta
            self.fc = self.model.base_model.classifier
        else:
            self.llm = self.model.base_model.model.model
            self.fc = self.model.base_model.score

    def features(self, x):
        outputs = self.llm(**x)
        if self.backbone == "bert":
            return outputs.pooler_output
        elif self.backbone == "roberta":
            return outputs.last_hidden_state[:, 0]
        else:
            return outputs.last_hidden_state[:, -1]

    def forward(self, x):
        outputs = self.model(**x)
        return outputs.logits
