# model.py (改造后的版本)

import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
from utils import get_best_checkpoint, create_model
from peft import get_peft_model, LoraConfig, TaskType

class Model(nn.Module):
    # --- 改动 1: __init__ 现在接收 tokenizer 作为参数 ---
    def __init__(self, args, tokenizer):
        super().__init__()
        self.backbone = args.backbone
        checkpoint_path = get_best_checkpoint(args.checkpoint_path)
        if checkpoint_path is None:
            # 如果没有训练好的 checkpoint，就使用原始的预训练模型路径
            checkpoint_path = args.model_path

        # --- 改动 2: 调用 create_model 时传入 tokenizer ---
        base_model = create_model(model_path=args.model_path, num_labels=args.num_labels, tokenizer=tokenizer)
        
        # --- 改动 3: 不再内部创建 tokenizer，直接使用传入的 ---
        # tokenizer = AutoTokenizer.from_pretrained(checkpoint_path) # <- 删除这一行

        # 加载 LoRA 权重
        self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
        self.model.eval()
        self.model.config.pad_token_id = tokenizer.pad_token_id
        self.model.config.eos_token_id = tokenizer.eos_token_id

        # 保留底层 LLM 和分类头 (这部分逻辑不变)
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
        if 'bert' in self.backbone:
            return outputs.pooler_output
        elif 'roberta' in self.backbone:
            return outputs.last_hidden_state[:, 0]  # use [CLS] token embedding
        else:
            # 适用于 Llama 等模型，取最后一个 token 的 embedding
            return outputs.last_hidden_state[:, -1]

    def forward(self, x):
        outputs = self.model(**x)
        return outputs.logits