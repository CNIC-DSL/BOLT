from sklearn.metrics import classification_report
import numpy as np
import os
import re


from peft import get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig
from transformers import (
    BertForSequenceClassification,
    AutoModelForSequenceClassification,
)
import torch


def compute_metrics(eval_predictions):

    preds, golds = eval_predictions
    preds = np.argmax(preds, axis=1)
    metrics = classification_report(preds, golds, output_dict=True)
    metrics["macro avg"].update({"accuracy": metrics["accuracy"]})
    return metrics["macro avg"]


def get_best_checkpoint(output_dir):

    checkpoints = [d for d in os.listdir(output_dir) if re.match(r"checkpoint-\d+", d)]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]), reverse=False)
    latest_checkpoint = os.path.join(output_dir, checkpoints[0])
    return latest_checkpoint


def create_model(model_path, num_labels, tokenizer):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_8bit_compute_dtype=torch.float32,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=False,
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        device_map="auto",
        attn_implementation="eager",
        quantization_config=quantization_config,
        torch_dtype=torch.float32,
    )

    base_model.resize_token_embeddings(
        len(tokenizer), pad_to_multiple_of=8, mean_resizing=False
    )

    return base_model
