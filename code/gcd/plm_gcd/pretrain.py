import os
from configs import args
from utils import get_best_checkpoint, create_model
import torch
import logging
import numpy as np
import pandas as pd

from load_dataset import (
    train_dataset,
    dataset_in_eval,
    collate_batch,
    tokenizer,
)

from sklearn.metrics import classification_report
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback


if (
    os.path.exists(args.checkpoint_path)
    and get_best_checkpoint(args.checkpoint_path) is not None
):
    exit()


def compute_metrics(eval_predictions):
    preds, golds = eval_predictions
    preds = np.argmax(preds, axis=1)
    metrics = classification_report(golds, preds, output_dict=True)

    metrics["macro avg"].update({"accuracy": metrics["accuracy"]})
    return metrics["macro avg"]


peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=(
        ["query", "key", "value"] if "bert" in args.model_path else ["q_proj", "v_proj"]
    ),
)


base_model = create_model(model_path=args.model_path, num_labels=args.num_labels)
model = get_peft_model(base_model, peft_config)


model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id


training_args = TrainingArguments(
    output_dir=args.checkpoint_path,
    eval_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    num_train_epochs=args.num_pretrain_epochs,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none",
)

callbacks = [
    EarlyStoppingCallback(
        early_stopping_patience=getattr(args, "es_patience", 3),
        early_stopping_threshold=getattr(args, "es_min_delta", 0.0),
    )
]

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dataset_in_eval,
    compute_metrics=compute_metrics,
    data_collator=collate_batch,
    tokenizer=tokenizer,
    callbacks=callbacks,
)


results = trainer.evaluate()
logging.info("initial %s", results)
trainer.train()
results = trainer.evaluate()
logging.info("final %s", results)
