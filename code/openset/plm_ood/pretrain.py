import os

os.environ["WANDB_DISABLED"] = "true"
from transformers import (
    BertForSequenceClassification,
    AutoModelForSequenceClassification,
    TrainingArguments,
)
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig
import logging
from sklearn.metrics import classification_report
import numpy as np
from transformers import EarlyStoppingCallback

from utils import get_best_checkpoint, create_model

from load_dataset import load_and_prepare_datasets
from reg_trainer import RegTrainer

from configs import create_parser, finalize_config
import yaml
import sys


def run_pretraining(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    logging.info("Loading and preparing datasets for pretraining...")
    data = load_and_prepare_datasets(args)

    tokenizer = data["tokenizer"]
    train_dataset = data["train_dataset"]
    dataset_in_eval = data["dataset_in_eval"]
    collate_batch = data["collate_batch"]
    logging.info("Datasets loaded successfully.")

    if (
        os.path.exists(args.checkpoint_path)
        and get_best_checkpoint(args.checkpoint_path) is not None
    ):
        logging.warning(
            f"Checkpoint already exists at {args.checkpoint_path}. Exiting pretraining."
        )
        return

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
            ["query", "key", "value"]
            if "bert" in args.model_path
            else ["q_proj", "v_proj"]
        ),
    )
    tokenizer = data["tokenizer"]
    base_model = create_model(
        model_path=args.model_path, num_labels=args.num_labels, tokenizer=tokenizer
    )
    model = get_peft_model(base_model, peft_config)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    early_stop_patience = getattr(args, "early_stop_patience", 3)
    early_stop_delta = getattr(args, "early_stop_delta", 0.0)

    training_args = TrainingArguments(
        output_dir=args.checkpoint_path,
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1-score",
        greater_is_better=True,
        report_to=[],
    )

    trainer = RegTrainer(
        reg_loss=args.reg_loss,
        num_labels=args.num_labels,
        device=args.device,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dataset_in_eval,
        compute_metrics=compute_metrics,
        data_collator=collate_batch,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stop_patience,
                early_stopping_threshold=early_stop_delta,
            )
        ],
    )

    logging.info("Evaluating before training...")
    results = trainer.evaluate()
    logging.info(f"Initial results: {results}")

    logging.info("Starting training...")
    trainer.train()

    logging.info("Evaluating after training...")
    results = trainer.evaluate()
    logging.info(f"Final results: {results}")


def apply_config_updates(args, config_dict, parser):

    type_map = {action.dest: action.type for action in parser._actions}

    for key, value in config_dict.items():

        if f"--{key}" not in sys.argv and hasattr(args, key):

            expected_type = type_map.get(key)

            if expected_type and value is not None:
                value = expected_type(value)
            setattr(args, key, value)


if __name__ == "__main__":

    parser = create_parser()

    parser.add_argument("--config", type=str, help="Path to the YAML config file")

    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
        apply_config_updates(args, yaml_config, parser)

    config_args = finalize_config(args)

    run_pretraining(config_args)
