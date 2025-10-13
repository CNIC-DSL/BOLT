import os
import logging
import shutil
import sys

import torch
import fitlog

import transformers
import transformers.utils.logging
from transformers import (
    AutoConfig,
    AutoTokenizer,
    set_seed,
    BertConfig,
)
from transformers.trainer_utils import is_main_process

import eval as my_eval
from handle_data import load_datasets, convert_to_nums, data_collator
from my_args import DataTrainingArguments, FitLogArguments, OtherArguments
from my_trainer import SimpleTrainer
from transformers.training_args import TrainingArguments
from models import BertForSequenceClassificationWithPabee
from my_hf_argparser import HfArgumentParser
import train_step_freelb, train_step_plain

logger = logging.getLogger(__name__)
torch.set_num_threads(6)


def main():

    parser = HfArgumentParser(
        (OtherArguments, DataTrainingArguments, TrainingArguments, FitLogArguments)
    )
    other_args: OtherArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments
    fitlog_args: FitLogArguments

    assert len(sys.argv) == 2 and sys.argv[1].endswith(".json")
    other_args, data_args, training_args, fitlog_args = parser.parse_json_file(
        json_file=os.path.abspath(sys.argv[1])
    )

    assert other_args.loss_type in [
        "original",
        "increase",
        "ce_and_div_drop-last-layer",
        "ce_and_div",
    ]

    training_args.remove_unused_columns = False

    set_seed(training_args.seed)

    if training_args.do_train:
        adv_args = f"{other_args.adv_k}{other_args.adv_lr}{other_args.adv_init_mag}"

        if other_args.scale == other_args.scale_ood:
            model_output_root = (
                f"./model_output/{data_args.data}_{data_args.known_ratio}/ad{adv_args}_lr{training_args.learning_rate:.1e}__epoch{training_args.num_train_epochs}__loss{other_args.loss_type}"
                f"__batchsize{training_args.per_device_train_batch_size}__lambda{other_args.diversity_loss_weight}__scale{other_args.scale}/"
            )
        else:
            model_output_root = (
                f"./model_output/{data_args.data}_{data_args.known_ratio}/ad{adv_args}_lr{training_args.learning_rate:.1e}__epoch{training_args.num_train_epochs}__loss{other_args.loss_type}"
                f"__batchsize{training_args.per_device_train_batch_size}__lambda{other_args.diversity_loss_weight}__scale{other_args.scale}{other_args.scale_ood}/"
            )
        if not os.path.exists(model_output_root):
            os.makedirs(model_output_root)
    else:

        model_output_root = os.path.dirname(os.path.abspath(sys.argv[1]))

    if training_args.do_train:
        json_file_name = (
            "_"
            + "_".join(
                [data_args.data, str(data_args.known_ratio), str(training_args.seed)]
            )
            + ".json"
        )

    fitlog.set_log_dir(other_args.fitlog_dir)
    fitlog_args_dict = {
        "seed": training_args.seed,
        "warmup_steps": training_args.warmup_steps,
        "task_name": f"{data_args.data}-{data_args.known_ratio}-{training_args.seed}",
    }
    fitlog_args_name = [i for i in dir(fitlog_args) if i[0] != "_"]
    for args_name in fitlog_args_name:
        args_value = getattr(fitlog_args, args_name)
        if args_value is not None:
            fitlog_args_dict[args_name] = args_value
    fitlog.add_hyper(fitlog_args_dict)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=(
            logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
        ),
    )

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    datasets = load_datasets(data_args)

    is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
    assert not is_regression

    label_list = datasets["train"].unique("label")
    label_list += ["oos"]
    num_all_labels = len(label_list)

    tokenizer: transformers.PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        other_args.bert_model,
        cache_dir=other_args.cache_dir,
        use_fast=True,
    )

    datasets = convert_to_nums(data_args, datasets, label_list, tokenizer)

    pertained_config: BertConfig = AutoConfig.from_pretrained(
        other_args.bert_model,
        finetuning_task=None,
        cache_dir=other_args.cache_dir,
    )

    model = BertForSequenceClassificationWithPabee(
        pertained_config=pertained_config,
        other_args=other_args,
        num_ind_labels=num_all_labels - 1,
    )

    model_file_name = (
        "_"
        + "_".join(
            [data_args.data, str(data_args.known_ratio), str(training_args.seed)]
        )
        + ".pt"
    )

    if other_args.adv_k > 0:
        train_step = train_step_freelb.FreeLB(
            adv_k=other_args.adv_k,
            adv_lr=other_args.adv_lr,
            adv_init_mag=other_args.adv_init_mag,
            adv_max_norm=other_args.adv_max_norm,
        )
    else:
        train_step = train_step_plain.TrainStep()
    trainer = SimpleTrainer(
        num_train_epochs=training_args.num_train_epochs,
        clip=other_args.clip,
        model_path_=os.path.join(model_output_root, model_file_name),
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid_seen"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_step=train_step,
        label_to_id={v: i for i, v in enumerate(label_list)},
    )

    print()
    print()

    if training_args.do_train:
        trainer.train_ce_loss()
    else:
        model.load_state_dict(
            torch.load(os.path.join(model_output_root, model_file_name))
        )
        model.to(training_args.device)

    dataset = trainer.get_eval_dataloader(datasets["valid_all"]).dataset
    print("valid_all", id(dataset))
    dataset = trainer.get_eval_dataloader(datasets["valid_seen"]).dataset
    print("valid_seen", id(dataset))
    dataset = trainer.get_train_dataloader().dataset
    print("train", id(dataset))
    dataset = trainer.get_test_dataloader(datasets["test"]).dataset
    print("test", id(dataset))


if __name__ == "__main__":
    main()
