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
import yaml
import json

logger = logging.getLogger(__name__)
torch.set_num_threads(6)


def apply_yaml_to_dataclasses(yaml_config, dataclass_tuple, cli_args):
    arg_list = [arg.lstrip("-") for arg in cli_args if arg.startswith("--")]

    for key, value in yaml_config.items():
        if key in arg_list:

            continue

        for dc_obj in dataclass_tuple:
            if hasattr(dc_obj, key):
                setattr(dc_obj, key, value)
                break


def main():

    parser = HfArgumentParser(
        (OtherArguments, DataTrainingArguments, TrainingArguments, FitLogArguments)
    )
    other_args: OtherArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments
    fitlog_args: FitLogArguments

    other_args, data_args, training_args, fitlog_args = (
        parser.parse_args_into_dataclasses()
    )

    all_args_tuple = (other_args, data_args, training_args, fitlog_args)
    if other_args.config:
        with open(other_args.config, "r") as f:
            yaml_config = yaml.safe_load(f)

        apply_yaml_to_dataclasses(yaml_config, all_args_tuple, sys.argv)

        if "dataset_specific_configs" in yaml_config:
            dataset_name = data_args.dataset
            if dataset_name in yaml_config["dataset_specific_configs"]:
                dataset_specific_config = yaml_config["dataset_specific_configs"][
                    dataset_name
                ]
                apply_yaml_to_dataclasses(
                    dataset_specific_config, all_args_tuple, sys.argv
                )

    assert other_args.loss_type in [
        "original",
        "increase",
        "ce_and_div_drop-last-layer",
        "ce_and_div",
    ]

    training_args.remove_unused_columns = False

    set_seed(training_args.seed)

    model_output_root = training_args.output_dir
    os.makedirs(model_output_root, exist_ok=True)

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
            [data_args.dataset, str(data_args.known_cls_ratio), str(training_args.seed)]
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
    )
    if training_args.do_train:
        file_postfix = (
            "_".join(
                [
                    data_args.dataset,
                    str(data_args.known_cls_ratio),
                    str(training_args.seed),
                ]
            )
            + ".csv"
        )

        trainer.train_ce_loss()

    else:
        model.load_state_dict(
            torch.load(os.path.join(model_output_root, model_file_name))
        )
        model.to(training_args.device)

    if training_args.do_predict:
        from sklearn.metrics import classification_report
        import pandas as pd

        file_postfix = (
            "_".join(
                [
                    data_args.dataset,
                    str(data_args.known_cls_ratio),
                    str(training_args.seed),
                ]
            )
            + ".csv"
        )

        valid_all_dataloader = trainer.get_eval_dataloader(datasets["valid_all"])
        valid_dataloader = trainer.get_eval_dataloader(datasets["valid_seen"])
        train_dataloader = trainer.get_train_dataloader()
        test_dataloader = trainer.get_test_dataloader(datasets["test"])

        model_forward_cache = {}

        kwargs = dict(
            model=model,
            root=model_output_root,
            file_postfix=file_postfix,
            dataset_name=data_args.dataset,
            device=training_args.device,
            num_labels=num_all_labels,
            tuning="valid",
            scale_ind=other_args.scale,
            scale_ood=other_args.scale_ood,
            valid_all_dataloader=valid_all_dataloader,
            valid_dataloader=valid_dataloader,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            model_forward_cache=model_forward_cache,
        )

        evaluator_classes = [
            (my_eval.KnnEvaluator, "knn"),
            (my_eval.MspEvaluator, "msp"),
            (my_eval.MaxLogitEvaluator, "maxLogit"),
            (my_eval.EnergyEvaluator, "energy"),
            (my_eval.EntropyEvaluator, "entropy"),
            (my_eval.OdinEvaluator, "odin"),
            (my_eval.MahaEvaluator, "maha"),
            (my_eval.LofCosineEvaluator, "lof_cosine"),
            (my_eval.LofEuclideanEvaluator, "lof_euclidean"),
        ]

        id_to_label = {i: v for i, v in enumerate(label_list)}

        for eval_class, method_name in evaluator_classes:
            print(f"\n---> Running Evaluation for method: {method_name}")

            if method_name in ["energy", "odin"]:
                temp = 1 if method_name == "energy" else 100
                evaluator = eval_class(temperature=temp, **kwargs)
            else:
                evaluator = eval_class(**kwargs)

            y_pred_ids, y_true_ids = evaluator.eval()

            y_true_labels = [
                id_to_label.get(label_id, "oos") for label_id in y_true_ids
            ]
            y_pred_labels = [
                id_to_label.get(label_id, "oos") for label_id in y_pred_ids
            ]

            report = classification_report(
                y_true_labels, y_pred_labels, output_dict=True, zero_division=0
            )

            final_results = {}
            final_results["dataset"] = data_args.dataset
            final_results["seed"] = training_args.seed
            final_results["known_cls_ratio"] = data_args.known_cls_ratio
            final_results["ood_method"] = method_name
            final_results["ACC"] = report["accuracy"]
            final_results["F1"] = report["macro avg"]["f1-score"]

            seen_class_labels = [l for l in label_list if l != "oos"]
            known_f1_scores = [
                report[label]["f1-score"]
                for label in seen_class_labels
                if label in report
            ]
            final_results["K-F1"] = (
                sum(known_f1_scores) / len(known_f1_scores) if known_f1_scores else 0.0
            )
            final_results["N-F1"] = (
                report["oos"]["f1-score"] if "oos" in report else 0.0
            )

            def safe_args_to_json(*args_objs):
                safe_dict = {}
                for args_obj in args_objs:
                    if args_obj is None:
                        continue
                    for k, v in vars(args_obj).items():
                        try:
                            json.dumps(v)
                            safe_dict[k] = v
                        except (TypeError, OverflowError):
                            safe_dict[k] = str(v)
                return json.dumps(safe_dict, ensure_ascii=False)

            final_results["args"] = safe_args_to_json(
                other_args, training_args, data_args
            )

            os.makedirs(other_args.save_results_path, exist_ok=True)
            results_path = os.path.join(other_args.save_results_path, "results.csv")

            df_to_save = pd.DataFrame([final_results])

            df_to_save["method"] = "DyEn"

            cols = [
                "method",
                "dataset",
                "known_cls_ratio",
                "labeled_ratio",
                "cluster_num_factor",
                "seed",
                "ACC",
                "F1",
                "K-F1",
                "N-F1",
                "args",
            ]

            val = lambda c: next(
                (
                    getattr(src, c)
                    for src in (other_args, training_args, data_args)
                    if hasattr(src, c)
                ),
                None,
            )

            for c in cols:
                df_to_save[c] = df_to_save.get(c, val(c))

            df_to_save = df_to_save[cols]

            if not os.path.exists(results_path):
                df_to_save.to_csv(results_path, index=False)
            else:
                existing_df = pd.read_csv(results_path)

                updated_df = pd.concat([existing_df, df_to_save], ignore_index=True)
                updated_df.to_csv(results_path, index=False)

            print(f"Saved results for {method_name} to {results_path}")

    return None


if __name__ == "__main__":
    main()
