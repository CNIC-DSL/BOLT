import argparse
import os
import logging
import sys
import yaml
import torch


parser = argparse.ArgumentParser()

parser.add_argument(
    "--config", type=str, default=None, help="Path to the YAML config file."
)

parser.add_argument("--dataset", default="banking", type=str)
parser.add_argument("--data_dir", default="./data", type=str)
parser.add_argument("--known_cls_ratio", default=0.25, type=float)
parser.add_argument("--num_pretrain_epochs", default=20, type=int)
parser.add_argument("--num_train_epochs", default=20, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--gpu_id", default="0", type=str)


parser.add_argument("--root", default="data", type=str)
parser.add_argument("--output_dir", default="outputs", type=str)
parser.add_argument("--save_results_path", default="results", type=str)
parser.add_argument("--backbone", default="Meta-Llama-3.1-8B-Instruct", type=str)
parser.add_argument(
    "--bert_model", default="./pretrained_models/bert-base-uncased", type=str
)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--eval_batch_size", default=128, type=int)
parser.add_argument("--labeled_ratio", default=0.1, type=float)
parser.add_argument("--fold_idx", default=0, type=int)
parser.add_argument("--fold_num", default=5, type=int)
parser.add_argument(
    "--es_patience",
    type=int,
    default=3,
    help="Early stopping patience (in eval steps or epochs when evaluation_strategy='epoch').",
)
parser.add_argument(
    "--es_min_delta",
    type=float,
    default=0.0,
    help="Minimum improvement to qualify as better.",
)
parser.add_argument(
    "--metric_for_best",
    type=str,
    default="accuracy",
    help="Metric name from compute_metrics to select best model.",
)
parser.add_argument(
    "--fold_type", type=str, default="fold", help="", choices=["imbalance_fold", "fold"]
)


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


def apply_config_updates(args, config_dict, parser):
    type_map = {action.dest: action.type for action in parser._actions}
    for key, value in config_dict.items():
        if f"--{key}" in sys.argv or not hasattr(args, key):
            continue
        expected_type = type_map.get(key)
        if expected_type and value is not None:
            try:

                if expected_type is bool:
                    value = str(value).lower() in ("true", "1", "t", "yes")
                else:
                    value = expected_type(value)
            except (ValueError, TypeError):
                print(
                    f"Warning: Could not cast YAML value '{value}' for key '{key}' to type {expected_type}."
                )
        setattr(args, key, value)


if args.config:
    with open(args.config, "r") as f:
        yaml_config = yaml.safe_load(f)
    apply_config_updates(args, yaml_config, parser)
    if "dataset_specific_configs" in yaml_config:
        dataset_configs = yaml_config["dataset_specific_configs"].get(args.dataset, {})
        apply_config_updates(args, dataset_configs, parser)


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


args.model_path = f"./pretrained_models/{args.backbone}"

args.output_dir = f"{args.output_dir}/gcd/plm_gcd/{args.dataset}_{args.known_cls_ratio}_{args.seed}_{args.backbone}"
args.checkpoint_path = os.path.join(args.output_dir, "checkpoints")
args.log_dir = os.path.join(args.output_dir, "logs")
args.case_path = os.path.join(args.output_dir, "case_study")
args.metric_dir = args.save_results_path


os.makedirs(args.metric_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.case_path, exist_ok=True)
os.makedirs(args.checkpoint_path, exist_ok=True)


args.metric_file = os.path.join(args.metric_dir, "results.csv")


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.log_dir, "run.log"), mode="a"),
        logging.StreamHandler(),
    ],
)

logging.info("=" * 20 + " New Run Initialized " + "=" * 20)
logging.info(f"Arguments loaded and processed: {args}")
