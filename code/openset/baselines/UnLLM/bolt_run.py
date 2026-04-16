"""BOLT entry point for UnLLM open-set baseline.

Translates BOLT CLI arguments → UnLLM JSON config → runs UnLLM main() → saves BOLT CSV results.
"""
import argparse
import json
import os
import sys
import time

import pandas as pd
import yaml


def get_bolt_parser():
    parser = argparse.ArgumentParser(description="UnLLM BOLT runner")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--known_cls_ratio", type=float, default=0.25)
    parser.add_argument("--labeled_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--fold_num", type=int, default=5)
    parser.add_argument("--fold_type", type=str, default="fold")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_results_path", type=str, default="./results/openset/unllm")
    parser.add_argument("--num_train_epochs", type=float, default=1)

    # UnLLM-specific
    parser.add_argument("--model_name_or_path", type=str, default="./pretrained_models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--segment_length", type=int, default=2)
    parser.add_argument("--neg_sample", type=int, default=0)
    parser.add_argument("--random_sample", type=int, default=1)
    parser.add_argument("--con_weight", type=float, default=0.1)
    parser.add_argument("--neg_weight", type=float, default=0.1)
    parser.add_argument("--vos_neg_sample", type=int, default=20)
    parser.add_argument("--epsilon", type=int, default=2)
    parser.add_argument("--e_dim", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--train_ablation", type=str, default="binary")
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    return parser


def apply_yaml_config(args, config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ds_cfgs = cfg.pop("dataset_specific_configs", {})
    for k, v in cfg.items():
        k_norm = k.replace("-", "_")
        if hasattr(args, k_norm) and v is not None:
            current = getattr(args, k_norm)
            if current == get_bolt_parser().get_default(k_norm):
                setattr(args, k_norm, v)

    if args.dataset in ds_cfgs:
        for k, v in ds_cfgs[args.dataset].items():
            k_norm = k.replace("-", "_")
            if hasattr(args, k_norm) and v is not None:
                setattr(args, k_norm, v)

    return args


def build_json_config(args, known_labels_file, data_dir):
    """Build a UnLLM-compatible JSON config from BOLT args."""
    default_config_path = os.path.join(os.path.dirname(__file__), "configs", "args.json")
    with open(default_config_path, "r") as f:
        config = json.load(f)

    metric_dir = os.path.join(args.output_dir, "metrics")
    os.makedirs(metric_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "ood_eval_scores"), exist_ok=True)

    config["model_name_or_path"] = args.model_name_or_path
    config["dataset_name"] = args.dataset
    config["rate"] = str(args.known_cls_ratio)
    config["output_dir"] = args.output_dir
    config["metric_dir"] = metric_dir
    config["gen_dir"] = os.path.join(args.output_dir, "gen")
    config["seed"] = args.seed
    config["mode"] = "train"
    config["per_device_train_batch_size"] = args.per_device_train_batch_size
    config["per_device_eval_batch_size"] = args.per_device_eval_batch_size
    config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    config["num_train_epochs"] = args.num_train_epochs
    config["train_ablation"] = args.train_ablation
    config["con_weight"] = args.con_weight
    config["neg_weight"] = args.neg_weight
    config["temperature"] = args.temperature
    config["segment_length"] = args.segment_length
    config["neg_sample"] = args.neg_sample
    config["random_sample"] = args.random_sample
    config["vos_neg_sample"] = args.vos_neg_sample
    config["epsilon"] = args.epsilon
    config["e_dim"] = args.e_dim
    config["learning_rate"] = args.learning_rate
    config["known_labels_file"] = known_labels_file
    config["data_dir"] = data_dir
    config["labeled_ratio"] = args.labeled_ratio
    config["splits"] = "train,eval,test"
    config["load_best_model_at_end"] = True
    config["eval_strategy"] = "steps"
    config["save_strategy"] = "steps"
    config["remove_unused_columns"] = False

    if "Instruct" in args.model_name_or_path:
        config["use_flash_attn"] = True
    config["metric_for_best_model"] = "f1-score--binary" if args.neg_sample == 0 else "OOD-N_recall--binary"

    # Compute eval_steps dynamically
    train_file = os.path.join(data_dir, args.dataset, "origin_data", "train.tsv")
    known_labels = pd.read_csv(known_labels_file, header=None)[0].tolist()
    train_df = pd.read_csv(train_file, sep="\t")
    train_count = len(train_df[train_df["label"].isin(known_labels)])
    if args.labeled_ratio < 1.0:
        labeled_info_path = os.path.join(data_dir, args.dataset, "labeled_data", str(args.labeled_ratio), "train.tsv")
        if os.path.exists(labeled_info_path):
            labeled_info = pd.read_csv(labeled_info_path, sep="\t")
            train_count = int(labeled_info["labeled"].astype(str).str.strip().str.lower().eq("true").sum())
    eval_step_num = 2
    n_gpus = len(args.gpu_id.split(","))
    eval_steps = max(
        train_count // (args.per_device_train_batch_size * args.gradient_accumulation_steps * n_gpus * eval_step_num),
        5,
    )
    config["eval_steps"] = eval_steps
    config["save_steps"] = eval_steps

    if str(args.known_cls_ratio) == "0.25":
        config["num_train_epochs"] = max(config["num_train_epochs"], 1)

    return config, metric_dir


def extract_bolt_metrics(metric_dir, mode="test"):
    """Extract ACC, F1, K-F1, N-F1 from UnLLM test results."""
    csv_path = os.path.join(metric_dir, f"{mode}.csv")
    if not os.path.exists(csv_path):
        json_path = os.path.join(metric_dir, f"{mode}.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                raw = json.load(f)
            acc = raw.get("eval_accuracy--binary", 0) * 100
            f1 = raw.get("eval_f1-score--binary", 0) * 100
            kf1 = raw.get("eval_K-f1--binary", 0) * 100
            nf1 = raw.get("eval_N-f1--binary", 0) * 100
            return acc, f1, kf1, nf1
        return None, None, None, None

    df = pd.read_csv(csv_path, sep="\t")
    binary_row = df[df["eval_ablation"] == "binary"]
    if binary_row.empty:
        binary_row = df.iloc[:1]
    row = binary_row.iloc[0]
    return row.get("ACC"), row.get("F1"), row.get("K-F1"), row.get("N-F1")


def save_bolt_results(args, acc, f1, kf1, nf1):
    import csv as _csv
    os.makedirs(args.save_results_path, exist_ok=True)
    results_file = os.path.join(args.save_results_path, "results.csv")

    fieldnames = ["method", "dataset", "known_cls_ratio", "labeled_ratio",
                  "cluster_num_factor", "seed", "ACC", "F1", "K-F1", "N-F1", "args"]
    row = {
        "method": "UnLLM",
        "dataset": args.dataset,
        "known_cls_ratio": args.known_cls_ratio,
        "labeled_ratio": args.labeled_ratio,
        "cluster_num_factor": 1.0,
        "seed": args.seed,
        "ACC": round(acc, 2) if acc is not None else "",
        "F1": round(f1, 4) if f1 is not None else "",
        "K-F1": round(kf1, 4) if kf1 is not None else "",
        "N-F1": round(nf1, 4) if nf1 is not None else "",
        "args": json.dumps(vars(args), default=str),
    }

    header_needed = not os.path.exists(results_file)
    with open(results_file, "a", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        if header_needed:
            writer.writeheader()
        writer.writerow(row)
    print(f"Results saved to {results_file}")


def main():
    parser = get_bolt_parser()
    args = parser.parse_args()

    if args.config:
        args = apply_yaml_config(args, args.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["WANDB_DISABLED"] = "true"

    # Resolve data paths
    data_dir = os.path.abspath("data")
    known_labels_file = os.path.join(
        data_dir,
        args.dataset,
        "label",
        f"{args.fold_type}{args.fold_num}",
        f"part{args.fold_idx}",
        f"label_known_{args.known_cls_ratio}.list",
    )
    if not os.path.exists(known_labels_file):
        print(f"ERROR: Known labels file not found: {known_labels_file}")
        sys.exit(1)

    # Build JSON config
    config, metric_dir = build_json_config(args, known_labels_file, data_dir)

    config_path = os.path.join(args.output_dir, "bolt_config.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    time.sleep(1)

    # Run train
    unllm_dir = os.path.dirname(os.path.abspath(__file__))
    if unllm_dir not in sys.path:
        sys.path.insert(0, unllm_dir)

    # args.py runs parse_args() and reads relative paths at module level.
    # 1) cd into UnLLM dir so relative paths (data/statics/, configs/) resolve
    # 2) swap sys.argv so args.py's parser doesn't choke on bolt_run flags
    _orig_cwd = os.getcwd()
    os.chdir(unllm_dir)

    _orig_argv = sys.argv
    sys.argv = [
        "args.py",
        "--dataset_name", args.dataset,
        "--rate", str(args.known_cls_ratio),
        "--model_name_or_path", os.path.basename(args.model_name_or_path),
        "--seed", str(args.seed),
        "--gpu_id", str(args.gpu_id),
        "--mode", "train",
        "--default_config", os.path.join(unllm_dir, "configs", "args.json"),
        "--con_weight", str(args.con_weight),
        "--neg_weight", str(args.neg_weight),
        "--temperature", str(args.temperature),
        "--segment_length", str(args.segment_length),
        "--neg_sample", str(args.neg_sample),
        "--random_sample", str(args.random_sample),
        "--vos_neg_sample", str(args.vos_neg_sample),
        "--epsilon", str(args.epsilon),
        "--e_dim", str(args.e_dim),
        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size", str(args.per_device_eval_batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--train_ablation", str(args.train_ablation),
    ]

    from transformers import HfArgumentParser, TrainingArguments
    from args import ModelArguments, DataTrainingArguments

    sys.argv = _orig_argv
    os.chdir(_orig_cwd)

    hf_parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = hf_parser.parse_json_file(json_file=config_path)

    from main import main as unllm_main
    unllm_main(model_args, data_args, training_args)

    # Run eval — generates best_class_thred.npy needed by test
    config["mode"] = "eval"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    model_args, data_args, training_args = hf_parser.parse_json_file(json_file=config_path)
    unllm_main(model_args, data_args, training_args)

    # Run test
    config["mode"] = "test"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    model_args, data_args, training_args = hf_parser.parse_json_file(json_file=config_path)
    unllm_main(model_args, data_args, training_args)

    # Extract and save results
    acc, f1, kf1, nf1 = extract_bolt_metrics(metric_dir, mode="test")
    if acc is not None:
        save_bolt_results(args, acc, f1, kf1, nf1)
        print(f"UnLLM results: ACC={acc:.2f}, F1={f1:.4f}, K-F1={kf1:.4f}, N-F1={nf1:.4f}")
    else:
        print("WARNING: Could not extract metrics from UnLLM output")


if __name__ == "__main__":
    main()
