"""BOLT entry point for LLM4OpenSSL GCD baseline.

Translates BOLT CLI arguments → JSON config → runs LLM4OpenSSL main() → saves BOLT CSV results.
"""
import argparse
import json
import os
import shutil
import sys
import pandas as pd
import yaml


def get_bolt_parser():
    parser = argparse.ArgumentParser(description="LLM4OpenSSL BOLT runner")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--known_cls_ratio", type=float, default=0.25)
    parser.add_argument("--labeled_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--fold_num", type=int, default=5)
    parser.add_argument("--fold_type", type=str, default="fold")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_results_path", type=str, default="./results/gcd/llm4openssl")
    parser.add_argument("--num_train_epochs", type=int, default=12)

    # LLM4OpenSSL-specific
    parser.add_argument("--model_name_or_path", type=str, default="./pretrained_models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--linear_learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_semi_warmup_epochs", type=int, default=5)
    parser.add_argument("--num_gen_warmup_epochs", type=int, default=10)
    parser.add_argument("--is_semi", type=str, default="semisurpervised")
    parser.add_argument("--is_mlp", type=str, default="mlp")
    parser.add_argument("--cca_loss_func", type=str, default="log")
    parser.add_argument("--cca_k", type=int, default=16)
    parser.add_argument("--cca_loss_weight", type=float, default=0.01)
    parser.add_argument("--class_loss_weight", type=float, default=1.0)
    parser.add_argument("--dis_loss_weight", type=float, default=1.0)
    parser.add_argument("--com_loss_weight", type=float, default=1.0)
    parser.add_argument("--gen_loss_weight", type=float, default=1.0)
    parser.add_argument("--class_pseudo_loss_weight", type=float, default=1.0)
    parser.add_argument("--dis_pseudo_loss_weight", type=float, default=1.0)
    parser.add_argument("--com_pseudo_loss_weight", type=float, default=1.0)
    parser.add_argument("--cca_pseudo_loss_weight", type=float, default=0.0)
    parser.add_argument("--metric_for_best_model", type=str, default="kmeans_mlp_K-ACC")
    parser.add_argument("--dataset_batch_size", type=int, default=100000)
    return parser


def apply_yaml_config(args, config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ds_cfgs = cfg.pop("dataset_specific_configs", {})
    defaults = {a.dest: a.default for a in get_bolt_parser()._actions if hasattr(a, 'dest')}
    for k, v in cfg.items():
        k_norm = k.replace("-", "_")
        if hasattr(args, k_norm) and v is not None:
            if getattr(args, k_norm) == defaults.get(k_norm):
                setattr(args, k_norm, v)

    if args.dataset in ds_cfgs:
        for k, v in ds_cfgs[args.dataset].items():
            k_norm = k.replace("-", "_")
            if hasattr(args, k_norm) and v is not None:
                setattr(args, k_norm, v)

    return args


def prepare_data_dir(args, bolt_data_dir, known_labels_file):
    """Prepare a data directory in LLM4OpenSSL format from BOLT data."""
    ds = args.dataset
    src = os.path.join(bolt_data_dir, ds)
    dst = os.path.join(args.output_dir, "bolt_data", ds)

    # Create dirs
    os.makedirs(os.path.join(dst, "label"), exist_ok=True)
    os.makedirs(os.path.join(dst, "labeled_data"), exist_ok=True)

    # Symlink TSV files (LLM4OpenSSL reads {ds}/train.tsv, dev.tsv, test.tsv at root)
    for split in ["train", "dev", "test"]:
        dst_tsv = os.path.join(dst, f"{split}.tsv")
        if not os.path.exists(dst_tsv):
            src_tsv = os.path.join(src, "origin_data", f"{split}.tsv")
            if os.path.exists(src_tsv):
                os.symlink(os.path.abspath(src_tsv), dst_tsv)

    # Symlink label.list
    dst_label_list = os.path.join(dst, "label", "label.list")
    if not os.path.exists(dst_label_list):
        src_label_list = os.path.join(src, "label", "label.list")
        if os.path.exists(src_label_list):
            os.symlink(os.path.abspath(src_label_list), dst_label_list)

    # Create label_{rate}.list from BOLT fold-based known labels
    rate_str = str(args.known_cls_ratio)
    dst_rate_list = os.path.join(dst, "label", f"label_{rate_str}.list")
    if not os.path.exists(dst_rate_list):
        shutil.copy2(known_labels_file, dst_rate_list)

    # Generate labeled_data/train_{ratio}.tsv and dev_{ratio}.tsv
    # BOLT format: labeled_data/{ratio}/train.tsv with columns (label, labeled)
    # LLM4OpenSSL format: labeled_data/train_{ratio}.tsv with columns (text, label)
    ratio_str = str(args.labeled_ratio)
    for split in ["train", "dev"]:
        dst_labeled = os.path.join(dst, "labeled_data", f"{split}_{ratio_str}.tsv")
        if os.path.exists(dst_labeled):
            continue
        bolt_labeled_path = os.path.join(src, "labeled_data", ratio_str, f"{split}.tsv")
        full_tsv_path = os.path.join(src, "origin_data", f"{split}.tsv")
        if os.path.exists(bolt_labeled_path) and os.path.exists(full_tsv_path):
            full_df = pd.read_csv(full_tsv_path, sep="\t")
            bolt_df = pd.read_csv(bolt_labeled_path, sep="\t")
            mask = bolt_df["labeled"].astype(str).str.strip().str.lower() == "true"
            labeled_df = full_df[mask.values].reset_index(drop=True)
            labeled_df.to_csv(dst_labeled, sep="\t", index=False)
        elif os.path.exists(full_tsv_path):
            # If no labeled_data split, use all data
            shutil.copy2(full_tsv_path, dst_labeled)

    # Create data_statics.json
    dst_statics = os.path.join(args.output_dir, "bolt_data", "data_statics.json")
    if not os.path.exists(dst_statics):
        src_statics = os.path.join(bolt_data_dir, "data_statics.json")
        if os.path.exists(src_statics):
            os.symlink(os.path.abspath(src_statics), dst_statics)

    return os.path.join(args.output_dir, "bolt_data")


def build_json_config(args, data_root_dir, known_labels_file):
    """Build LLM4OpenSSL-compatible JSON config."""
    default_config_path = os.path.join(os.path.dirname(__file__), "configs", "args.json")
    with open(default_config_path, "r") as f:
        config = json.load(f)

    metric_dir = os.path.join(args.output_dir, "metrics")
    vector_dir = os.path.join(args.output_dir, "vectors")
    generate_dir = os.path.join(args.output_dir, "generate")
    logs_dir = os.path.join(args.output_dir, "logs")
    pretrain_output_dir = os.path.join(args.output_dir, "pretrain")

    for d in [metric_dir, vector_dir, logs_dir, pretrain_output_dir]:
        os.makedirs(d, exist_ok=True)

    config["model_name_or_path"] = args.model_name_or_path
    config["dataset_name"] = args.dataset
    config["data_root_dir"] = data_root_dir
    config["rate"] = args.known_cls_ratio
    config["labeled_ratio"] = args.labeled_ratio
    config["output_dir"] = args.output_dir
    config["metric_dir"] = metric_dir
    config["vector_dir"] = vector_dir
    config["generate_dir"] = generate_dir
    config["logs_dir"] = logs_dir
    config["pretrain_output_dir"] = pretrain_output_dir
    config["known_labels_file"] = known_labels_file
    config["seed"] = args.seed
    config["mode"] = "train"
    config["per_device_train_batch_size"] = args.per_device_train_batch_size
    config["per_device_eval_batch_size"] = args.per_device_eval_batch_size
    config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    config["num_train_epochs"] = args.num_train_epochs
    config["learning_rate"] = args.learning_rate
    config["linear_learning_rate"] = args.linear_learning_rate
    config["num_semi_warmup_epochs"] = args.num_semi_warmup_epochs
    config["num_gen_warmup_epochs"] = args.num_gen_warmup_epochs
    config["is_semi"] = args.is_semi
    config["is_mlp"] = args.is_mlp
    config["cca_loss_func"] = args.cca_loss_func
    config["cca_k"] = args.cca_k
    config["cca_loss_weight"] = args.cca_loss_weight
    config["class_loss_weight"] = args.class_loss_weight
    config["dis_loss_weight"] = args.dis_loss_weight
    config["com_loss_weight"] = args.com_loss_weight
    config["gen_loss_weight"] = args.gen_loss_weight
    config["class_pseudo_loss_weight"] = args.class_pseudo_loss_weight
    config["dis_pseudo_loss_weight"] = args.dis_pseudo_loss_weight
    config["com_pseudo_loss_weight"] = args.com_pseudo_loss_weight
    config["cca_pseudo_loss_weight"] = args.cca_pseudo_loss_weight
    config["metric_for_best_model"] = args.metric_for_best_model
    config["dataset_batch_size"] = args.dataset_batch_size
    config["load_best_model_at_end"] = True
    config["eval_strategy"] = "epoch"
    config["save_strategy"] = "epoch"
    config["remove_unused_columns"] = False
    config["logit_adjustent"] = "default"
    config["num_iters_sk"] = 3
    config["epsilon_sk"] = 0.1
    config["imb_factor"] = 1.0
    config["num_return_sequences"] = 4
    config["num_labels"] = 0  # will be set from data_statics.json

    # Set num_labels from data_statics.json
    statics_path = os.path.join(data_root_dir, "data_statics.json")
    if os.path.exists(statics_path):
        with open(statics_path) as f:
            statics = json.load(f)
        if args.dataset in statics and "num_labels" in statics[args.dataset]:
            config["num_labels"] = statics[args.dataset]["num_labels"]

    return config, metric_dir


def extract_gcd_metrics(metric_dir, mode="eval-test"):
    """Extract ACC, K-ACC, N-ACC, H-Score, ARI, NMI from LLM4OpenSSL results."""
    file_name = f"{mode}_3_0.1_1.0"  # default: num_iters_sk=3, epsilon_sk=0.1, imb_factor=1.0

    csv_path = os.path.join(metric_dir, f"{file_name}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, sep="\t")
        # Look for kmeans_mlp row (primary metric)
        row = df[df["method"] == "kmeans_mlp"]
        if row.empty:
            row = df.iloc[:1]
        row = row.iloc[0]
        return {
            "ACC": row.get("ACC"),
            "H-Score": row.get("H-Score"),
            "K-ACC": row.get("K-ACC"),
            "N-ACC": row.get("N-ACC"),
            "ARI": row.get("ARI"),
            "NMI": row.get("NMI"),
        }

    json_path = os.path.join(metric_dir, f"{file_name}.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            raw = json.load(f)
        return {
            "ACC": raw.get(f"{mode}_kmeans_mlp_ACC"),
            "H-Score": raw.get(f"{mode}_kmeans_mlp_H-Score"),
            "K-ACC": raw.get(f"{mode}_kmeans_mlp_K-ACC"),
            "N-ACC": raw.get(f"{mode}_kmeans_mlp_N-ACC"),
            "ARI": raw.get(f"{mode}_kmeans_mlp_ARI"),
            "NMI": raw.get(f"{mode}_kmeans_mlp_NMI"),
        }

    return None


def save_bolt_results(args, metrics):
    os.makedirs(args.save_results_path, exist_ok=True)
    results_file = os.path.join(args.save_results_path, "results.csv")

    row = {
        "method": "LLM4OpenSSL",
        "dataset": args.dataset,
        "known_cls_ratio": args.known_cls_ratio,
        "labeled_ratio": args.labeled_ratio,
        "cluster_num_factor": 1.0,
        "seed": args.seed,
        "ACC": round(metrics["ACC"], 2) if metrics.get("ACC") is not None else "",
        "H-Score": round(metrics["H-Score"], 2) if metrics.get("H-Score") is not None else "",
        "K-ACC": round(metrics["K-ACC"], 2) if metrics.get("K-ACC") is not None else "",
        "N-ACC": round(metrics["N-ACC"], 2) if metrics.get("N-ACC") is not None else "",
        "ARI": round(metrics["ARI"], 2) if metrics.get("ARI") is not None else "",
        "NMI": round(metrics["NMI"], 2) if metrics.get("NMI") is not None else "",
        "args": json.dumps(vars(args), default=str),
    }

    header_needed = not os.path.exists(results_file)
    with open(results_file, "a") as f:
        if header_needed:
            f.write(",".join(row.keys()) + "\n")
        f.write(",".join(str(v) for v in row.values()) + "\n")
    print(f"Results saved to {results_file}")


def main():
    parser = get_bolt_parser()
    args = parser.parse_args()

    if args.config:
        args = apply_yaml_config(args, args.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["WANDB_DISABLED"] = "true"

    # Resolve data paths
    bolt_data_dir = os.path.abspath("data")
    known_labels_file = os.path.join(
        bolt_data_dir,
        args.dataset,
        "label",
        f"{args.fold_type}{args.fold_num}",
        f"part{args.fold_idx}",
        f"label_known_{args.known_cls_ratio}.list",
    )
    if not os.path.exists(known_labels_file):
        print(f"ERROR: Known labels file not found: {known_labels_file}")
        sys.exit(1)

    # Prepare LLM4OpenSSL-compatible data directory
    data_root_dir = prepare_data_dir(args, bolt_data_dir, known_labels_file)

    # Build JSON config
    config, metric_dir = build_json_config(args, data_root_dir, known_labels_file)

    config_path = os.path.join(args.output_dir, "bolt_config.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Add project dir to path
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    from transformers import HfArgumentParser, TrainingArguments
    from init_parameters import ModelArguments, DataTrainingArguments

    hf_parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # Run train
    model_args, data_args, training_args = hf_parser.parse_json_file(json_file=config_path)
    from main import main as ssl_main
    ssl_main(model_args, data_args, training_args)

    # Run eval-test
    config["mode"] = "eval-test"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    model_args, data_args, training_args = hf_parser.parse_json_file(json_file=config_path)
    ssl_main(model_args, data_args, training_args)

    # Extract and save results
    metrics = extract_gcd_metrics(metric_dir, mode="eval-test")
    if metrics is not None:
        save_bolt_results(args, metrics)
        print(f"LLM4OpenSSL results: {metrics}")
    else:
        print("WARNING: Could not extract metrics from LLM4OpenSSL output")


if __name__ == "__main__":
    main()
