import os
import json
import argparse
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
import pandas as pd
import yaml
import sys


from utils import set_seed, Split
from ADBThreshold import ADBThreshold
from ood_train import evaluate


def get_embed_function(emb_name):
    if emb_name == "use_dan":
        return hub.load(
            "plm/universal-sentence-encoder-tensorflow2-universal-sentence-encoder-v2"
        )
    elif emb_name == "use_tran":
        return hub.load("plm/universal-sentence-encoder-tensorflow2-large-v2")
    elif emb_name == "sbert":
        return SentenceTransformer("./pretrained_models/stsb-roberta-base").encode
    else:
        raise ValueError(f"Unknown embedding name: {emb_name}")


def main(args):

    set_seed(args.seed)

    print("Loading standardized data...")
    known_label_path = os.path.join(
        args.data_dir,
        args.dataset,
        "label",
        f"{args.fold_type}{args.fold_num}",
        f"part{args.fold_idx}",
        f"label_known_{args.known_cls_ratio}.list",
    )
    seen_labels = pd.read_csv(known_label_path, header=None)[0].tolist()

    origin_train_path = os.path.join(
        args.data_dir, args.dataset, "origin_data", "train.tsv"
    )
    labeled_train_path = os.path.join(
        args.data_dir,
        args.dataset,
        "labeled_data",
        str(args.labeled_ratio),
        "train.tsv",
    )
    test_path = os.path.join(args.data_dir, args.dataset, "origin_data", "test.tsv")

    origin_train_df = pd.read_csv(origin_train_path, sep="\t")
    labeled_train_df = pd.read_csv(labeled_train_path, sep="\t")
    df_test = pd.read_csv(test_path, sep="\t")

    df_train = labeled_train_df.copy()
    df_train["text"] = origin_train_df["text"]

    df_train_filtered = df_train[
        (df_train["label"].isin(seen_labels)) & (df_train["labeled"].astype(bool))
    ]
    train_data = [
        list(row)
        for row in df_train_filtered[["text", "label"]].itertuples(index=False)
    ]

    df_test.loc[~df_test["label"].isin(seen_labels), "label"] = "oos"
    test_data = [
        list(row) for row in df_test[["text", "label"]].itertuples(index=False)
    ]

    dataset = {"train": train_data, "test": test_data}
    print(
        f"Data loaded. Train samples: {len(train_data)}, Test samples: {len(test_data)}"
    )

    print(f"Loading embedding model: {args.emb_name}...")
    embed_f = get_embed_function(args.emb_name)

    model = ADBThreshold(alpha=args.alpha)
    model_name = type(model).__name__

    print("Starting evaluation...")
    results_dct = evaluate(dataset, model, model_name, embed_f, limit_num_sents=None)

    final_results = {}
    final_results["dataset"] = args.dataset
    final_results["seed"] = args.seed
    final_results["known_cls_ratio"] = args.known_cls_ratio
    final_results["emb_name"] = args.emb_name
    final_results["alpha"] = args.alpha

    final_results["ACC"] = results_dct.get("accuracy_all", 0.0)
    final_results["F1"] = results_dct.get("f1_all", 0.0)
    final_results["K-F1"] = results_dct.get("f1_id", 0.0)
    final_results["N-F1"] = results_dct.get("f1_ood", 0.0)

    final_results["args"] = json.dumps(vars(args), ensure_ascii=False)

    metric_dir = args.save_results_path
    os.makedirs(metric_dir, exist_ok=True)
    results_path = os.path.join(metric_dir, "results.csv")

    df_to_save = pd.DataFrame([final_results])
    df_to_save["method"] = "ab"
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
    for col in cols:
        if col in df_to_save:
            continue
        df_to_save[col] = getattr(args, col)
    df_to_save = df_to_save[cols]

    if not os.path.exists(results_path):
        df_to_save.to_csv(results_path, index=False)
    else:
        pd.concat([pd.read_csv(results_path), df_to_save], ignore_index=True).to_csv(
            results_path, index=False
        )

    print("\nResults have been saved to:", results_path)
    print("Appended new result row:")
    print(df_to_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", type=str, default=None, help="Path to the YAML config file."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--dataset", type=str, default="banking")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--known_cls_ratio", type=float, default=0.25)
    parser.add_argument("--labeled_ratio", type=float, default=1.0)
    parser.add_argument("--cluster_num_factor", type=float, default=1.0)
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--fold_num", type=int, default=5)
    parser.add_argument(
        "--emb_name",
        type=str,
        choices=["sbert", "use_dan", "use_tran"],
        default="sbert",
    )
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--output_dir", type=str, default="./outputs/openset/ab")

    parser.add_argument(
        "--save_results_path",
        default="./results",
        type=str,
        help="The metric directory where results and models will be written.",
    )
    parser.add_argument(
        "--fold_type",
        type=str,
        default="fold",
        help="",
        choices=["imbalance_fold", "fold"],
    )

    args = parser.parse_args()

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
            dataset_configs = yaml_config["dataset_specific_configs"].get(
                args.dataset, {}
            )
            apply_config_updates(args, dataset_configs, parser)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    main(args)
