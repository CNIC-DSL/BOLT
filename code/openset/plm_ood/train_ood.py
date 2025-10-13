import os
import copy
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve
from src.pytorch_ood.detector import (
    EnergyBased,
    Entropy,
    KLMatching,
    Mahalanobis,
    MaxLogit,
    MaxSoftmax,
    ViM,
    OpenMax,
    TemperatureScaling,
    ASH,
    SHE,
    LogitNorm,
)
from src.pytorch_ood.utils import OODMetrics, fix_random_seed, custom_metrics
import numpy as np
import logging
import sys
import yaml
import json

from model import Model
from load_dataset import load_and_prepare_datasets

from configs import create_parser, finalize_config


def run_ood_evaluation(args):

    logging.info("Loading and preparing datasets...")

    data = load_and_prepare_datasets(args)
    tokenizer = data["tokenizer"]

    test_loader = data["test_loader"]
    loader_in_train = data["loader_in_train"]
    logging.info("Datasets loaded successfully.")

    fix_random_seed(args.seed)
    model = Model(args, tokenizer=tokenizer).to(args.device)

    with torch.no_grad():
        model.eval()
        if not os.path.exists(f"{args.vector_path}/logits.npy"):
            logging.info("Calculating logits, predictions, and features...")
            preds, golds, logits, features = [], [], [], []

            for batch in tqdm(test_loader, desc="Inference"):
                y = batch["labels"].to(args.device)
                batch = {
                    i: v.to(args.device) for i, v in batch.items() if i != "labels"
                }
                logit = model(batch)
                feature = model.features(batch)
                pred = logit.max(dim=1).indices

                preds.append(pred)
                logits.append(logit)
                golds.append(y)
                features.append(feature)

            logits = torch.concat(logits).detach().to(torch.float32).cpu().numpy()
            preds = torch.concat(preds).detach().cpu().numpy()
            golds = torch.concat(golds).detach().cpu().numpy()
            features = torch.concat(features).detach().to(torch.float32).cpu().numpy()

            np.save(f"{args.vector_path}/logits.npy", logits)
            np.save(f"{args.vector_path}/preds.npy", preds)
            np.save(f"{args.vector_path}/golds.npy", golds)
            np.save(f"{args.vector_path}/features.npy", features)
        else:
            logging.info("Loading pre-calculated logits, predictions, and features...")
            logits = np.load(f"{args.vector_path}/logits.npy")
            preds = np.load(f"{args.vector_path}/preds.npy")
            golds = np.load(f"{args.vector_path}/golds.npy")
            features = np.load(f"{args.vector_path}/features.npy")

        ID_metrics = custom_metrics(preds, golds)
        logging.info(f"Test Accuracy: {ID_metrics['macro avg']}")

    logging.info("STAGE 2: Creating OOD Detectors")
    detectors = {
        "TemperatureScaling": TemperatureScaling(model),
        "LogitNorm": LogitNorm(model),
        "OpenMax": OpenMax(model),
        "Entropy": Entropy(model),
        "Mahalanobis": Mahalanobis(model.features, eps=0.0),
        "KLMatching": KLMatching(model),
        "MaxSoftmax": MaxSoftmax(model),
        "EnergyBased": EnergyBased(model),
        "MaxLogit": MaxLogit(model),
    }

    logging.info(f"> Fitting {len(detectors)} detectors")
    for name, detector in detectors.items():
        logging.info(f"--> Fitting {name}")

        detector.fit(loader_in_train, device=args.device)

    logging.info(f"STAGE 3: Evaluating {len(detectors)} detectors.")
    results = []
    with torch.no_grad():
        for detector_name, detector in detectors.items():
            logging.info(f"> Evaluating {detector_name}")
            metrics = OODMetrics()
            scores = []

            for batch in tqdm(test_loader, desc=f"Evaluating {detector_name}"):
                y = batch["labels"].to(args.device)
                batch = {
                    i: v.to(args.device) for i, v in batch.items() if i != "labels"
                }
                score = detector(batch)
                metrics.update(score, y)
                scores.append(score)

            r = {"Detector": detector_name}
            r.update(metrics.compute())
            scores = torch.concat(scores).detach().cpu().numpy()
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min())

            if detector_name == "Vim":
                new_logits = np.concatenate(
                    [logits, scores.reshape(scores.shape[0], 1)], axis=1
                )
                new_preds = new_logits.argmax(axis=1)
                new_preds[new_preds == logits.shape[-1]] = -1
                preds = new_preds
                r.update(custom_metrics(preds, golds, norm_scores))
                final_preds = copy.deepcopy(preds)
            else:
                r.update(custom_metrics(preds, golds, norm_scores))
                final_preds = copy.deepcopy(preds)
                final_preds[norm_scores > 0.5] = -1

            results.append(r)
            np.save(args.case_path + f"/{detector_name}_preds.npy", final_preds)
            np.save(args.case_path + f"/{detector_name}_golds.npy", golds)
            np.save(args.case_path + f"/{detector_name}_features.npy", features)

    df_to_save = pd.DataFrame(results)
    df_to_save["method"] = "PLM_OOD"
    df_to_save["K-F1"] = df_to_save["K-f1"]
    df_to_save["N-F1"] = df_to_save["N-f1"]
    df_to_save["F1"] = df_to_save["f1-score"]
    df_to_save["ACC"] = df_to_save["accuracy"]

    def func(args, detecor):
        args["Detecor"] = detecor
        return json.dumps(args)

    df_to_save["args"] = df_to_save["Detector"].apply(lambda x: func(vars(args), x))
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

    os.makedirs(os.path.dirname(args.metric_file), exist_ok=True)
    if not os.path.exists(args.metric_file):
        df_to_save.to_csv(args.metric_file, index=False)
    else:
        pd.concat(
            [pd.read_csv(args.metric_file), df_to_save], ignore_index=True
        ).to_csv(args.metric_file, index=False)


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

    run_ood_evaluation(config_args)
