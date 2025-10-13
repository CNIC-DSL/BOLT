import numpy as np
import os
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
import json


def save_results(method_name, args, results, num_labels):
    config_to_save = {
        "dataset": args.dataset,
        "method": method_name,
        "known_cls_ratio": args.known_cls_ratio,
        "labeled_ratio": args.labeled_ratio,
        "cluster_num_factor": args.cluster_num_factor,
        "seed": args.seed,
    }

    full_results = {**config_to_save, **results}
    full_results["args"] = json.dumps(vars(args), ensure_ascii=False)

    desired_order = [
        "method",
        "dataset",
        "known_cls_ratio",
        "labeled_ratio",
        "cluster_num_factor",
        "seed",
        "ACC",
        "H-Score",
        "K-ACC",
        "N-ACC",
        "ARI",
        "NMI",
        "args",
    ]
    full_results = {i: full_results[i] for i in desired_order}

    save_path = args.save_results_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    results_file = os.path.join(save_path, "results.csv")

    new_df = pd.DataFrame([full_results])

    if not os.path.exists(results_file):
        new_df.to_csv(results_file, index=False)
    else:
        new_df.to_csv(results_file, mode="a", header=False, index=False)

    print(f"Results successfully saved to {results_file}")


def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w


def clustering_accuracy_score(y_true, y_pred, known_lab):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    ind_map = {j: i for i, j in ind}

    old_acc = 0
    total_old_instances = 0
    for i in known_lab:

        if i in ind_map:
            old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])

    if total_old_instances == 0:
        old_acc = 0.0
    else:
        old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in range(len(np.unique(y_true))):
        if i not in known_lab:

            if i in ind_map:
                new_acc += w[ind_map[i], i]
            total_new_instances += sum(w[:, i])

    if total_new_instances == 0:
        new_acc = 0.0
    else:
        new_acc /= total_new_instances

    if old_acc == 0 or new_acc == 0:
        h_score = 0.0
    else:
        h_score = 2 * old_acc * new_acc / (old_acc + new_acc)

    metrics = {
        "ACC": round(acc * 100, 2),
        "H-Score": round(h_score * 100, 2),
        "K-ACC": round(old_acc * 100, 2),
        "N-ACC": round(new_acc * 100, 2),
    }

    return metrics


def clustering_score(y_true, y_pred, known_lab):
    metrics = clustering_accuracy_score(y_true, y_pred, known_lab)
    metrics["ARI"] = round(adjusted_rand_score(y_true, y_pred) * 100, 2)
    metrics["NMI"] = round(normalized_mutual_info_score(y_true, y_pred) * 100, 2)
    return metrics
