import copy
from typing import List, Dict, Any


import torch


import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


def harmonic_mean(data: List[float]) -> float:
    total = 0
    for i in data:
        if i == 0:
            return 0
        total += 1 / i
    return len(data) / total


def tensor2numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def au_sklearn(
    num_labels_ind: int, y_true: np.ndarray, y_prob: np.ndarray, round_control: int = 2
) -> Dict[str, Any]:

    y_true_ = copy.deepcopy(y_true)
    oos_mask = y_true_ == num_labels_ind
    y_true_[oos_mask] = 0
    y_true_[~oos_mask] = 1

    y_prob_ = y_prob

    aupr_oos = round(roc_auc_score(y_true_, y_prob_) * 100, round_control)

    aupr_in = round(average_precision_score(y_true_, y_prob_) * 100, round_control)

    y_true_ = copy.deepcopy(y_true)
    oos_mask = y_true_ == num_labels_ind
    y_true_[oos_mask] = 1
    y_true_[~oos_mask] = 0

    y_prob_ = -y_prob

    aupr_ood = round(average_precision_score(y_true_, y_prob_) * 100, round_control)

    return {"auroc": aupr_oos, "aupr_in": aupr_in, "aupr_ood": aupr_ood}


def F_measure(
    pred: torch.Tensor, labels: torch.Tensor, round_setting: int = 2
) -> Dict[str, np.float64]:
    y_pred = tensor2numpy(pred)
    y_true = tensor2numpy(labels)

    cm: np.ndarray = confusion_matrix(y_true, y_pred)

    acc = round(accuracy_score(y_true, y_pred) * 100, round_setting)

    recalls, precisions, f1s = [], [], []

    n_class = cm.shape[0]
    for index in range(n_class):
        tp = cm[index][index]

        recall = tp / cm[index].sum() if cm[index].sum() != 0 else 0

        precision = tp / cm[:, index].sum() if cm[:, index].sum() != 0 else 0

        f1 = (
            2 * recall * precision / (recall + precision)
            if (recall + precision) != 0
            else 0
        )

        recalls.append(recall * 100)
        precisions.append(precision * 100)
        f1s.append(f1 * 100)

    f1 = np.mean(f1s).round(round_setting)
    f1_seen = np.mean(f1s[:-1]).round(round_setting)
    f1_unseen = round(f1s[-1], round_setting)

    results = {"ACC-all": acc, "F1": f1, "F1-ood": f1_unseen, "F1-ind": f1_seen}

    return results


def estimate_best_threshold(
    cur: Dict[str, float], best_results: Dict[str, float], strategy: str
) -> bool:
    assert strategy in ["ALL", "SUM", "HARMONIC"]

    if len(best_results) == 0:
        return True

    best_metrics_names = ["F1_IND", "F1_OOD"]
    if strategy == "ALL":
        return all(
            cur[metric_name] >= best_results[metric_name]
            for metric_name in best_metrics_names
        )

    if strategy == "SUM":
        return sum(cur[metric_name] for metric_name in best_metrics_names) >= sum(
            best_results[metric_name] for metric_name in best_metrics_names
        )

    assert strategy == "HARMONIC"
    return harmonic_mean(
        [cur[metric_name] for metric_name in best_metrics_names]
    ) >= harmonic_mean(
        [best_results[metric_name] for metric_name in best_metrics_names]
    )
