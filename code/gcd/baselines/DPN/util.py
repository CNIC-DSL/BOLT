import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w


def clustering_accuracy_score(y_true, y_pred, known_lab):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum(w[i, j] for i, j in ind) / y_pred.size
    ind_map = {j: i for i, j in ind}

    unique_true = np.unique(y_true)

    old_correct = 0.0
    old_total = 0.0
    for cls in unique_true:
        if cls in known_lab:
            col_sum = float(np.sum(w[:, cls]))
            if col_sum > 0 and cls in ind_map:
                old_correct += w[ind_map[cls], cls]
                old_total += col_sum
    old_acc = (old_correct / old_total) if old_total > 0 else 0.0

    new_correct = 0.0
    new_total = 0.0
    for cls in unique_true:
        if cls not in known_lab:
            col_sum = float(np.sum(w[:, cls]))
            if col_sum > 0 and cls in ind_map:
                new_correct += w[ind_map[cls], cls]
                new_total += col_sum
    new_acc = (new_correct / new_total) if new_total > 0 else 0.0

    h_denom = old_acc + new_acc
    h_score = (2 * old_acc * new_acc / h_denom) if h_denom > 0 else 0.0

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
