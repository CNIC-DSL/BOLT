import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

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
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in range(len(np.unique(y_true))):
        if i not in known_lab:
            new_acc += w[ind_map[i], i]
            total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances

    h_score = 2*old_acc*new_acc / (old_acc + new_acc)

    return (round(acc*100, 2), round(old_acc*100, 2), round(new_acc*100, 2), round(h_score*100, 2))

def clustering_score(y_true, y_pred, known_lab):
    acc_all, acc_known, acc_novel, h_score = clustering_accuracy_score(y_true, y_pred, known_lab)

    return {
        'ACC_all': acc_all,
        'ACC_known': acc_known,
        'ACC_novel': acc_novel,
        'H-Score': h_score,
        'ARI': round(adjusted_rand_score(y_true, y_pred) * 100, 2),
        'NMI': round(normalized_mutual_info_score(y_true, y_pred) * 100, 2)
    }
