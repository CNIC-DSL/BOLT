import itertools
import subprocess
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import copy
import torch.nn.functional as F
import random
import csv
import sys
from torch import nn
from tqdm import tqdm_notebook, trange, tqdm
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import (
    WEIGHTS_NAME,
    CONFIG_NAME,
    BertPreTrainedModel,
    BertModel,
)
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix,
    normalized_mutual_info_score,
    adjusted_rand_score,
    accuracy_score,
)
from scipy.optimize import linear_sum_assignment
from sklearn import metrics


def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w


def clustering_accuracy_score(y_true, y_pred, known_lab):

    ind, w = hungray_aligment(y_true, y_pred)
    acc = float(sum(w[i, j] for i, j in ind)) / float(y_pred.size)

    ind_map = {j: i for i, j in ind}

    unique_true = set(np.unique(y_true))

    known_in_test = [i for i in known_lab if i in unique_true]

    new_in_test = [i for i in unique_true if i not in known_in_test]

    old_correct = 0.0
    total_old_instances = 0.0
    for i in known_in_test:

        old_correct += w[ind_map[i], i]
        total_old_instances += float(w[:, i].sum())
    old_acc = (old_correct / total_old_instances) if total_old_instances > 0 else 0.0

    new_correct = 0.0
    total_new_instances = 0.0
    for i in new_in_test:
        new_correct += w[ind_map[i], i]
        total_new_instances += float(w[:, i].sum())
    new_acc = (new_correct / total_new_instances) if total_new_instances > 0 else 0.0

    denom = old_acc + new_acc
    h_score = (2.0 * old_acc * new_acc / denom) if denom > 0 else 0.0

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
