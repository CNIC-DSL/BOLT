import random
import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import logging
import datetime
import json
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    confusion_matrix,
    normalized_mutual_info_score,
    adjusted_rand_score,
    accuracy_score,
)
from scipy.optimize import linear_sum_assignment


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def set_seed_v2(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_yaml_config(config_path):

    yaml_file = open(config_path, "r", encoding="utf-8")
    yaml_data = yaml_file.read()
    yaml_file.close()

    configs = yaml.load(yaml_data, Loader=yaml.FullLoader)
    return configs


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


def mask_tokens(inputs, tokenizer, special_tokens_mask=None, mlm_probability=0.15):
    labels = inputs.clone()

    probability_matrix = torch.full(labels.shape, mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    probability_matrix[torch.where(inputs == 0)] = 0.0
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels


class view_generator:
    def __init__(self, tokenizer, rtr_prob, seed):
        set_seed(seed)
        self.tokenizer = tokenizer
        self.rtr_prob = rtr_prob

    def random_token_replace(self, ids):
        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        ids, _ = mask_tokens(ids, self.tokenizer, mlm_probability=self.rtr_prob)
        random_words = torch.randint(len(self.tokenizer), ids.shape, dtype=torch.long)
        indices_replaced = torch.where(ids == mask_id)
        ids[indices_replaced] = random_words[indices_replaced]
        return ids

    def shuffle_tokens(self, ids):
        view_pos = []
        for inp in torch.unbind(ids):
            new_ids = copy.deepcopy(inp)
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(
                inp, already_has_special_tokens=True
            )
            sent_tokens_inds = np.where(np.array(special_tokens_mask) == 0)[0]
            inds = np.arange(len(sent_tokens_inds))
            np.random.shuffle(inds)
            shuffled_inds = sent_tokens_inds[inds]
            inp[sent_tokens_inds] = new_ids[shuffled_inds]
            view_pos.append(new_ids)
        view_pos = torch.stack(view_pos, dim=0)
        return view_pos


def save_model(args, model, epoch):
    model_path = args.output_dir
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_file = os.path.join(model_path, args.model_file_name + f"_epoch_{epoch}.pt")
    model_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }

    torch.save(model_dict, model_file)
    print(f"Model saved to {model_file}")


def save_results(args, test_results):

    config_to_save = {
        "method": args.method,
        "dataset": args.dataset,
        "known_cls_ratio": args.known_cls_ratio,
        "labeled_ratio": args.labeled_ratio,
        "cluster_num_factor": args.cluster_num_factor,
        "seed": args.seed,
        "K": args.num_labels,
    }

    full_results = {**config_to_save, **test_results}

    full_results["args"] = json.dumps(vars(args), ensure_ascii=False)

    if not os.path.exists(args.save_results_path):
        os.makedirs(args.save_results_path)
    results_path = os.path.join(args.save_results_path, args.results_file_name)

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

    new_df = pd.DataFrame([full_results])

    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
        df_to_save = new_df
    else:
        try:
            old_df = pd.read_csv(results_path)
            df_to_save = pd.concat([old_df, new_df], ignore_index=True)
        except Exception as e:
            print(f"Warning: Could not append to CSV due to {e}. Overwriting file.")
            df_to_save = new_df

    final_columns = [col for col in desired_order if col in df_to_save.columns]
    df_to_save = df_to_save[final_columns]

    df_to_save.to_csv(results_path, index=False)

    print(f"Results successfully saved to {results_path}")


def save_results_pter(args, test_results):

    pred_labels_path = os.path.join(args.output_dir, "y_pred.npy")
    np.save(pred_labels_path, test_results["y_pred"])
    true_labels_path = os.path.join(args.output_dir, "y_true.npy")
    np.save(true_labels_path, test_results["y_true"])

    del test_results["y_pred"]
    del test_results["y_true"]

    if not os.path.exists(args.save_results_path):
        os.makedirs(args.save_results_path)

    var = [
        args.dataset,
        args.method,
        args.known_cls_ratio,
        args.labeled_ratio,
        args.cluster_num_factor,
        args.seed,
    ]
    names = [
        "dataset",
        "method",
        "known_cls_ratio",
        "labeled_ratio",
        "cluster_num_factor",
        "seed",
    ]
    vars_dict = {k: v for k, v in zip(names, var)}
    results = dict(test_results, **vars_dict)
    results["args"] = json.dumps(vars(args), ensure_ascii=False)
    keys = list(results.keys())
    values = list(results.values())

    results_path = os.path.join(args.save_results_path, args.results_file_name)

    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori, columns=keys)
        df1.to_csv(results_path, index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results, index=[1])

        df1 = pd.concat([df1, new], ignore_index=True)
        df1.to_csv(results_path, index=False)
    data_diagram = pd.read_csv(results_path)

    print("test_results", data_diagram)
