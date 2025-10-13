import copy
import random
from typing import Tuple, List, Dict, Any, Optional

import torch
import torch.nn.functional as F
import os.path
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

import utils
from utils import F_measure, tensor2numpy


class Evaluator:
    METHOD = "NONE"

    def __init__(
        self,
        model: torch.nn.Module,
        root: str,
        file_postfix: str,
        dataset_name: str,
        device: torch.device,
        num_labels: int,
        tuning: str,
        scale_ind: float,
        scale_ood: float,
        valid_all_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        model_forward_cache: Optional[
            Dict[DataLoader, Tuple[torch.Tensor, ...]]
        ] = None,
    ):
        self.dataset_name = dataset_name
        self.valid_all_dataloader = valid_all_dataloader
        self.valid_dataloader = valid_dataloader
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.model = model
        self.tuning = tuning
        self.tpr = 95
        self.scale_ind = scale_ind
        self.scale_ood = scale_ood
        self.device = device

        output_file = os.path.join(root, f"{self.METHOD}_{file_postfix}")
        self.output_file = output_file

        self.num_labels = num_labels

        self.num_labels_IND = self.num_labels - 1

        if model_forward_cache is None:
            self.model_forward_cache = {}
        else:
            self.model_forward_cache = model_forward_cache

    def model_forward_with_cache(
        self, dataloader: DataLoader
    ) -> Tuple[torch.Tensor, ...]:
        if dataloader in self.model_forward_cache:

            return tuple(
                (item.clone().to(self.device))
                for item in self.model_forward_cache[dataloader]
            )

        result = self.model_forward(dataloader)

        self.model_forward_cache[dataloader] = tuple(
            (item.clone().cpu()) for item in result
        )
        return result

    def model_forward(
        self, dataloader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        total_feats = None

        total_logits = None

        total_labels = None
        self.model.eval()
        with torch.no_grad():
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                " + ",
                TimeRemainingColumn(),
            ) as progress:
                len_dataloader = len(dataloader)
                epoch_tqdm = progress.add_task(
                    description="epoch progress", total=len(dataloader)
                )
                for step, batch in enumerate(dataloader, start=1):
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(self.device)
                    labels = batch["labels"]
                    feats, logits = self.model(batch, mode="eval")
                    feats, logits = feats.double(), logits.double()

                    if feats.dim() == 2 and logits.dim() == 2:
                        feats = feats.unsqueeze(dim=0)
                        logits = logits.unsqueeze(dim=0)

                    if total_feats is not None:
                        total_feats = torch.cat((total_feats, feats), dim=1)
                        total_logits = torch.cat((total_logits, logits), dim=1)
                        total_labels = torch.cat((total_labels, labels))
                    else:
                        total_feats = feats
                        total_logits = logits
                        total_labels = labels

                    progress.update(
                        epoch_tqdm,
                        advance=1,
                        description=f"test_Evaluator - {step:03d}/{len_dataloader:03d}",
                    )

        return total_feats, total_logits, total_labels

    def search_threshold_by_valid_all(
        self, scores: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor
    ) -> torch.Tensor:
        thresholds = []
        for layer_index, cur_layer_scores in enumerate(scores):

            scores_ind = cur_layer_scores[labels != self.num_labels_IND]

            left: torch.Tensor = sorted(scores_ind)[
                round(len(scores_ind) * (1 - 0.96))
            ].cpu()
            right: torch.Tensor = sorted(scores_ind)[
                round(len(scores_ind) * (1 - 0.7))
            ].cpu()

            cur_layer_preds = preds[layer_index]

            best_f1 = -1
            best_acc = -1
            best_threshold = -1
            for threshold in np.linspace(left, right, 400):

                new_pred = copy.deepcopy(cur_layer_preds)
                new_pred[cur_layer_scores < threshold] = self.num_labels_IND

                res = F_measure(new_pred, labels)
                if res["F1"] > best_f1 and res["ACC-all"] > best_acc:
                    best_threshold = threshold
                    best_f1 = res["F1"]
                    best_acc = res["ACC-all"]
            thresholds.append(best_threshold)

        return torch.tensor(thresholds).to(self.device)

    def search_threshold_by_valid(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                sorted(cur_layer_scores)[
                    round(len(cur_layer_scores) * (1 - self.tpr * 0.01))
                ]
                for cur_layer_scores in scores
            ]
        )

    def predict_ind_labels(
        self, dataloader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def get_ensemble_pred(
        self, total_preds: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        t_ind = self.scale_ind
        t_ood = self.scale_ood
        k = 0.75
        n_layers, n = total_preds.shape

        base = [t_ind] * self.num_labels_IND
        base.append(t_ood)
        esm_pred = [-1] * n
        exit_layer_index = [-1] * n
        for data_index in range(n):

            preds = total_preds[:, data_index]
            vote = [0] * (self.num_labels_IND + 1)
            for layer_index, label in enumerate(preds):
                vote[label] += 1
                if vote[label] >= base[label] * pow(layer_index + 1, k):
                    esm_pred[data_index] = label
                    exit_layer_index[data_index] = layer_index + 1
                    break
                elif layer_index == n_layers - 1:
                    esm_pred[data_index] = label
                    exit_layer_index[data_index] = layer_index + 1

        esm_pred = torch.tensor(esm_pred).to(self.device)
        speed_up = n_layers / np.mean(exit_layer_index)
        return esm_pred, speed_up

    def get_pabee_pred(self, total_preds: torch.Tensor) -> Tuple[torch.Tensor, float]:
        patience = 4
        layers_nums, n = total_preds.shape

        esm_pred = [-1] * n
        exit_layer_index = [-1] * n
        for data_index in range(n):

            preds = total_preds[:, data_index]
            last_label = preds[0]
            count = 0
            for layer_index, label in enumerate(preds):
                count = count + 1 if label == last_label else 0
                if count >= patience:
                    esm_pred[data_index] = label
                    exit_layer_index[data_index] = layer_index + 1
                    break
                elif layer_index == layers_nums - 1:
                    esm_pred[data_index] = label
                    exit_layer_index[data_index] = layer_index + 1

        esm_pred = torch.tensor(esm_pred).to(self.device)
        pabee_speedup = layers_nums / np.mean(exit_layer_index)
        return esm_pred, pabee_speedup

    def get_random_pred(self, total_preds: torch.Tensor) -> Tuple[torch.Tensor, float]:
        n_layers, n = total_preds.shape

        esm_pred = [-1] * n
        exit_layer_index = [-1] * n
        for data_index in range(n):

            preds = total_preds[:, data_index]

            choosen_layer_index = random.randrange(0, n_layers)
            esm_pred[data_index] = preds[choosen_layer_index]
            exit_layer_index[data_index] = choosen_layer_index + 1

        esm_pred = torch.tensor(esm_pred).to(self.device)
        speed_up = n_layers / np.mean(exit_layer_index)
        return esm_pred, speed_up

    def get_threshold_score(self, mode: str) -> torch.Tensor:
        dataloader: DataLoader = (
            self.valid_all_dataloader if mode == "valid_all" else self.valid_dataloader
        )

        valid_score, valid_pred, valid_labels = self.predict_ind_labels(dataloader)

        if mode is "valid_all":

            return self.search_threshold_by_valid_all(
                valid_score, valid_labels, valid_pred
            )
        else:

            return self.search_threshold_by_valid(valid_score)

    def eval(self):

        test_scores, test_preds, test_labels = self.predict_ind_labels(
            self.test_dataloader
        )

        threshold_score = self.get_threshold_score(self.tuning)

        test_preds[test_scores < threshold_score.view(-1, 1)] = self.num_labels_IND

        final_pred, _ = self.get_pabee_pred(test_preds)

        return final_pred.cpu().numpy(), test_labels.cpu().numpy()

    def auc(self):

        test_scores, _, test_labels = self.predict_ind_labels(self.test_dataloader)

        print(f"{self.METHOD}: ")
        for layer_index in range(len(test_scores)):
            results = {
                "METRIC": self.METHOD,
                "tuning": self.tuning,
                "layer": layer_index + 1,
                **utils.au_sklearn(
                    self.num_labels_IND,
                    y_true=utils.tensor2numpy(test_labels),
                    y_prob=utils.tensor2numpy(test_scores[layer_index]),
                ),
            }

        print("#" * 80)


class KnnEvaluator(Evaluator):
    METHOD = "knn"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.knn_neighs = self._cal_knn_basic()

    def _cal_knn_basic(self) -> List[NearestNeighbors]:
        train_feats, _, _ = self.model_forward_with_cache(self.train_dataloader)

        def cal_by_layer(feats) -> NearestNeighbors:
            feats = utils.tensor2numpy(feats)
            neigh = NearestNeighbors(n_neighbors=5, n_jobs=4)
            neigh.fit(feats)
            return neigh

        return [cal_by_layer(feats) for feats in train_feats]

    def _get_score(self, total_feats: torch.Tensor) -> torch.Tensor:
        total_score = []
        for layer_index in range(len(total_feats)):
            dist, _ = self.knn_neighs[layer_index].kneighbors(
                tensor2numpy(total_feats[layer_index]), n_neighbors=5
            )
            total_score.append(dist[:, -1])

        return -torch.tensor(np.array(total_score)).to(self.device)

    def predict_ind_labels(
        self, dataloader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        feats, logits, ground_truth_labels = self.model_forward_with_cache(dataloader)
        scores = self._get_score(feats)
        _, pred = torch.max(logits, dim=-1)
        return scores, pred, ground_truth_labels


class MspEvaluator(Evaluator):
    METHOD = "msp"

    def predict_ind_labels(
        self, dataloader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        _, logits, ground_truth_labels = self.model_forward_with_cache(dataloader)

        scores, pred = torch.max(F.softmax(logits, dim=-1), dim=-1)
        return scores, pred, ground_truth_labels


class MaxLogitEvaluator(Evaluator):
    METHOD = "maxLogit"

    def predict_ind_labels(
        self, dataloader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        _, logits, ground_truth_labels = self.model_forward_with_cache(dataloader)

        scores, pred = torch.max(logits, dim=-1)
        return scores, pred, ground_truth_labels


class EnergyEvaluator(Evaluator):
    METHOD = "energy"

    def __init__(self, temperature: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.T = temperature

    def predict_ind_labels(
        self, dataloader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        _, logits, ground_truth_labels = self.model_forward_with_cache(dataloader)

        _, pred = torch.max(logits, dim=-1)

        scores = torch.mul(torch.logsumexp(torch.div(logits, self.T), dim=-1), self.T)
        return scores, pred, ground_truth_labels


class EntropyEvaluator(Evaluator):
    METHOD = "entropy"

    def predict_ind_labels(
        self, dataloader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        _, logits, ground_truth_labels = self.model_forward_with_cache(dataloader)

        _, pred = torch.max(logits, dim=-1)

        probs = F.softmax(logits, dim=-1)
        scores = -torch.sum((-probs * torch.log(probs)), dim=-1)

        return scores, pred, ground_truth_labels


class OdinEvaluator(Evaluator):
    METHOD = "odin"

    def __init__(self, temperature: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.T = temperature

    def predict_ind_labels(
        self, dataloader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        _, logits, ground_truth_labels = self.model_forward_with_cache(dataloader)

        scores, pred = torch.max(F.softmax(torch.div(logits, self.T), dim=-1), dim=-1)
        return scores, pred, ground_truth_labels


class MahaEvaluator(Evaluator):
    METHOD = "maha"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.centroids, self.linv = self._cal_maha_basic()

    def _cal_maha_basic(self) -> Tuple[torch.Tensor, torch.Tensor]:

        train_feats, _, train_labels = self.model_forward_with_cache(
            self.train_dataloader
        )

        def cal_per_layer(feats: torch.Tensor, labels: torch.Tensor):

            centroids = torch.zeros(self.num_labels_IND, 768).to(self.device)
            class_count = [0] * self.num_labels_IND
            for i, feature in enumerate(feats):
                label = labels[i]
                centroids[label] += feature
                class_count[label] += 1
            centroids /= torch.tensor(class_count).float().unsqueeze(1).to(self.device)

            mu = centroids[labels]

            x_mu = feats - mu

            sigma = torch.matmul(x_mu.T, x_mu)
            epsilon = 1e-6
            sigma += (
                torch.eye(sigma.shape[0], device=sigma.device, dtype=sigma.dtype)
                * epsilon
            )

            linv = torch.linalg.inv(torch.linalg.cholesky(sigma))
            return centroids, linv

        result_centroids, result_linv = [], []
        for layer_index in range(len(train_feats)):
            cur_layer_centroids, cur_layer_linv = cal_per_layer(
                train_feats[layer_index], train_labels
            )
            result_centroids.append(cur_layer_centroids)
            result_linv.append(cur_layer_linv)
        return torch.stack(result_centroids), torch.stack(result_linv)

    def _get_score_pred(self, total_feats) -> Tuple[torch.Tensor, torch.Tensor]:
        def get_score_by_layer(layer_index: int) -> Tuple[torch.Tensor, torch.Tensor]:

            cur_layer_centroids = self.centroids[layer_index]
            cur_layer_linv = self.linv[layer_index]

            cur_layer_feats = total_feats[layer_index]

            n = cur_layer_feats.shape[0]
            num_labels_ind = cur_layer_centroids.shape[0]

            x = cur_layer_feats.unsqueeze(1).expand(
                n, num_labels_ind, -1
            ) - cur_layer_centroids.unsqueeze(0).expand(n, num_labels_ind, -1)

            x = torch.matmul(x, cur_layer_linv)

            d = -torch.sum(x**2, dim=-1)

            values, indices = torch.max(d, dim=-1)

            return values, indices

        print("ST")
        total_score, total_pred = [], []
        for layer_index_ in range(len(total_feats)):
            cur_layer_score, cur_layer_pred = get_score_by_layer(layer_index_)
            total_score.append(cur_layer_score)
            total_pred.append(cur_layer_pred)
        return torch.stack(total_score), torch.stack(total_pred)

    def predict_ind_labels(
        self, dataloader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        feats, _, ground_truth_labels = self.model_forward_with_cache(dataloader)
        scores, pred = self._get_score_pred(feats)
        return scores, pred, ground_truth_labels


class _LofEvaluator(Evaluator):
    METHOD = "lof"

    def __init__(self, distance_metric: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lof_adapters = self._cal_lof_basic(distance_metric)

    def _cal_lof_basic(self, metric: str) -> List[LocalOutlierFactor]:
        train_feats, _, _ = self.model_forward_with_cache(self.train_dataloader)

        def cal_by_layer(feats: torch.Tensor) -> LocalOutlierFactor:
            lof = LocalOutlierFactor(
                n_neighbors=20, metric=metric, novelty=True, n_jobs=4
            )
            lof.fit(tensor2numpy(feats))
            return lof

        return [cal_by_layer(feats) for feats in train_feats]

    def _get_score(self, total_feats: torch.Tensor) -> torch.Tensor:

        total_score = [
            torch.tensor(
                self.lof_adapters[layer_index].score_samples(
                    tensor2numpy(total_feats[layer_index])
                )
            )
            for layer_index in range(len(total_feats))
        ]
        return torch.stack(total_score).to(self.device)

    def predict_ind_labels(
        self, dataloader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        feats, logits, ground_truth_labels = self.model_forward_with_cache(dataloader)

        scores = self._get_score(feats)

        _, pred = torch.max(logits, dim=-1)
        return scores, pred, ground_truth_labels


class LofCosineEvaluator(_LofEvaluator):
    METHOD = "lof_cosine"

    def __init__(self, *args, **kwargs):
        super().__init__(distance_metric="cosine", *args, **kwargs)


class LofEuclideanEvaluator(_LofEvaluator):
    METHOD = "lof_euclidean"

    def __init__(self, *args, **kwargs):
        super().__init__(distance_metric="euclidean", *args, **kwargs)
