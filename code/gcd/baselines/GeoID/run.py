from model import CLBert
from init_parameter import init_model
from dataloader import Data
from mtp import InternalPretrainModelManager
from utils.tools import *
from utils.memory import MemoryBank, fill_memory_bank
from utils.neighbor_dataset import NeighborsDataset
from utils.contrastive import *
from utils.sinkhorn_knopp import *
from utils.evaluate_utils import *
from utils.kmeans import K_Means as SemiKMeans
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from torch import nn
import numpy as np
import torch.nn as nn
import yaml
import sys
import json

torch.autograd.set_detect_anomaly(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CLNNModelManager:
    def __init__(self, args, data, pretrained_model=None):
        set_seed(args.seed)
        n_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = data.num_labels
        self.num_known = data.num_known
        self.model = CLBert(
            args.bert_model,
            device=self.device,
            label_num=data.num_labels,
            feat_dim=args.feat_dim,
        )

        if n_gpu > 1:
            self.model = nn.DataParallel(self.model)

        if not args.disable_pretrain:
            self.pretrained_model = pretrained_model
            if pretrained_model is not None:
                self.load_pretrained_model()

        self.freeze_parameters()

        self.num_train_optimization_steps = (
            int(len(data.train_semi_dataset) / args.train_batch_size)
            * args.num_train_epochs
        )
        self.num_train_optimization_warmer_steps = (
            int(data.num_labeled_samples / args.train_batch_size) * args.num_warm_epochs
        )

        self.optimizer, self.scheduler = self.get_optimizer(args)

        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
        self.generator = view_generator(self.tokenizer, args.rtr_prob, args.seed)
        self.feature_bank = torch.zeros(
            len(data.train_semi_dataset), int(args.feat_dim), device=self.device
        )
        self.centroids = None
        self.prototype = torch.zeros(
            self.num_labels, int(args.feat_dim), device=self.device
        )
        cur_M = self.model.classifier_new.ori_M.T.cuda()
        self.prototype[:, :] = cur_M[:, :]
        self.feature_bank = torch.zeros(
            len(data.train_semi_dataset), int(args.feat_dim), device=self.device
        )
        self.pesudo_bank = torch.zeros(
            (len(data.train_semi_dataset), self.num_labels), device=self.device
        )
        self.chosen_bank = np.zeros(len(data.train_semi_dataset))
        self.label_ids = torch.zeros(
            len(data.train_semi_dataset), dtype=torch.bool, device=self.device
        )
        self.cluster_label_bank = torch.zeros(
            len(data.train_semi_dataset), device=self.device
        )
        print("len(data.train_semi_dataset)", len(data.train_semi_dataset))
        self.k_top = int(5)

    def freeze_parameters(self):
        for name, param in self.model.backbone.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def get_neighbor_dataset(self, args, data, indices):
        dataset = NeighborsDataset(data.train_semi_dataset, indices)
        self.train_dataloader = DataLoader(
            dataset, batch_size=args.train_batch_size, shuffle=True
        )

    def get_pesudo_label(self, args, pesudo_labels, data, indices):
        data = TensorDataset(
            data.semi_input_ids,
            data.semi_input_mask,
            data.semi_segment_ids,
            pesudo_labels,
        )
        dataset = NeighborsDataset(data, indices)
        self.train_dataloader = DataLoader(
            dataset, batch_size=args.train_batch_size, shuffle=True
        )

    def get_neighbor_inds(self, args, data):
        memory_bank = MemoryBank(
            len(data.train_semi_dataset), args.feat_dim, len(data.all_label_list), 0.1
        )
        fill_memory_bank(data.train_semi_dataloader, self.model, memory_bank)
        indices = memory_bank.mine_nearest_neighbors(
            args.topk, calculate_accuracy=False
        )
        return memory_bank, indices

    def get_adjacency(self, args, inds, neighbors, targets):
        adj = torch.zeros(inds.shape[0], inds.shape[0])
        for b1, n in enumerate(neighbors):
            adj[b1][b1] = 1
            num_ng = 0
            for b2, j in enumerate(inds):
                if (
                    (targets[b1] == targets[b2])
                    and (targets[b1] >= 0)
                    and (targets[b2] >= 0)
                ):
                    adj[b1][b2] = 1
        return adj

    def evaluation(self, args, data, save_results=True, plot_cm=True):

        feats_test, labels = self.get_features_labels(
            data.test_dataloader, self.model, args
        )
        feats_test = feats_test.cpu().numpy()

        km = KMeans(n_clusters=self.num_labels).fit(feats_test)

        y_pred = km.labels_
        y_true = labels.cpu().numpy()

        results = clustering_score(y_true, y_pred, data.known_lab)
        print("results", results)

        if plot_cm:
            ind, _ = hungray_aligment(y_true, y_pred)
            map_ = {i[0]: i[1] for i in ind}
            y_pred = np.array([map_[idx] for idx in y_pred])

            cm = confusion_matrix(y_true, y_pred)
            print("confusion matrix", cm)
            self.test_results = results

        if save_results:
            self.save_results(args)
        return results["ACC"]

    def compute_kl_loss(self, p, q, pad_mask=None):
        p_loss = F.kl_div(
            F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction="none"
        )
        q_loss = F.kl_div(
            F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction="none"
        )

        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.0)
            q_loss.masked_fill_(pad_mask, 0.0)

        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    def supcon_knn(self, features, features_all, mask, temperature=0.07):
        assert (
            mask.shape[0] == features.shape[0]
            and mask.shape[1] == features_all.shape[0]
        )
        features_all = features_all.detach().clone()
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")
        mask = mask.float().detach().clone().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features_all.T), temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos.mean()
        return loss

    def alignment(self, prototype, centroids, logits):
        old_centroids = centroids.cpu().numpy()
        new_centroids = prototype.cpu().numpy()
        logits = logits.cpu().numpy()
        DistanceMatrix = np.linalg.norm(
            old_centroids[:, np.newaxis, :] - new_centroids[np.newaxis, :, :], axis=2
        )
        row_ind, col_ind = linear_sum_assignment(DistanceMatrix)
        list_ = list(col_ind)
        alignment_labels = list_
        logits = logits[:, alignment_labels]
        logits = torch.tensor(logits, dtype=torch.float32).to(self.device)

        return logits, alignment_labels

    def train(self, args, data):

        if isinstance(self.model, nn.DataParallel):
            criterion = self.model.loss_cl
        else:
            criterion = self.model.loss_cl

        bank, indices = self.get_neighbor_inds(args, data)
        bank_length = len(data.train_semi_dataset)
        self.get_neighbor_dataset(args, data, indices)

        best_res_cluster = 0

        best_score_scalar = float("-inf")
        wait = 0
        best_model = None
        monitor_name = getattr(args, "es_metric", "ACC")
        min_delta = getattr(args, "es_min_delta", 0.0)
        patience = getattr(args, "es_patience", 10)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            w = linear_rampup(epoch, 0, 0.3 * args.num_train_epochs)
            rho_l = 0.3 + (0.8 - 0.3) * linear_rampup(
                epoch, 0, args.num_train_epochs * 0.7
            )
            rho_n = 0.1 + (0.8 - 0.1) * linear_rampup(
                epoch, 0, args.num_train_epochs * 0.7
            )

            if epoch >= 1:
                km = SemiKMeans(
                    k=self.num_labels,
                    tolerance=1e-4,
                    max_iterations=200,
                    init="k-means++",
                    n_init=100,
                    random_state=None,
                    n_jobs=None,
                    pairwise_batch_size=2048,
                    mode=None,
                )
                all_features = self.feature_bank
                l_feats = all_features[self.label_ids]
                u_feats = all_features[~self.label_ids]
                l_targets = torch.argmax(self.pesudo_bank[self.label_ids], dim=1)
                km.fit_mix(u_feats, l_feats, l_targets)
                self.cluster_label_bank = km.labels_
                global ids_topk
                ids_topk = km.topk(self.k_top, all_features)

            for batch in tqdm(self.train_dataloader, desc="Iteration"):

                anchor = tuple(t.to(self.device) for t in batch["anchor"])
                neighbor = tuple(t.to(self.device) for t in batch["neighbor"])
                pos_neighbors = batch["possible_neighbors"]
                target = batch["target"]
                mask_labeled = target >= 0
                data_inds = batch["index"]

                X = {
                    "input_ids": anchor[0],
                    "attention_mask": anchor[1],
                    "token_type_ids": anchor[2],
                }

                X_an = {
                    "input_ids": self.generator.random_token_replace(
                        anchor[0].cpu()
                    ).to(self.device),
                    "attention_mask": anchor[1],
                    "token_type_ids": anchor[2],
                }
                X_ng = {
                    "input_ids": self.generator.random_token_replace(
                        neighbor[0].cpu()
                    ).to(self.device),
                    "attention_mask": neighbor[1],
                    "token_type_ids": neighbor[2],
                }

                with torch.set_grad_enabled(True):
                    logit_ = self.model(X)["logits"]
                    feature_ori = self.model(X)["features"]
                    out_dict_1 = self.model(X_an)
                    an_1 = out_dict_1["features"]
                    len_data = an_1.shape[0]
                    logit_1 = out_dict_1["logits"]
                    out_dict_2 = self.model(X_an)
                    an_2 = out_dict_2["features"]
                    logit_2 = out_dict_2["logits"]
                    sinkhorn = SinkhornKnopp(
                        num_iters_sk=3, epsilon_sk=0.05, imb_factor=1
                    )
                    pesudo_label = sinkhorn(logit_1.detach().clone())
                    self.pesudo_bank[data_inds] = pesudo_label.detach()
                    max_probs, targets = torch.max(pesudo_label, dim=-1)
                    max_probs_all, targets_all = torch.max(self.pesudo_bank, dim=-1)
                    if epoch >= 1:
                        km_label = self.cluster_label_bank[data_inds]
                        km_label = km_label.cpu()
                        centriods = km.cluster_centers_
                        label_oenhot = torch.eye(self.num_labels)[km_label]
                        logit_cluster, project_map = self.alignment(
                            self.prototype, centriods, label_oenhot
                        )
                    else:
                        loss_cluster = 0
                    adjacency = self.get_adjacency(
                        args, data_inds, pos_neighbors, target
                    )
                    temp_chosen = np.zeros(targets_all.shape[0])
                    for ids, label in enumerate(target):
                        if label >= 0:
                            targets[ids] = label
                            max_probs[ids] = 0
                    if epoch > 0:
                        for j in range(self.num_known):
                            index_j = np.where(targets_all.cpu().numpy() == j)[0]
                            max_score_j = max_probs_all[index_j]
                            sort_index_j = (-max_score_j).sort()[1].cpu().numpy()
                            partition_j = int(len(index_j) * rho_l)
                            if len(index_j) == 0:
                                continue
                            update_ids = index_j[sort_index_j[:partition_j]]
                            temp_chosen[update_ids] = 1
                        for j in range(self.num_known, self.num_labels):
                            index_j = np.where(targets_all.cpu().numpy() == j)[0]
                            max_score_j = max_probs_all[index_j]
                            sort_index_j = (-max_score_j).sort()[1].cpu().numpy()
                            partition_j = int(len(index_j) * rho_n)
                            if len(index_j):
                                features_j = self.feature_bank[index_j]

                            else:
                                continue
                            update_ids = index_j[sort_index_j[:partition_j]]
                            temp_chosen[update_ids] = 2
                    self.chosen_bank = temp_chosen

                    targets_ = targets_all[data_inds]

                    index_chosen_known = []
                    for x, y in enumerate(data_inds):
                        if self.chosen_bank[y] == 1:
                            index_chosen_known.append(x)
                    index_chosen_known = torch.tensor(index_chosen_known)

                    index_chosen_novel = []
                    for x, y in enumerate(data_inds):
                        if self.chosen_bank[y] == 2:
                            index_chosen_novel.append(x)
                    index_chosen_novel = torch.tensor(index_chosen_novel)

                    labeled_ids = []
                    for ids, label in enumerate(target):
                        if label >= 0 and label < self.num_known:
                            labeled_ids.append(ids)
                            self.pesudo_bank[data_inds[ids]] = torch.eye(
                                self.num_labels
                            )[label]
                            if epoch == 0:
                                self.label_ids[data_inds[ids]] = 1
                    labeled_ids = torch.tensor(labeled_ids)

                    loss_ce_known = (
                        (
                            F.cross_entropy(logit_1, targets_, reduction="none")[
                                index_chosen_known
                            ]
                        ).mean()
                        if len(index_chosen_known)
                        else 0
                    )
                    loss_ce_novel = (
                        (
                            F.cross_entropy(logit_1, targets_, reduction="none")[
                                index_chosen_novel
                            ]
                        ).mean()
                        if len(index_chosen_novel)
                        else 0
                    )
                    loss_ce_sup = (
                        (
                            F.cross_entropy(logit_1, targets_, reduction="none")[
                                labeled_ids
                            ]
                        ).mean()
                        if len(labeled_ids)
                        else 0
                    )
                    if epoch >= 1:
                        loss_cluster = (
                            (
                                F.cross_entropy(
                                    logit_1, logit_cluster, reduction="none"
                                )[index_chosen_novel]
                            ).mean()
                            if len(index_chosen_novel)
                            else 0
                        )

                    ng = self.model(X_ng)["features"]
                    f_pos_1 = torch.stack([an_1, ng], dim=1)
                    f_pos_2 = torch.stack([an_2, ng], dim=1)
                    knn_loss_1 = criterion(
                        f_pos_1, mask=adjacency, temperature=args.temp
                    )
                    knn_loss_2 = criterion(
                        f_pos_2, mask=adjacency, temperature=args.temp
                    )
                    knn_loss = 0.5 * (knn_loss_1 + knn_loss_2)

                    self.feature_bank[data_inds] = feature_ori.detach().clone()

                    rdrop_loss = self.compute_kl_loss(logit_1, logit_2)

                    q = torch.Tensor([1] * data.num_labels).cuda()
                    q = q / q.sum()
                    loss_reg = -1 * entropy(
                        torch.mean(F.softmax(logit_1 / 0.3, dim=1), 0),
                        input_as_probabilities=True,
                        q=q,
                    )
                    loss = (
                        3 * w * (loss_ce_known + loss_ce_novel)
                        + 3 * loss_ce_sup
                        + loss_cluster
                        + 5 * loss_reg
                        + rdrop_loss
                        + knn_loss
                    )
                    tr_loss += loss.item()

                    loss.backward()

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    nb_tr_examples += anchor[0].size(0)
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            print("epoch:", epoch)
            print("train_loss", loss)
            print("evaluation")

            res_cluster = self.evaluation(args, data, save_results=False, plot_cm=False)
            if res_cluster > best_res_cluster:
                best_res_cluster = res_cluster
            print("best_res_cluster:", best_res_cluster)

            cur_score = float(res_cluster)
            is_better = (cur_score - best_score_scalar) > min_delta

            if is_better:
                best_score_scalar = cur_score
                wait = 0

                best_model = copy.deepcopy(self.model)
            else:
                wait += 1
                if wait >= patience:
                    print(
                        f"[EarlyStop-CLNN] No improvement on {monitor_name} for {patience} epoch(s). "
                        f"Best {monitor_name}={best_score_scalar:.4f}. Stop training."
                    )

                    if best_model is not None:
                        self.model = best_model
                    break

            if ((epoch + 1) % args.update_per_epoch) == 0:
                bank, indices = self.get_neighbor_inds(args, data)
                self.get_neighbor_dataset(args, data, indices)

    def save_features(self, args, data, epoch):
        feats_test, labels = self.get_features_labels(
            data.train_eval_dataloader, self.model, args
        )
        torch.save(feats_test, "features_{}.pt".format(epoch))
        torch.save(labels, "labels_{}.pt".format(epoch))

    def get_optimizer(self, args):
        num_warmup_steps = int(
            args.warmup_proportion * self.num_train_optimization_steps
        )
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_train_optimization_steps,
        )
        return optimizer, scheduler

    def load_pretrained_model(self):
        if isinstance(self.pretrained_model, nn.DataParallel):
            pretrained_dict = self.pretrained_model.module.backbone.state_dict()
        else:
            pretrained_dict = self.pretrained_model.backbone.state_dict()
        if isinstance(self.model, nn.DataParallel):
            self.model.backbone.load_state_dict(pretrained_dict, strict=False)
        else:
            self.model.backbone.load_state_dict(pretrained_dict, strict=False)

    def get_features_labels(self, dataloader, model, args):
        model.eval()
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            X = {
                "input_ids": input_ids,
                "attention_mask": input_mask,
                "token_type_ids": segment_ids,
            }
            with torch.no_grad():
                feature = model(X, output_hidden_states=True)["features"]

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels

    def get_logits(self, dataloader, model, args):
        model.eval()
        total_logits = torch.empty((0, self.num_labels)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            X = {
                "input_ids": input_ids,
                "attention_mask": input_mask,
                "token_type_ids": segment_ids,
            }
            with torch.no_grad():
                logit = model(X, output_hidden_states=True)["logits"]

            total_logits = torch.cat((total_logits, logit))
            total_labels = torch.cat((total_labels, label_ids))

        return total_logits, total_labels

    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [
            args.method,
            args.dataset,
            args.known_cls_ratio,
            args.labeled_ratio,
            args.seed,
        ]
        names = ["method", "dataset", "known_cls_ratio", "labeled_ratio", "seed"]
        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(self.test_results, **vars_dict)
        results["args"] = json.dumps(vars(args), ensure_ascii=False)
        results["cluster_num_factor"] = args.cluster_num_factor

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
        keys = desired_order
        values = [results[i] for i in desired_order]

        file_name = "results.csv"
        results_path = os.path.join(args.save_results_path, file_name)

        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results, index=[1])
            df1 = df1._append(new, ignore_index=True)
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)

        print("test_results", data_diagram)


def apply_config_updates(args, config_dict, parser):
    type_map = {action.dest: action.type for action in parser._actions}
    for key, value in config_dict.items():
        if f"--{key}" not in sys.argv and hasattr(args, key):
            expected_type = type_map.get(key)
            if expected_type and value is not None:
                try:
                    value = expected_type(value)
                except (TypeError, ValueError):
                    pass
            setattr(args, key, value)


if __name__ == "__main__":

    print("Data and Parameters Initialization...")
    parser = init_model()

    parser.add_argument("--config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)

        apply_config_updates(args, yaml_config, parser)

        if "dataset_specific_configs" in yaml_config:
            dataset_configs = yaml_config["dataset_specific_configs"].get(
                args.dataset, {}
            )
            apply_config_updates(args, dataset_configs, parser)

    if args.known_cls_ratio == 0:
        args.disable_pretrain = True
    else:
        args.disable_pretrain = False

    if not args.disable_pretrain:
        data = Data(args)
        print("Pre-training begin...")
        manager_p = InternalPretrainModelManager(args, data)
        manager_p.train(args, data)
        print("Pre-training finished!")
        manager = CLNNModelManager(args, data, manager_p.model)
    else:
        data = Data(args)
        manager = CLNNModelManager(args, data)

    if args.report_pretrain:
        method = args.method
        args.method = "pretrain"

        manager.evaluation(args, data, save_results=False, plot_cm=False)
        args.method = method

    print("Training begin...")
    manager.train(args, data)
    print("Training finished!")

    print("Evaluation begin...")
    manager.evaluation(args, data)
    print("Evaluation finished!")

    print("Saving Model ...")
    if args.save_model_path:
        manager.model.save_backbone(args.save_model_path)
    print("Finished!")
