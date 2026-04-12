import os
import pandas as pd
from init_parameter import init_model
from data import Data
from model import BertForModel, PretrainModelManager
import numpy as np
import torch.nn.functional as F
from util import clustering_score
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import torch
from sklearn.cluster import KMeans
from transformers import logging, AutoTokenizer, WEIGHTS_NAME
import random
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment
from pytorch_pretrained_bert.optimization import BertAdam
import math
import warnings
from tqdm import tqdm


class ModelManager:
    def __init__(self, args, data, pretrained_model=None):
        self.set_seed(args.seed)
        if pretrained_model is None:
            pretrained_model = BertForModel(args, data.n_known_cls)
            if os.path.exists(args.pretrain_dir):
                pretrained_model = self.restore_model(args, pretrained_model)
        self.pretrained_model = pretrained_model
        self.labelMap = None

        self.seed = args.seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.cluster_num_factor > 1:
            self.num_labels = self.predict_k(args, data)
        else:
            self.num_labels = data.num_labels

        self.model = BertForModel(args, self.num_labels)

        self.load_pretrained_model()

        if args.freeze_bert_parameters:
            self.freeze_parameters(self.model)

        self.model.to(self.device)

        self.data = data
        num_train_examples = 2 * len(data.train_labeled_examples) + 2 * len(data.train_unlabeled_examples)
        self.num_training_steps = math.ceil(num_train_examples / args.train_batch_size) * 100
        self.num_warmup_steps= int(args.warmup_proportion * self.num_training_steps)
        self.optimizer = self.get_optimizer(args)
        self.centroids = None
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def get_features_labels(self, dataloader, model, args):
        model.eval()
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        for _, batch in enumerate(dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                feature = model(input_ids, segment_ids, input_mask, mode='feature_extract')
            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))
        return total_features, total_labels

    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                         lr = args.lr,
                         warmup = args.warmup_proportion,
                         t_total = self.num_training_steps)
        return optimizer

    def evaluation(self, args, data):
        print("\n" + "="*80)
        print("Starting Evaluation...")
        print(f"Dataset: {args.dataset}, Known Ratio: {args.known_cls_ratio}, Seed: {args.seed}")
        print("="*80)

        self.model.eval()
        feats, labels = self.get_features_labels(data.test_dataloader, self.model, args)
        feats = feats.cpu().numpy()
        print(f"Extracted features shape: {feats.shape}")
        print(f"Performing KMeans clustering with {self.num_labels} clusters...")
        km = KMeans(n_clusters = self.num_labels, n_init=20).fit(feats)
        y_true = labels.cpu().numpy()
        print(f"True labels shape: {y_true.shape}, Unique labels: {len(np.unique(y_true))}")
        print(f"Predicted labels shape: {km.labels_.shape}, Unique clusters: {len(np.unique(km.labels_))}")

        results = clustering_score(y_true, km.labels_, data.known_lab)
        print("\n" + "-"*80)
        print("Clustering Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        print("-"*80 + "\n")

        self.save_results(args, results)

        return results

    def save_results(self, args, test_results):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)
            print(f"Created results directory: {args.save_results_path}")

        var = [args.dataset, args.method, args.known_cls_ratio,
               args.labeled_ratio, args.cluster_num_factor, args.seed]
        names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio',
                'cluster_num_factor', 'seed']
        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(test_results, **vars_dict)

        ordered_keys = ['ACC_all', 'H-Score', 'ACC_known', 'ACC_novel', 'ARI', 'NMI',
                       'dataset', 'method', 'known_cls_ratio', 'labeled_ratio',
                       'cluster_num_factor', 'seed']

        key_mapping = {
            'ACC_all': 'ACC',
            'ACC_known': 'K-ACC',
            'ACC_novel': 'N-ACC'
        }

        final_results = {}
        for key in ordered_keys:
            if key in key_mapping:
                final_results[key_mapping[key]] = results[key]
            else:
                final_results[key] = results[key]

        keys = list(final_results.keys())
        values = list(final_results.values())

        file_name = 'results.csv'
        results_path = os.path.join(args.save_results_path, file_name)

        print("\n" + "="*80)
        print("Saving Results to CSV...")
        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
            print(f"Created new results file: {results_path}")
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(final_results, index=[1])
            df1 = pd.concat([df1, new], ignore_index=True)
            df1.to_csv(results_path, index=False)
            print(f"Appended results to existing file: {results_path}")

        print(f"\nCurrent results in {file_name}:")
        print(pd.read_csv(results_path).to_string())
        print("="*80 + "\n")

    def alignment(self, km, args):
        if self.centroids is not None:
            old_centroids = self.centroids.cpu().numpy()
            new_centroids = km.cluster_centers_

            DistanceMatrix = np.linalg.norm(old_centroids[:,np.newaxis,:]-new_centroids[np.newaxis,:,:],axis=2)
            row_ind, col_ind = linear_sum_assignment(DistanceMatrix)

            new_centroids = torch.tensor(new_centroids).to(self.device)
            self.centroids = torch.empty(self.num_labels ,args.feat_dim).to(self.device)

            alignment_labels = list(col_ind)
            for i in range(self.num_labels):
                label = alignment_labels[i]
                self.centroids[i] = new_centroids[label]

            pseudo2label = {label:i for i,label in enumerate(alignment_labels)}
            pseudo_labels = np.array([pseudo2label[label] for label in km.labels_])
            self.labelMap = pseudo2label
        else:
            self.centroids = torch.tensor(km.cluster_centers_).to(self.device)
            pseudo_labels = km.labels_

        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long).to(self.device)

        return pseudo_labels

    def update_cluster_ids(self, pseudo_labels, args, data):
        train_data = TensorDataset(data.semi_input_ids, data.semi_input_mask, data.semi_segment_ids, pseudo_labels)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = args.train_batch_size)
        return train_dataloader

    def train(self, args, data):
        feats_label, labels = self.get_features_labels(data.train_labeled_dataloader, self.model, args)
        feats_label = feats_label.cpu().numpy()
        labels = labels.cpu().numpy()
        [rows, cols] = feats_label.shape
        num = np.zeros(data.n_known_cls)
        # labeled prototypes
        proto_l = np.zeros((data.n_known_cls, args.feat_dim))
        for i in range(rows):
            proto_l[labels[i]] += feats_label[i]
            num[labels[i]] += 1
        for i in range(data.n_known_cls):
            proto_l[i] = proto_l[i] / num[i]

        feats, _ = self.get_features_labels(data.train_semi_dataloader, self.model, args)
        feats = feats.cpu().numpy()

        print(f"\nPerforming k-means clustering on {feats.shape[0]} samples with {self.num_labels} clusters...")
        km = KMeans(n_clusters = self.num_labels, n_init=20).fit(feats)
        print(f"K-means finished. Cluster sizes: {np.bincount(km.labels_)}")
        # unlabeled prototypes
        proto_u = km.cluster_centers_
        distance = dist.cdist(proto_l, proto_u, 'euclidean')
        row_ind, col_ind = linear_sum_assignment(distance)
        pro_l = []
        for i in range(len(col_ind)):
            pro_l.append(proto_u[col_ind[i]][:])
        pro_u = []
        for j in range(self.num_labels):
            if j not in col_ind:
                pro_u.append(proto_u[j][:])

        # aligned unlabeled prototypes
        pro_u = pro_l + pro_u
        proto_u = torch.tensor(np.array(pro_u)).to(self.device)

        # update cluster_ids for unlabeled data
        self.centroids = proto_u
        pseudo_labels = self.alignment(km, args)
        train_dataloader = self.update_cluster_ids(pseudo_labels, args, data)

        labelediter = iter(data.train_labeled_dataloader)

        self.proto_l = proto_l
        self.proto_u = proto_u

        for epoch in tqdm(range(1, int(args.num_train_epochs)+1, 1), desc="Training Epochs"):

            feats_label, labels = self.get_features_labels(data.train_labeled_dataloader, self.model, args)
            feats_label = feats_label.cpu().numpy()
            labels = labels.cpu().numpy()
            [rows, cols] = feats_label.shape
            num = np.zeros(data.n_known_cls)
            proto_l = np.zeros((data.n_known_cls, args.feat_dim))
            for i in range(rows):
                proto_l[labels[i]] += feats_label[i]
                num[labels[i]] += 1
            for i in range(data.n_known_cls):
                proto_l[i] = proto_l[i] / num[i]

            # update labeled prototypes
            self.proto_l = args.momentum_factor * self.proto_l + (1 - args.momentum_factor) * proto_l

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.model.train()

            for _, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                pooled = self.model(input_ids, segment_ids, input_mask, label_ids, mode='feature_extract')

                sim_mat = self.pairwise_cosine_sim(pooled, self.proto_u) / args.temperature
                s_dist = F.softmax(sim_mat, dim=1)
                cost_mat = self.EuclideanDistances(pooled, self.proto_u)
                loss_u = (cost_mat * s_dist).sum(1).mean()

                mask = torch.where(label_ids < len(self.proto_l), 1, 0).reshape((-1,1))
                sim_mat = self.pairwise_cosine_sim(pooled, torch.tensor(self.proto_l).float().cuda()) / args.temperature
                s_dist = F.softmax(sim_mat, dim=1)
                cost_mat = 1 - self.pairwise_cosine_sim(pooled, torch.tensor(self.proto_l).float().cuda())
                loss_l = (cost_mat * s_dist * mask).sum(1).mean()

                loss_pro = loss_u + args.gamma * loss_l

                try:
                    batch = next(labelediter)
                except StopIteration:
                    labelediter = iter(data.train_labeled_dataloader)
                    batch = next(labelediter)
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss_ce, _ = self.model(input_ids, segment_ids, input_mask, label_ids, mode='train')

                loss = loss_pro + loss_ce

                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer.step()
                self.optimizer.zero_grad()

            avg_loss = tr_loss/nb_tr_steps
            print(f'Epoch {epoch}/{int(args.num_train_epochs)} - Loss: {avg_loss:.6f} - Steps: {nb_tr_steps}')

    def load_pretrained_model(self):
        pretrained_dict = self.pretrained_model.state_dict()
        classifier_params = ['classifier.weight','classifier.bias']
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def restore_model(self, args, model):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model

    def freeze_parameters(self, model):
        for name, param in model.bert.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name or "encoder.layer.10" in name or "encoder.layer.9" in name:
                param.requires_grad = True

    def pairwise_cosine_sim(self, x, y):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return torch.matmul(x, y.T)

    def EuclideanDistances(self, a, b):
        sq_a = a**2
        sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
        sq_b = b**2
        sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
        bt = b.t()
        return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))

    def predict_k(self, args, data):
        feats, _ = self.get_features_labels(data.train_semi_dataloader, self.pretrained_model.cuda(), args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters = data.num_labels).fit(feats)
        y_pred = km.labels_

        pred_label_list = np.unique(y_pred)
        drop_out = len(feats) / data.num_labels * 0.9

        cnt = 0
        for label in pred_label_list:
            num = len(y_pred[y_pred == label])
            if num < drop_out:
                cnt += 1

        num_labels = len(pred_label_list) - cnt
        print('predict_K', num_labels)

        return num_labels

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print('\n' + '='*80)
    print('Data and Parameters Initialization...')
    print('='*80)
    parser = init_model()
    args = parser.parse_args()
    args.bert_model = '../../pretrained_models/bert-base-chinese' if args.dataset == 'ecdt' else args.bert_model
    args.bert_model = './pretrained_models/bert-base-chinese' if args.dataset == 'thucnews' else args.bert_model

    print(f"\nExperiment Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Known Class Ratio: {args.known_cls_ratio}")
    print(f"  Labeled Ratio: {args.labeled_ratio}")
    print(f"  Seed: {args.seed}")
    print(f"  GPU ID: {args.gpu_id}")
    print(f"  BERT Model: {args.bert_model}")
    print(f"  Batch Size (train/eval): {args.train_batch_size}/{args.eval_batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Training Epochs: {args.num_train_epochs}")
    print('='*80 + '\n')

    data = Data(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.pretrain:
        print('\n' + '='*80)
        print('Pre-training begin...')
        print('='*80)
        manager_p = PretrainModelManager(args, data)
        manager_p.train()
        print('\n' + '='*80)
        print('Pre-training finished!')
        print('='*80 + '\n')
        manager = ModelManager(args, data, manager_p.model)
    else:
        manager = ModelManager(args, data, None)

    print('\n' + '='*80)
    print('Training begin...')
    print('='*80)
    manager.train(args,data)
    print('\n' + '='*80)
    print('Training finished!')
    print('='*80 + '\n')

    manager.evaluation(args, data)
    print('\n' + '='*80)
    print('Evaluation finished!')
    print('='*80 + '\n')
