import os
import time
import torch
import numpy as np
import pandas as pd
from dataloader import *
from model import *
from pretrain import *
from utils.utils import *
import train
from init_parameter import init_model

class KTNModelManager:
    def __init__(self, args):
        self.args = args
        self.data = Data(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_results = None

    def pretrain_stage(self):
        """Multi-task pretraining stage"""
        print("=" * 50)
        print("Stage 1: Multi-task Pretraining")
        print("=" * 50)

        manager_p = PretrainModelManager(self.args, self.data)
        manager_p.train(self.args, self.data)
        pretrain_results = manager_p.evaluation(self.args, self.data)

        print(f"Pretrain results: {pretrain_results}")
        return manager_p.model

    def train_stage(self, pretrained_model):
        """Pseudo-label training stage"""
        print("=" * 50)
        print("Stage 2: Pseudo-label Training")
        print("=" * 50)

        manager = train.Manager(self.args, self.data, pretrained_model)
        manager.train(self.args, self.data)
        return manager

    def evaluation_stage(self, manager):
        """Evaluation stage"""
        print("=" * 50)
        print("Stage 3: Evaluation")
        print("=" * 50)

        start_time = time.time()
        manager.model.eval()
        pred_labels = torch.empty(0, dtype=torch.long).to(manager.device)
        total_labels = torch.empty(0, dtype=torch.long).to(manager.device)

        for batch in self.data.test_dataloader:
            batch = tuple(t.to(manager.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            X = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.no_grad():
                _, logits = manager.model(X, output_hidden_states=True)
            labels = torch.argmax(logits, dim=1)

            pred_labels = torch.cat((pred_labels, labels))
            total_labels = torch.cat((total_labels, label_ids))

        y_pred = pred_labels.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        results = clustering_score(y_true, y_pred, self.data.known_lab)
        end_time = time.time()

        print(f'Final results: {results}')
        print(f'Evaluation time: {end_time - start_time:.2f} seconds')

        self.test_results = results
        return results

    def save_results(self):
        """Save results to CSV file"""
        if not os.path.exists(self.args.save_results_path):
            os.makedirs(self.args.save_results_path)

        var = [self.args.dataset, self.args.method, self.args.known_cls_ratio,
               self.args.labeled_ratio, self.args.cluster_num_factor, self.args.seed]
        names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio',
                'cluster_num_factor', 'seed']
        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(self.test_results, **vars_dict)

        # Reorder columns to match DPN format
        ordered_keys = ['ACC_all', 'H-Score', 'ACC_known', 'ACC_novel', 'ARI', 'NMI',
                       'dataset', 'method', 'known_cls_ratio', 'labeled_ratio',
                       'cluster_num_factor', 'seed']

        # Rename keys to match DPN format
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
        results_path = os.path.join(self.args.save_results_path, file_name)

        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(final_results, index=[1])
            df1 = pd.concat([df1, new], ignore_index=True)
            df1.to_csv(results_path, index=False)

        print(f"Results saved to {results_path}")
        print(pd.read_csv(results_path))

    def run_full_pipeline(self):
        """Run the full training and evaluation pipeline"""
        print(f"Running KTN on dataset: {self.args.dataset}")
        print(f"Known class ratio: {self.args.known_cls_ratio}")
        print(f"Labeled ratio: {self.args.labeled_ratio}")
        print(f"Seed: {self.args.seed}")

        # Stage 1: Pretraining
        pretrained_model = self.pretrain_stage()

        # Stage 2: Pseudo-label training
        manager = self.train_stage(pretrained_model)

        # Stage 3: Evaluation
        results = self.evaluation_stage(manager)

        # Save results
        self.save_results()

        return results

def main():
    parser = init_model()
    args = parser.parse_args()

    # Use Chinese BERT for Chinese datasets
    if args.dataset in ['ecdt', 'thucnews']:
        args.bert_model = './pretrained_models/bert-base-chinese'
        args.tokenizer = './pretrained_models/bert-base-chinese'
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    ktn_manager = KTNModelManager(args)
    results = ktn_manager.run_full_pipeline()

    return results

if __name__ == "__main__":
    main()
