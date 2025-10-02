import os
import torch
import argparse
import pandas as pd
from model import BertForModel
from dataloader import Data
from pretrain import PretrainModelManager
from util import F_measure
from loss import BoundaryLoss
from tqdm import tqdm, trange
from sklearn.metrics import confusion_matrix, accuracy_score
from transformers.utils import WEIGHTS_NAME
import numpy as np
import torch.nn.functional as F 
from init_parameter import init_model
import yaml
import sys
import json
import os, shutil

class ModelManager:
    def __init__(self, args, data, pretrained_model):
        self.model = pretrained_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.best_eval_score = 0
        self.trained = False
        self.centroids = self.centroids_cal(args, data)
        self.delta = None
        self.test_results = None

    def open_classify(self, features, data, args):
        logits = torch.nn.functional.normalize(features, dim=1) @ torch.nn.functional.normalize(self.centroids, dim=1).T
        probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1)
        cos_dis = 1 - torch.nn.functional.cosine_similarity(features, self.centroids[preds], dim=1)
        preds[cos_dis >= self.delta[preds]] = data.unseen_token_id
        return preds

    def evaluation(self, args, data, mode="eval"):
        self.model.eval()
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)
        dataloader = data.eval_dataloader if mode == "eval" else data.test_dataloader

        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                pooled_output, _ = self.model(input_ids, segment_ids, input_mask)
                preds = self.open_classify(pooled_output, data, args)
                total_labels = torch.cat((total_labels, label_ids))
                total_preds = torch.cat((total_preds, preds))
        
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        
        if mode == "eval":
            cm = confusion_matrix(y_true, y_pred)
            return F_measure(cm)["F1-score"]
        elif mode == "test":
            cm = confusion_matrix(y_true, y_pred)
            results = F_measure(cm)
            results["Accuracy"] = round(accuracy_score(y_true, y_pred) * 100, 2)
            self.test_results = results

    def train(self, args, data):
        criterion_boundary = BoundaryLoss(num_labels=data.num_labels, feat_dim=args.feat_dim, cosine=args.cosine)
        self.delta = criterion_boundary.delta
        optimizer = torch.optim.Adam(criterion_boundary.parameters(), lr=args.lr_boundary)
        wait = 0
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_steps = 0
            for step, batch in enumerate(data.train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                    loss, self.delta = criterion_boundary(features, self.centroids, label_ids)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    tr_loss += loss.item()
                    nb_tr_steps += 1
            
            print(f"Epoch {epoch+1}, Train Loss: {tr_loss / nb_tr_steps}")
            eval_score = self.evaluation(args, data, mode="eval")
            print(f"Eval F1-score: {eval_score}")

            if eval_score >= self.best_eval_score:
                wait = 0
                self.best_eval_score = eval_score
                best_delta = self.delta
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break
        self.delta = best_delta

    def class_count(self, labels):
        class_data_num = [len(labels[labels == l]) for l in np.unique(labels)]
        return class_data_num

    def centroids_cal(self, args, data):
        centroids = torch.zeros(data.num_labels, args.feat_dim).cuda()
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        with torch.no_grad():
            for batch in data.train_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                total_labels = torch.cat((total_labels, label_ids))
                for i in range(len(label_ids)):
                    centroids[label_ids[i]] += features[i]
        
        total_labels_np = total_labels.cpu().numpy()
        class_counts = self.class_count(total_labels_np)
        centroids /= torch.tensor(class_counts).float().unsqueeze(1).cuda()
        
        if args.cosine:
            centroids = torch.nn.functional.normalize(centroids, dim=1)
        return centroids

    def save_results(self, args):
        final_results = {}
        final_results['dataset'] = args.dataset
        final_results['seed'] = args.seed
        final_results['known_cls_ratio'] = args.known_cls_ratio
        final_results['ood_method'] = 'CLAP'
        final_results['ACC'] = self.test_results.get('Accuracy', 0.0)
        final_results['F1'] = self.test_results.get('F1-score', 0.0)
        final_results['K-F1'] = self.test_results.get('F1-score_seen', 0.0)
        final_results['N-F1'] = self.test_results.get('F1-score_unseen', 0.0)
        final_results['args'] = json.dumps(vars(args), ensure_ascii=False)

        os.makedirs(args.save_results_path, exist_ok=True)
        results_path = os.path.join(args.save_results_path, 'results.csv')

        if not os.path.exists(results_path):
            df_to_save = pd.DataFrame([final_results])

            df_to_save['method'] = 'clap'
            cols = ['method','dataset','known_cls_ratio','labeled_ratio','cluster_num_factor','seed','ACC','F1','K-F1','N-F1','args']
            for col in cols:
                if col in df_to_save:
                    continue
                df_to_save[col] = getattr(args, col)
            df_to_save = df_to_save[cols]

            df_to_save.to_csv(results_path, index=False)
        else:
            existing_df = pd.read_csv(results_path)
            new_row_df = pd.DataFrame([final_results])


            new_row_df['method'] = 'clap'
            cols = ['method','dataset','known_cls_ratio','labeled_ratio','cluster_num_factor','seed','ACC','F1','K-F1','N-F1','args']
            for col in cols:
                if col in new_row_df:
                    continue
                new_row_df[col] = getattr(args, col)
            new_row_df = new_row_df[cols]

            updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            updated_df.to_csv(results_path, index=False)
            
        print(f"\nResults have been saved to: {results_path}")
        print("Appended new result row:")
        print(pd.DataFrame([final_results]))

def main(args):
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # 加载数据
    data = Data(args)

    pretrained_model = BertForModel.from_pretrained(args.output_dir + '/finetuned_model', num_labels=data.num_labels, cosine=args.cosine)

    # 初始化模型管理器
    manager = ModelManager(args, data, pretrained_model)
    
    # 训练边界
    print("Training boundary...")
    manager.train(args, data)
    
    # 评估
    print("Evaluating...")
    manager.evaluation(args, data, mode="test")
    
    # 保存结果
    print("Saving results...")
    manager.save_results(args)

if __name__ == "__main__":
    parser = init_model() # init_model 来自同目录的 init_parameter.py
    # 新增 config 和 output_dir
    parser.add_argument('--config', type=str, default=None, help="Path to the YAML config file.")
    # parser.add_argument("--output_dir", type=str, default='./outputs/openset/clap', help="Standardized output directory.")
    args = parser.parse_args()

    # 2. 配置注入
    def apply_config_updates(args, config_dict, parser):
        # ... (此处省略我们标准的 apply_config_updates 函数，请从之前的文件复制过来)
        type_map = {action.dest: action.type for action in parser._actions}
        for key, value in config_dict.items():
            if f'--{key}' in sys.argv or not hasattr(args, key): continue
            expected_type = type_map.get(key)
            if expected_type and value is not None:
                try:
                    if expected_type is bool or (hasattr(expected_type, '__name__') and expected_type.__name__ == 'str_to_bool'):
                        value = str(value).lower() in ('true', '1', 't', 'yes')
                    else: value = expected_type(value)
                except (ValueError, TypeError): print(f"Warning: Could not cast YAML value '{value}' for key '{key}' to type {expected_type}.")
            setattr(args, key, value)

    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        # 先应用通用配置
        apply_config_updates(args, yaml_config, parser)
        # 再应用 boundary_adjustment 专属配置
        if 'boundary_adjustment' in yaml_config:
            apply_config_updates(args, yaml_config['boundary_adjustment'], parser)
        # (可选) 应用数据集专属配置
        if 'dataset_specific_configs' in yaml_config:
            dataset_configs = yaml_config['dataset_specific_configs'].get(args.dataset, {})
            if 'boundary_adjustment' in dataset_configs:
                apply_config_updates(args, dataset_configs['boundary_adjustment'], parser)
    
    # 调用主函数
    main(args)