# train_ood.py (最终修正版)

import os

import copy
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve
from src.pytorch_ood.detector import (
    EnergyBased,
    Entropy,
    KLMatching,
    Mahalanobis,
    MaxLogit,
    MaxSoftmax,
    ViM,
    OpenMax,
    TemperatureScaling,
    ASH,
    SHE,
    LogitNorm
)
from src.pytorch_ood.utils import OODMetrics, fix_random_seed, custom_metrics
import numpy as np
import logging
import sys
import yaml
import json
from tqdm import tqdm
tqdm.pandas()
from model import Model
from load_dataset import load_and_prepare_datasets
# from configs import get_plm_ood_config
from configs import create_parser, finalize_config
from llm_query import LLMQuerier  # 假设你把上面的类保存为 llm_querier.py


from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd

def run_zero_few_shot(row, clf, known_labels, upper_thred=0.9, lower_thred=0.5, label_samples=None):
    """单行任务函数"""
    if row['norm_scores'] >= upper_thred or row['norm_scores'] <= lower_thred:
        return None
    try:
        res = clf.zero_shot(text=row['text'][:256], pred=row['pred'], candidate_labels=known_labels, label_samples=label_samples[row['pred']] if row['pred'] in label_samples else None)
        return res
    except Exception as e:
        return None  # 或者记录错误

def parallel_shot(df_pred, clf, known_labels, upper_thred=0.9, lower_thred=0.5, label_samples=None, max_workers=5):
    results = [None] * len(df_pred)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_zero_few_shot, row, clf, known_labels, upper_thred, lower_thred, label_samples): idx
            for idx, row in df_pred.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="LLM OOD parallel"):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = None
    return results

def run_ood_evaluation(args):
    # ====== Step 2. 初始化 LLMQuerier ======
    base_url = "https://openrouter.ai/api/v1"
    model_name = "deepseek/deepseek-chat-v3-0324"
    clf = LLMQuerier(base_url, model_name)
    
    """
    OOD 评测的主函数，包含了所有的核心逻辑。
    """
    # --- 改动 2: 在函数开头，调用数据加载函数 ---
    logging.info("Loading and preparing datasets...")
    # 调用函数，获取一个包含所有数据对象的字典
    data = load_and_prepare_datasets(args)
    tokenizer = data['tokenizer']
    # 从字典中解包出我们需要的变量
    test_loader = data['test_loader']
    loader_in_train = data['loader_in_train']
    logging.info("Datasets loaded successfully.")
    # --- 改动结束 ---

    fix_random_seed(args.seed)
    model = Model(args, tokenizer=tokenizer).to(args.device)
    
    with torch.no_grad():
        model.eval()
        # vec_dir = '/'.join(args.vector_path.split('/')[:3] + args.vector_path.split('/')[4:]).replace('plm_ood-llm', 'PLM_OOD')
        # vec_dir = '/'.join(args.vector_path.split('/')[:3] + args.vector_path.split('/')[4:]).replace('plm_ood-llm', 'PLM_OOD')
        vec_dir = args.vector_path

        if not os.path.exists(f'{vec_dir}/logits.npy'):
            logging.info("Calculating logits, predictions, and features...")
            preds, golds, logits, features = [], [], [], []
            # --- 改动 3: 使用上面解包出来的局部变量 test_loader ---
            for batch in tqdm(test_loader, desc="Inference"):
                y = batch['labels'].to(args.device)
                batch = {i: v.to(args.device) for i, v in batch.items() if i != 'labels'}
                logit = model(batch)
                feature = model.features(batch)
                pred = logit.max(dim=1).indices
                
                preds.append(pred)
                logits.append(logit)
                golds.append(y)
                features.append(feature)
            
            logits = torch.concat(logits).detach().to(torch.float32).cpu().numpy()
            preds = torch.concat(preds).detach().cpu().numpy()
            golds = torch.concat(golds).detach().cpu().numpy()
            features = torch.concat(features).detach().to(torch.float32).cpu().numpy()

            np.save(f'{vec_dir}/logits.npy', logits)
            np.save(f'{vec_dir}/preds.npy', preds)
            np.save(f'{vec_dir}/golds.npy', golds)
            np.save(f'{vec_dir}/features.npy', features)
        else:
            logging.info("Loading pre-calculated logits, predictions, and features...")
            logits = np.load(f'{vec_dir}/logits.npy')
            preds = np.load(f'{vec_dir}/preds.npy')
            golds = np.load(f'{vec_dir}/golds.npy')
            features = np.load(f'{vec_dir}/features.npy')

        ID_metrics = custom_metrics(preds, golds)
        logging.info(f"Test Accuracy: {ID_metrics['macro avg']}")

    

    logging.info("STAGE 2: Creating OOD Detectors")
    detectors = {
        # "TemperatureScaling": TemperatureScaling(model),
        # "LogitNorm": LogitNorm(model),
        # "OpenMax": OpenMax(model),
        # "Entropy": Entropy(model),
        # "Mahalanobis": Mahalanobis(model.features, eps=0.0),
        # "KLMatching": KLMatching(model),
        # "MaxSoftmax": MaxSoftmax(model),
        "EnergyBased": EnergyBased(model),
        # "MaxLogit": MaxLogit(model)
    }

    logging.info(f"> Fitting {len(detectors)} detectors")
    # for name, detector in detectors.items():
    #     logging.info(f"--> Fitting {name}")
    #     # --- 改动 4: 使用上面解包出来的局部变量 loader_in_train ---
    #     detector.fit(loader_in_train, device=args.device)

    # logging.info(f"STAGE 3: Evaluating {len(detectors)} detectors.")
    results = []
    
    for detector_name, detector in detectors.items():
        if os.path.exists(f'{vec_dir}/{detector_name}_norm_scores.npy'):
            norm_scores = np.load(f'{vec_dir}/{detector_name}_norm_scores.npy')
            final_preds = np.load(f'{vec_dir}/{detector_name}_preds.npy')
            golds = np.load(f'{vec_dir}/golds.npy')
        else:
            logging.info(f"--> Fitting {detector_name}")
            # --- 改动 4: 使用上面解包出来的局部变量 loader_in_train ---
            detector.fit(loader_in_train, device=args.device)
            with torch.no_grad():
                logging.info(f"> Evaluating {detector_name}")
                scores = []
                # --- 改动 5: 再次使用 test_loader ---
                for batch in tqdm(test_loader, desc=f"Evaluating {detector_name}"):
                    y = batch['labels'].to(args.device)
                    batch = {i: v.to(args.device) for i, v in batch.items() if i != 'labels'}
                    score = detector(batch)
                    scores.append(score)
                scores = torch.concat(scores).detach().cpu().numpy()
                norm_scores = (scores - scores.min()) / (scores.max() - scores.min())

                if detector_name == 'Vim':
                    new_logits = np.concatenate([logits, scores.reshape(scores.shape[0], 1)], axis=1)
                    new_preds = new_logits.argmax(axis=1)
                    new_preds[new_preds==logits.shape[-1]] = -1
                    preds = new_preds
                    final_preds = copy.deepcopy(preds)
                else:
                    final_preds = copy.deepcopy(preds)
                    final_preds[norm_scores > 0.5] = -1

                np.save(f'{vec_dir}/{detector_name}_norm_scores.npy', norm_scores)
                np.save(f'{vec_dir}/{detector_name}_preds.npy', final_preds)
                np.save(f'{vec_dir}/golds.npy', golds)

        df_pred = data['test_data'][:]
        df_pred['pred'] = final_preds
        df_pred['gold'] = golds
        df_pred['norm_scores'] = norm_scores

        # # 只保留 gold != -1 的行
        # known_df = df_pred[df_pred['gold'] != -1]

        # # 取每个 gold 的第一个 label（保证按 gold 顺序）
        # known_labels = (
        #     known_df.drop_duplicates(subset='gold')  # 每个 gold 只保留一条
        #             .sort_values('gold')            # 按 gold 排序
        #             ['label']
        #             .tolist()
        # )

        known_labels = data['known_label_list']


        df_pred['pred'] = df_pred['pred'].apply(lambda x: known_labels[x] if x in known_labels else "OOD") 
        df_pred['gold'] = df_pred['label'].apply(lambda x: x if x in known_labels else "OOD") 

        label_samples =  data['train_data'].groupby('label').agg(list)[['text']]
        label_samples['text'] = label_samples['text'].apply(lambda x: x[:20])
        
        label_samples = label_samples.to_dict()['text']
        metrics = custom_metrics(final_preds, golds, norm_scores, thred=0.5)
        # metrics = custom_metrics(df_pred['pred'].apply(lambda x: known_labels.index(x) if x in known_labels else -1), df_pred['label'].apply(lambda x: known_labels.index(x) if x in known_labels else -1), norm_scores, thred=0.5)


        df_pred['llm_pred_zero-shot'] = parallel_shot(
            df_pred=df_pred,
            clf=clf,
            known_labels=known_labels,
            max_workers=5,
            label_samples=label_samples,
            upper_thred=0.6,
            lower_thred=0.4
        )

        # df_pred['llm_pred_few-shot'] = parallel_shot(
        #     df_pred=df_pred,
        #     clf=clf,
        #     known_labels=known_labels,
        #     max_workers=5,
        #     label_samples=label_samples,
        #     upper_thred=0.7,
        #     lower_thred=0.4
        # )
        
        df_pred['llm_pred_zero'] = df_pred.apply(lambda x: x['pred'] if x['llm_pred_zero-shot'] is None or not x['llm_pred_zero-shot']['is_ood'] or x['llm_pred_zero-shot']['confidence'] < 0.9 else 'OOD', axis=1)
        zero_metrics = custom_metrics(df_pred['llm_pred_zero'].apply(lambda x: known_labels.index(x) if x in known_labels else -1), df_pred['label'].apply(lambda x: known_labels.index(x) if x in known_labels else -1), norm_scores, thred=1)

        zero_metrics['detector'] = detector_name
        zero_metrics['prompt_method'] = 'zero-shot'
        results.append(zero_metrics)

        # few_metrics = custom_metrics(df_pred['llm_pred_few'].apply(lambda x: known_labels.index(x) if x in known_labels else -1), golds, norm_scores, thred=1)
        # few_metrics['detector'] = detector_name
        # few_metrics['prompt_method'] = 'few-shot'
        # results.append(few_metrics)


    # df = pd.DataFrame(results)
    # mean_scores = df.groupby(["Detector"]).mean() * 100
    # mean_scores['args'] = json.dumps(vars(args), ensure_ascii=False)
    # logging.info(mean_scores.sort_values("AUROC").to_csv(float_format="%.2f"))
    # mean_scores.to_csv(args.metric_file, sep='\t')

    df_to_save = pd.DataFrame(results)
    df_to_save['method'] = 'PLM_OOD'
    df_to_save['K-F1'] = df_to_save['K-f1']
    df_to_save['N-F1'] = df_to_save['N-f1']
    df_to_save['F1'] = df_to_save['f1-score']
    df_to_save['ACC'] = df_to_save['accuracy']
    def func(args, detecor,  prompt_method):
        args['Detecor'] = detecor
        args['prompt_method'] = prompt_method
        return json.dumps(args)

    df_to_save['args'] = df_to_save.apply(lambda x: func(vars(args), x['detector'], x['prompt_method']), axis=1)
    cols = ['method','dataset','known_cls_ratio','labeled_ratio','cluster_num_factor','seed','ACC','F1','K-F1','N-F1','args']
    for col in cols:
        if col in df_to_save:
            continue
        df_to_save[col] = getattr(args, col)
    df_to_save = df_to_save[cols]

    os.makedirs(os.path.dirname(args.metric_file), exist_ok=True)
    if not os.path.exists(args.metric_file):
        df_to_save.to_csv(args.metric_file, index=False)
    else:
        pd.concat([pd.read_csv(args.metric_file), df_to_save], ignore_index=True).to_csv(args.metric_file, index=False)

def apply_config_updates(args, config_dict, parser):
    """
    使用配置字典中的值更新 args 对象，同时进行类型转换。
    命令行中显式给出的参数不会被覆盖。
    """
    # 创建一个从 dest 到 action.type 的映射
    type_map = {action.dest: action.type for action in parser._actions}

    for key, value in config_dict.items():
        # 检查参数是否在命令行中被用户显式提供
        if f'--{key}' not in sys.argv and hasattr(args, key):
            # 获取该参数预期的类型
            expected_type = type_map.get(key)
            # 如果有预期类型且值不为None，则进行类型转换
            if expected_type and value is not None:
                value = expected_type(value)
            setattr(args, key, value)

if __name__ == '__main__':
    parser = create_parser()
    parser.add_argument("--config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        apply_config_updates(args, yaml_config, parser)
        
    config_args = finalize_config(args)
    
    run_ood_evaluation(config_args)