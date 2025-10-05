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
from utils import llm_ood_eval

from model import Model
from load_dataset import load_and_prepare_datasets
# from configs import get_plm_ood_config
from configs import create_parser, finalize_config

def run_ood_evaluation(args):
    """
    OOD 评测的主函数，包含了所有的核心逻辑。
    """
    # --- 改动 2: 在函数开头，调用数据加载函数 ---
    logging.info("Loading and preparing datasets...")
    # 调用函数，获取一个包含所有数据对象的字典
    data = load_and_prepare_datasets(args)
    
    test_loader = data["test_loader"]
    loader_in_train = data["loader_in_train"]
    test_texts = data.get("test_texts", None)
    test_labels_str = data.get("test_labels_str", None)
    known_label_list = data.get("known_label_list", None)

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

        # === 统一定位缓存目录 ===
        auto_dir = os.path.join(args.output_dir, "case_study")  # 当前方法(plm_ood-llm)默认目录
        # 兼容旧方法(plm_ood)的缓存目录
        legacy_dir = None
        if "plm_ood-llm" in auto_dir:
            legacy_dir = auto_dir.replace("plm_ood-llm", "plm_ood")

        # 候选顺序：显式 vector_path -> 显式 case_path -> 当前默认 -> 旧方法默认
        vec_candidates = []
        for p in [getattr(args, "vector_path", ""),
                getattr(args, "case_path", ""),
                auto_dir,
                legacy_dir]:
            if p:
                vec_candidates.append(p)

        # # 选第一个“已存在 logits.npy”的目录；否则回落到 auto_dir
        # vec_dir = None
        # for p in vec_candidates:
        #     if os.path.exists(os.path.join(p, "logits.npy")):
        #         vec_dir = p
        #         logging.info(f"[VEC_DIR] reuse cached vectors from: {p}")
        #         break
        # if vec_dir is None:
        #     vec_dir = auto_dir
        #     logging.info(f"[VEC_DIR] no cache found; will use: {vec_dir}")
        # os.makedirs(vec_dir, exist_ok=True)
        # args.vector_path = vec_dir  # 后续保存/读取都用它

        vec_dir = '/'.join(args.vector_path.split('/')[:3] + args.vector_path.split('/')[4:]).replace('plm_ood-llm', 'PLM_OOD')

        # === 加载或计算四个中间值 ===
        if not os.path.exists(os.path.join(vec_dir, "logits.npy")):
            logging.info("Calculating logits, predictions, and features...")
            logits_list, preds_list, golds_list, feats_list = [], [], [], []
            for batch in tqdm(test_loader, desc="Inference"):
                y = batch['labels'].to(args.device)
                batch = {k: v.to(args.device) for k, v in batch.items() if k != 'labels'}
                logit = model(batch)
                feat  = model.features(batch)
                pred  = logit.max(dim=1).indices
                logits_list.append(logit)
                preds_list.append(pred)
                golds_list.append(y)
                feats_list.append(feat)

            logits   = torch.cat(logits_list).float().cpu().numpy()
            preds    = torch.cat(preds_list).cpu().numpy()
            golds    = torch.cat(golds_list).cpu().numpy()
            features = torch.cat(feats_list).float().cpu().numpy()

            np.save(os.path.join(vec_dir, "logits.npy"),   logits)
            np.save(os.path.join(vec_dir, "preds.npy"),    preds)
            np.save(os.path.join(vec_dir, "golds.npy"),    golds)
            np.save(os.path.join(vec_dir, "features.npy"), features)
        else:
            logging.info(f"Loading pre-calculated logits/preds/golds/features from {vec_dir}")
            logits   = np.load(os.path.join(vec_dir, "logits.npy"))
            preds    = np.load(os.path.join(vec_dir, "preds.npy"))
            golds    = np.load(os.path.join(vec_dir, "golds.npy"))
            features = np.load(os.path.join(vec_dir, "features.npy"))

        # —— 基线副本，供各探测器独立使用，避免互相污染 ——
        base_logits, base_preds, base_golds, base_features = logits, preds, golds, features
        ID_metrics = custom_metrics(base_preds, base_golds)
        logging.info(f"Test Accuracy(Macro): {ID_metrics.get('macro avg', ID_metrics)}")
        # ==== 全局 LLM 指标（与 detector 无关，只算一次） ====
        global_llm_ood_acc = None
        global_llm_id_confirm_acc = None
        global_llm_preds_full = None           # LLM 对每条样本的 OOD 判定（0/1）
        global_llm_id_confirm_full = None      # LLM 对“ID样本”的标签确认（其余置空）

        if getattr(args, "llm_ood", False) and test_texts is not None \
           and test_labels_str is not None and known_label_list is not None:

            # 1) LLM 的 OOD 判定与准确率
            global_llm_preds_full = llm_ood_eval(args, test_texts, known_label_list, args.output_dir)  # list[int 0/1]
            gt_ood = np.array([1 if lbl == 'ood' else 0 for lbl in test_labels_str], dtype=int)
            global_llm_ood_acc = float((np.array(global_llm_preds_full, dtype=int) == gt_ood).mean() * 100.0)

            # 2) LLM 的“ID标签确认”与准确率（仅对真值为 ID 的样本）
            from utils import llm_id_confirm_eval  # 你前面新增的函数
            id_idx = [i for i, lbl in enumerate(test_labels_str) if lbl != 'ood']
            id_texts = [test_texts[i] for i in id_idx]
            # baseline 的类别名：用 logits.argmax 的 preds 对应 known_label_list
            id_predlabs = [known_label_list[int(preds[i])] for i in id_idx]

            id_flags = llm_id_confirm_eval(args, id_texts, id_predlabs, known_label_list, args.output_dir)  # 1=YES,0=NO
            # 还原成全长；GT=OOD 的位置置空字符串，方便导表
            global_llm_id_confirm_full = [np.nan] * len(test_texts)
            for k, i in enumerate(id_idx):
                global_llm_id_confirm_full[i] = int(id_flags[k])

            # 计算 ID 确认准确率：真值类==baseline预测类 -> 期望 LLM 回答 YES(1)，否则 NO(0)
            agree_target = [1 if test_labels_str[i] == id_predlabs[k] else 0 for k, i in enumerate(id_idx)]
            global_llm_id_confirm_acc = float(
                (np.array(id_flags, dtype=int) == np.array(agree_target, dtype=int)).mean() * 100.0
            ) if agree_target else np.nan
        # ==== 全局 LLM 指标结束 ====



    logging.info("STAGE 2: Creating OOD Detectors")
    detectors = {
        # "TemperatureScaling": TemperatureScaling(model),
        # "LogitNorm": LogitNorm(model),
        # "OpenMax": OpenMax(model),
        # "Entropy": Entropy(model),
        # "Mahalanobis": Mahalanobis(model.features, eps=0.0),
        # "KLMatching": KLMatching(model),
        "MaxSoftmax": MaxSoftmax(model),
        "EnergyBased": EnergyBased(model),
        # "MaxLogit": MaxLogit(model)
    }

    logging.info(f"> Fitting {len(detectors)} detectors")
    for name, detector in detectors.items():
        logging.info(f"--> Fitting {name}")
        # --- 改动 4: 使用上面解包出来的局部变量 loader_in_train ---
        detector.fit(loader_in_train, device=args.device)

    logging.info(f"STAGE 3: Evaluating {len(detectors)} detectors.")
    results = []
    with torch.no_grad():
        for detector_name, detector in detectors.items():
            logging.info(f"> Evaluating {detector_name}")
            metrics = OODMetrics()
            scores = []
            # --- 改动 5: 再次使用 test_loader ---
            for batch in tqdm(test_loader, desc=f"Evaluating {detector_name}"):
                y = batch['labels'].to(args.device)
                batch = {i: v.to(args.device) for i, v in batch.items() if i != 'labels'}
                score = detector(batch)
                metrics.update(score, y)
                scores.append(score)
            
            # r = {"Detector": detector_name}
            r = {"Detector": detector_name, "LLM_OOD_Accuracy": np.nan} 
            r.update(metrics.compute())

            if global_llm_ood_acc is not None:
                r["LLM_OOD_Accuracy"] = global_llm_ood_acc
            if global_llm_id_confirm_acc is not None:
                r["LLM_ID_Confirm_Accuracy"] = global_llm_id_confirm_acc

            scores = torch.concat(scores).detach().cpu().numpy()
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min())

            if detector_name == 'Vim':
                new_logits = np.concatenate([logits, scores.reshape(scores.shape[0], 1)], axis=1)
                new_preds = new_logits.argmax(axis=1)
                new_preds[new_preds==logits.shape[-1]] = -1
                preds = new_preds
                r.update(custom_metrics(preds, golds, norm_scores))
                final_preds = copy.deepcopy(preds)
            else:
                r.update(custom_metrics(preds, golds, norm_scores))
                final_preds = copy.deepcopy(preds)
                final_preds[norm_scores > 0.5] = -1
            
            # === LLM OOD + 导出统一表（仅在选定的探测器上做一次） ===
            if (
                getattr(args, "llm_ood", False)
                and test_texts is not None and test_labels_str is not None and known_label_list is not None
            ):
                # 1) LLM 的 OOD 判定（全量；会用缓存，不会重复计费）
                llm_preds = llm_ood_eval(args, test_texts, known_label_list, args.output_dir)  # 0=ID, 1=OOD

                # 2) 当前 detector 的 OOD 判定（0=ID, 1=OOD）
                thr = float(getattr(args, "ood_threshold", 0.5))
                det_ood_pred = (norm_scores > thr).astype(int).tolist()

                # 3) 仅对“当前 detector 判为 ID”的样本做 LLM 标签确认；OOD 行 NaN
                plm_pred_labels = [
                    (known_label_list[int(preds[i])] if det_ood_pred[i] == 0 else np.nan)
                    for i in range(len(test_texts))
                ]
                id_idx      = [i for i, z in enumerate(det_ood_pred) if z == 0]
                id_texts    = [test_texts[i] for i in id_idx]
                id_predlabs = [plm_pred_labels[i] for i in id_idx]  # 这里不会是 NaN

                from utils import llm_id_confirm_eval
                id_flags = llm_id_confirm_eval(args, id_texts, id_predlabs, known_label_list, args.output_dir)  # 1=YES, 0=NO

                # 回填到全长：仅 ID 行有 0/1；OOD 行 NaN
                llm_id_confirm = [np.nan] * len(test_texts)
                for k, i in enumerate(id_idx):
                    llm_id_confirm[i] = int(id_flags[k])

                # 4) 两个 LLM 指标（写入 r，同时也可放进明细表）
                gt_ood       = np.array([1 if lbl == 'ood' else 0 for lbl in test_labels_str], dtype=int)
                llm_ood_acc  = float((np.array(llm_preds, dtype=int) == gt_ood).mean() * 100.0)

                agree_target = [1 if test_labels_str[i] == plm_pred_labels[i] else 0 for i in id_idx]
                id_confirm_acc = float(
                    (np.array(id_flags, dtype=int) == np.array(agree_target, dtype=int)).mean() * 100.0
                ) if len(agree_target) > 0 else np.nan

                r["LLM_OOD_Accuracy"] = llm_ood_acc
                r["LLM_ID_Confirm_Accuracy"] = id_confirm_acc

                # 4.1) 官方（探测器）这一行的来源标记
                r["decision_source"] = "plm_ood"

                # 4.2) 追加一行“大模型覆盖”的整体结果到 results
                # 构造 LLM 覆盖后的最终预测：1=OOD -> 置为 -1
                final_preds_llm = preds.copy()
                final_preds_llm[np.array(llm_preds, dtype=int) == 1] = -1

                # 计算 golds_num（ID=类索引，OOD=-1）
                golds_num = np.array([-1 if lbl == 'ood' else known_label_list.index(lbl) for lbl in test_labels_str], dtype=int)

                # 计算总体准确率（与项目里 ACC 对齐：百分比）
                overall_acc_llm = float((final_preds_llm == golds_num).mean() * 100.0)

                # 复制官方结果为一份 LLM 覆盖行，并覆盖/补充关键字段
                r_llm = dict(r)
                r_llm["decision_source"] = "llm_override"
                # 若你的 results->DataFrame 是用 r['accuracy'] 派生 'ACC'，这里同步设置
                r_llm["accuracy"] = overall_acc_llm

                # 没有 LLM 连续置信度时，曲线类指标无意义；若 r 中本就有，可显式置 NaN（安全起见加判断）
                for k in ("AUROC", "AUPR", "auroc", "aupr"):
                    if k in r_llm:
                        r_llm[k] = np.nan

                # 把“大模型覆盖”这一行塞进 results（与官方行并列）
                try:
                    results.append(r_llm)
                except NameError:
                    # 如果外层没有 results（极少数情况），就临时写一个附加文件以免丢失
                    tmp_df = pd.DataFrame([r_llm])
                    tmp_path = os.path.join(args.output_dir, f"results_llm_override_{detector_name}.csv")
                    tmp_df.to_csv(tmp_path, index=False)
                    print("[Saved LLM override row]", tmp_path)

                # 5) 导出“当前 detector”的明细表 —— 文件名带上 detector_name
                rows = []
                for i in range(len(test_texts)):
                    rows.append({
                        "text":             test_texts[i],
                        "label":            test_labels_str[i],
                        "llm_pred":         int(llm_preds[i]),        # LLM 的 OOD (0/1)
                        "det_ood_pred":     int(det_ood_pred[i]),     # 当前 detector 的 OOD (0/1)
                        "ood_score":        float(norm_scores[i]),    # 当前 detector 的分数
                        "plm_pred":         plm_pred_labels[i],       # 仅 ID 行类名；OOD 行 NaN
                        "llm_id_confirm":   llm_id_confirm[i],        # 仅 ID 行 0/1；OOD 行 NaN
                        "llm_ood_acc":      llm_ood_acc,              # 整表统一值
                        "detector":         detector_name,
                        "decision_source":  "plm_ood",                # 明细里也标注来源
                    })
                out_path = os.path.join(args.output_dir, f"ood_eval_{detector_name}.csv")
                pd.DataFrame(
                    rows,
                    columns=["text","label","llm_pred","det_ood_pred","ood_score","plm_pred","llm_id_confirm","llm_ood_acc","detector","decision_source"]
                ).to_csv(out_path, index=False)
                print("[Saved]", out_path)

            # === LLM OOD + 导出统一表结束 ===


            results.append(r)
            np.save(args.case_path + f'/{detector_name}_preds.npy', final_preds)
            np.save(args.case_path + f'/{detector_name}_golds.npy', golds)
            np.save(args.case_path + f'/{detector_name}_features.npy', features)

    # df = pd.DataFrame(results)
    # mean_scores = df.groupby(["Detector"]).mean() * 100
    # mean_scores['args'] = json.dumps(vars(args), ensure_ascii=False)
    # logging.info(mean_scores.sort_values("AUROC").to_csv(float_format="%.2f"))
    # mean_scores.to_csv(args.metric_file, sep='\t')


    df_to_save = pd.DataFrame(results)
    df_to_save['method'] = 'PLM_OOD-LLM'
    df_to_save['K-F1'] = df_to_save['K-f1']
    df_to_save['N-F1'] = df_to_save['N-f1']
    df_to_save['F1'] = df_to_save['f1-score']
    df_to_save['ACC'] = df_to_save['accuracy']
    def func(args, detecor):
        args['Detecor'] = detecor
        return json.dumps(args)

    df_to_save['args'] = df_to_save['Detector'].apply(lambda x: func(vars(args), x))
    cols = ['method','dataset','known_cls_ratio','labeled_ratio','cluster_num_factor','seed',
        'ACC','F1','K-F1','N-F1','LLM_OOD_Accuracy','args']
    
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