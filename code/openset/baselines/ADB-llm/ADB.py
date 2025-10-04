# ADB.py (已改造)

from model import *
from init_parameter import *
from dataloader import *
from pretrain import *
from util import *
from loss import *
import yaml  # <-- 新增
import sys   # <-- 新增
import json
from util import _read_api_key_from_env, _safe_load_cache, _append_cache, _hash
from util import llm_chat_batch, normalize_ood_answer
import csv

class ModelManager:
    
    def __init__(self, args, data, pretrained_model=None):
        
        self.model = pretrained_model

        if self.model is None:
            self.model = BertForModel.from_pretrained(args.bert_model, cache_dir = "", num_labels = data.num_labels)
            self.restore_model(args)

        # GPU设置移到主程序入口，这里不再需要
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id     
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.best_eval_score = 0
        self.delta = None
        self.delta_points = []
        self.centroids = None

        self.test_results = None
        self.predictions = None
        self.true_labels = None

    def open_classify(self, features, data): # <-- 修正：需要传入 data 对象
        logits = euclidean_metric(features, self.centroids)
        probs, preds = F.softmax(logits.detach(), dim = 1).max(dim = 1)
        euc_dis = torch.norm(features - self.centroids[preds], 2, 1).view(-1)
        preds[euc_dis >= self.delta[preds]] = data.unseen_token_id
        return preds

    def evaluation(self, args, data, mode="eval"):
        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        dataloader = data.eval_dataloader if mode == 'eval' else data.test_dataloader

        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                pooled_output, _ = self.model(input_ids, segment_ids, input_mask)
                preds = self.open_classify(pooled_output, data) # <-- 修正：传入 data

                total_labels = torch.cat((total_labels,label_ids))
                total_preds = torch.cat((total_preds, preds))
        
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        if mode == 'eval':
            cm = confusion_matrix(y_true, y_pred)
            eval_score = F_measure(cm)['f1-score']
            return eval_score

        elif mode == 'test':
            self.predictions = list([data.label_list[idx] for idx in y_pred])
            self.true_labels = list([data.label_list[idx] for idx in y_true])
            
            cm = confusion_matrix(y_true,y_pred)
            results = F_measure(cm)
            acc = round(accuracy_score(y_true, y_pred) * 100, 2)
            results['accuracy'] = acc
            self.test_results = results

            if args.llm_ood:
                known_labels = [l for i, l in enumerate(data.label_list) if i != data.unseen_token_id]
                test_examples = data.test_examples  # Data 里已暴露
                llm_acc = llm_ood_eval(args, data, test_examples, known_labels, args.output_dir)
                self.test_results['LLM_OOD_Accuracy'] = llm_acc
                print(f'[LLM] OOD Accuracy: {llm_acc:.4f}')

            self.save_results(args) # <-- 路径管理已在 save_results 中实现

    def train(self, args, data):     
        criterion_boundary = BoundaryLoss(num_labels = data.num_labels, feat_dim = args.feat_dim)
        self.delta = F.softplus(criterion_boundary.delta)
        optimizer = torch.optim.Adam(criterion_boundary.parameters(), lr = args.lr_boundary)
        self.centroids = self.centroids_cal(args, data)

        wait = 0
        best_delta, best_centroids = None, None

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(data.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                    loss, self.delta = criterion_boundary(features, self.centroids, label_ids)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            self.delta_points.append(self.delta)
            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)
            
            eval_score = self.evaluation(args, data, mode="eval")
            print('eval_score',eval_score)
            
            if eval_score >= self.best_eval_score:
                wait = 0
                self.best_eval_score = eval_score
                best_delta = self.delta
                best_centroids = self.centroids
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break
        
        self.delta = best_delta
        self.centroids = best_centroids

    def class_count(self, labels):
        class_data_num = []
        for l in np.unique(labels):
            num = len(labels[labels == l])
            class_data_num.append(num)
        return class_data_num

    def centroids_cal(self, args, data):
        centroids = torch.zeros(data.num_labels, args.feat_dim).cuda()
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        with torch.set_grad_enabled(False):
            for batch in data.train_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                total_labels = torch.cat((total_labels, label_ids))
                for i in range(len(label_ids)):
                    label = label_ids[i]
                    centroids[label] += features[i]
                
        total_labels = total_labels.cpu().numpy()
        centroids /= torch.tensor(self.class_count(total_labels)).float().unsqueeze(1).cuda()
        return centroids

    def restore_model(self, args):
        # 路径现在由主程序入口的 output_dir 控制
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        self.model.load_state_dict(torch.load(output_model_file))
    
    def save_results(self, args):
        # 路径现在由主程序入口的 output_dir 控制
        results_path = args.output_dir
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        var = [args.dataset, args.known_cls_ratio, args.labeled_ratio, args.seed]
        names = ['dataset', 'known_cls_ratio', 'labeled_ratio', 'seed']
        vars_dict = {k: v for k, v in zip(names, var)}
        vars_dict['args'] = json.dumps(vars(args), ensure_ascii=False)

        # 合并结果；把 LLM_OOD_Accuracy 先拿出来，避免后面强制改列名时报长度不匹配
        results = dict(self.test_results, **vars_dict)
        _llm = results.pop('LLM_OOD_Accuracy', None)

        np.save(os.path.join(results_path, 'centroids.npy'), self.centroids.detach().cpu().numpy())
        np.save(os.path.join(results_path, 'deltas.npy'), self.delta.detach().cpu().numpy())

        results_csv_path = os.path.join(args.save_results_path, 'results.csv')
        os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)

        df_new_row = pd.DataFrame([results])
        df_new_row.columns = ['K-F1', 'N-F1', 'F1', 'ACC', 'dataset', 'known_cls_ratio', 'labeled_ratio', 'seed', 'args']

        # 把大模型指标加回 DataFrame
        if _llm is not None:
            df_new_row['LLM_OOD_Accuracy'] = _llm

        df_new_row['method'] = 'ADB'
        cols = ['method', 'dataset', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor',
                'seed', 'ACC', 'LLM_OOD_Accuracy', 'F1', 'K-F1', 'N-F1', 'args']

        for col in cols:
            if col in df_new_row:
                continue
            if col == 'LLM_OOD_Accuracy':
                df_new_row[col] = self.test_results.get('LLM_OOD_Accuracy', None)
            else:
                df_new_row[col] = getattr(args, col, None)

        df_new_row = df_new_row[cols]

        if not os.path.exists(results_csv_path):
            df_new_row.to_csv(results_csv_path, index=False)
        else:
            df_existing = pd.read_csv(results_csv_path)
            df_updated = pd.concat([df_existing, df_new_row], ignore_index=True)
            df_updated.to_csv(results_csv_path, index=False)

        print('Test results saved to:', results_csv_path)
        print(df_new_row)


def _build_ood_prompt(text, known_labels):
    sys = ("You are an OOD detector for intent classification. "
           "Given a user utterance and the list of KNOWN intent labels, "
           "decide if the utterance belongs to an UNKNOWN intent (OOD) or not. "
           "Answer STRICTLY with one token: 'OOD' or 'ID'.")
    user = (f"KNOWN LABELS: {', '.join(known_labels)}\n"
            f"UTTERANCE: {text}\n"
            f"Answer: ")
    return [
        {"role":"system", "content": sys},
        {"role":"user", "content": user}
    ]

def llm_ood_eval(args, data, test_examples, known_labels, output_dir):
    # 准备缓存
    cache_path = os.path.join(output_dir, args.llm_cache)
    cache = _safe_load_cache(cache_path)
    # api_key = _read_api_key_from_env(args.llm_api_key_env)
    api_key = args.llm_api_key or _read_api_key_from_env(args.llm_api_key_env)


    # 真实 OOD：等于 unseen_token_id 的样本为 1，否则 0
    gt = []
    texts = []
    keys = []
    msg_batches = []
    batch = []

    label2id = {lbl: i for i, lbl in enumerate(data.label_list)}
    unseen_id = data.unseen_token_id

    gt, texts, keys, msg_batches = [], [], [], []
    batch = []

    for ex in test_examples:
        # 取文本：先尝试 ex.text，再退回 ex.text_a
        txt = getattr(ex, 'text', None) or getattr(ex, 'text_a', '')
        # 取标签：优先字符串 label；如果（极少数实现）有 label_id 也兼容一下
        if hasattr(ex, 'label') and ex.label is not None:
            label_id = label2id.get(ex.label, unseen_id)
        elif hasattr(ex, 'label_id'):
            label_id = ex.label_id
        else:
            label_id = unseen_id  # 兜底

        is_ood = 1 if label_id == unseen_id else 0
        gt.append(is_ood)
        texts.append(txt)

        prompt_messages = _build_ood_prompt(txt, known_labels)
        k = _hash(prompt_messages[-1]['content'])
        keys.append(k)

        if k not in cache:
            batch.append(prompt_messages)
            if len(batch) >= args.llm_batch:
                outs = llm_chat_batch(args.llm_api_base, args.llm_model, api_key, batch, args.llm_temperature)
                for pm, out in zip(batch, outs):
                    kk = _hash(pm[-1]['content'])
                    _append_cache(cache_path, kk, {'response': out})
                    cache[kk] = {'response': out}
                batch = []
    # 处理余下的
    if batch:
        outs = llm_chat_batch(args.llm_api_base, args.llm_model, api_key, batch, args.llm_temperature)
        for pm, out in zip(batch, outs):
            kk = _hash(pm[-1]['content'])
            _append_cache(cache_path, kk, {'response': out})
            cache[kk] = {'response': out}

    # 归一化预测
    preds = []
    for k in keys:
        out = cache[k]['response']
        pred = normalize_ood_answer(out)
        preds.append(pred)

    # 计算准确率与简单统计
    correct = sum(int(p == y) for p, y in zip(preds, gt))
    acc = correct / max(1, len(gt))

    # 保存逐样本结果
    out_csv = os.path.join(output_dir, 'llm_ood_predictions.csv')
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['text', 'gt_is_ood', 'llm_pred_is_ood'])
        for t, y, p in zip(texts, gt, preds):
            w.writerow([t, y, p])

    return acc

# --- 核心改造：将参数解析和配置注入逻辑放在主入口 ---
if __name__ == '__main__':
    
    print('Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()

    def apply_config_updates(args, config_dict, parser):
        type_map = {action.dest: action.type for action in parser._actions}
        for key, value in config_dict.items():
            if f'--{key}' in sys.argv or not hasattr(args, key):
                continue
            expected_type = type_map.get(key)
            if expected_type and value is not None:
                try:
                    if expected_type is bool or expected_type.__name__ == 'str_to_bool':
                        value = str(value).lower() in ('true', '1', 't', 'yes')
                    else:
                        value = expected_type(value)
                except (ValueError, TypeError):
                    print(f"Warning: Could not cast YAML value '{value}' for key '{key}' to type {expected_type}.")
            setattr(args, key, value)

    # 执行配置注入
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        apply_config_updates(args, yaml_config, parser)
        if 'dataset_specific_configs' in yaml_config:
            dataset_configs = yaml_config['dataset_specific_configs'].get(args.dataset, {})
            apply_config_updates(args, dataset_configs, parser)
    
    # --- 路径管理和环境设置 ---
    # 使用动态 output_dir (如果命令行未提供，则使用YAML/默认值)
    # 这会覆盖掉 YAML 中的 pretrain_dir 和 save_results_path，实现集中管理
    args.pretrain_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(args.pretrain_dir, exist_ok=True)
    os.makedirs(args.save_results_path, exist_ok=True)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # 开始执行任务
    data = Data(args)

    print('Pre-training begin...')
    manager_p = PretrainModelManager(args, data)
    manager_p.train(args, data)
    print('Pre-training finished!')
    
    manager = ModelManager(args, data, manager_p.model)
    print('Training begin...')
    manager.train(args, data)
    print('Training finished!')
    
    print('Evaluation begin...')
    manager.evaluation(args, data, mode="test")  
    print('Evaluation finished!')
