# utils.py (改造后的版本)

from sklearn.metrics import classification_report
import numpy as np
import os
import re
# --- 改动 1: 删除这一行，因为它不再有效 ---
# from load_dataset import train_dataset, dataset_in_test, collate_batch, dataset_in_eval, tokenizer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification
import torch
import os, json, time, hashlib, requests

def compute_metrics(eval_predictions):
    # 这个函数是独立的，不需要改动
    preds, golds = eval_predictions
    preds = np.argmax(preds, axis=1)
    metrics = classification_report(preds, golds, output_dict=True)
    metrics['macro avg'].update({'accuracy': metrics['accuracy']})
    return metrics['macro avg']

def get_best_checkpoint(output_dir):
    # 这个函数是独立的，不需要改动
    checkpoints = [d for d in os.listdir(output_dir) if re.match(r'checkpoint-\d+', d)]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]), reverse=False)
    latest_checkpoint = os.path.join(output_dir, checkpoints[0])
    return latest_checkpoint

# --- 改动 2: 为 create_model 函数添加 tokenizer 参数 ---
def create_model(model_path, num_labels, tokenizer):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_8bit_compute_dtype=torch.float32,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=False,
    )
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        device_map="auto",
        attn_implementation="eager",
        quantization_config=quantization_config,
        torch_dtype=torch.float32, 
    )

    # 使用传入的 tokenizer 参数
    base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8,  mean_resizing=False)

    return base_model


def _read_api_key_from_env(env_name: str) -> str:
    key = os.environ.get(env_name, '')
    if not key:
        raise RuntimeError(f'No API key in env: {env_name}')
    return key

def _hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def _safe_load_cache(path: str):
    items = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    items[rec['key']] = rec['value']
                except Exception:
                    pass
    return items

def _append_cache(path: str, key: str, value: dict):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'key': key, 'value': value}, ensure_ascii=False) + '\n')

def llm_chat_batch(api_base: str, model: str, api_key: str, prompts: list, temperature: float=0.0):
    url = api_base.rstrip('/') + '/chat/completions'
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    outputs = []
    for messages in prompts:
        payload = {'model': model, 'messages': messages, 'temperature': temperature}
        for retry in range(5):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=60)
                if r.status_code == 200:
                    data = r.json()
                    outputs.append(data['choices'][0]['message']['content'])
                    break
                else:
                    time.sleep(1.5 * (retry + 1))
            except Exception:
                time.sleep(1.5 * (retry + 1))
        else:
            outputs.append('')
    return outputs

def normalize_ood_answer(text: str) -> int:
    t = (text or '').strip().lower()
    if 'ood' in t or 'out-of-distribution' in t or 'unknown' in t or 'unseen' in t:
        if 'not ood' in t or 'in-distribution' in t or 'known' in t:
            return 0
        return 1
    if 'id' in t or 'in-distribution' in t or 'known' in t:
        return 0
    if t.startswith('yes'):
        return 1
    if t.startswith('no'):
        return 0
    if 'answer' in t and 'ood' in t:
        return 1
    if 'answer' in t and 'id' in t:
        return 0
    return 1

# === 统一的批处理入口（带缓存） ===
def llm_ood_eval(args, texts, known_labels, out_dir):
    """
    texts: list[str]    测试文本（与 test_loader 顺序一致）
    known_labels: list[str]  仅 ID 类别名称（不含 'ood'）
    return: list[int]   llm_preds（0=ID, 1=OOD）
    """
    os.makedirs(out_dir, exist_ok=True)
    cache_path = args.llm_cache_path
    cache = _safe_load_cache(cache_path)

    api_key = args.llm_api_key or _read_api_key_from_env(args.llm_api_key_env)
    bs = max(1, int(getattr(args, "llm_batch_size", 8)))

    # === 新版 system 提示（英文）===
    sys_msg = (
        "You are an OOD detector for intent classification. "
        "Given a user utterance and the list of KNOWN intent labels, "
        "decide if the utterance belongs to an UNKNOWN intent (OOD) or not. "
        "Answer STRICTLY with one token: 'OOD' or 'ID'."
    )

    preds = []
    hit = 0
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        prompts = []
        keys = []
        pending_texts = []  # 与 prompts 一一对应，便于回写缓存
        for x in batch:
            key = _hash(args.llm_model + '|' + '|'.join(known_labels) + '|' + x)
            keys.append(key)
            if key in cache:
                preds.append(int(cache[key]['pred']))
                hit += 1
            else:
                # ★ 关键修改：user 内容包含 KNOWN LABELS + UTTERANCE + "Answer: "
                prompts.append([
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": f"KNOWN LABELS: {', '.join(known_labels)}\nUTTERANCE: {x}\nAnswer: "}
                ])
                pending_texts.append(x)

        if prompts:
            outs = llm_chat_batch(
                args.llm_api_base, args.llm_model, api_key,
                prompts, temperature=args.llm_temperature
            )
            # outs 与 pending_texts 顺序一致
            for x_text, out_text in zip(pending_texts, outs):
                ans = normalize_ood_answer(out_text)
                preds.append(ans)
                key = _hash(args.llm_model + '|' + '|'.join(known_labels) + '|' + x_text)
                _append_cache(cache_path, key, {"text": x_text, "pred": ans})

    # 为保证顺序一致，再按 texts 重读一次（命中 cache 即可）
    final = []
    cache = _safe_load_cache(cache_path)  # 重新加载，包含本轮新写入
    for x in texts:
        key = _hash(args.llm_model + '|' + '|'.join(known_labels) + '|' + x)
        final.append(int(cache[key]["pred"]))
    return final


def _normalize_yesno(text: str) -> int:
    t = (text or "").strip().lower()
    # yes / y / correct / true -> 1；其余当 0
    if t.startswith("y") or "yes" in t or "correct" in t or "true" in t:
        return 1
    return 0

def llm_id_confirm_eval(args, texts, pred_labels, known_labels, out_dir):
    """
    针对“被基线判为ID”的样本，请 LLM 判断：给定 PREDICTED LABEL，这条文本是否应归为该类？
    返回：list[int]，1=YES(同意该类)，0=NO(不同意)
    """
    os.makedirs(out_dir, exist_ok=True)
    cache_path = args.llm_cache_path
    cache = _safe_load_cache(cache_path)

    api_key = args.llm_api_key or _read_api_key_from_env(args.llm_api_key_env)
    bs = max(1, int(getattr(args, "llm_batch_size", 8)))

    sys_msg = (
        "You are a label verifier for intent classification. "
        "Given the KNOWN labels, a PREDICTED label, and an UTTERANCE, "
        "decide if the utterance should be classified as the predicted label. "
        "Answer STRICTLY with one token: 'YES' or 'NO'."
    )

    preds = []
    for i in range(0, len(texts), bs):
        batch_texts  = texts[i:i+bs]
        batch_labels = pred_labels[i:i+bs]
        prompts, pending_pairs, keys = [], [], []
        for x, lab in zip(batch_texts, batch_labels):
            key = _hash("confirm|" + args.llm_model + '|' + '|'.join(known_labels) + '|' + lab + '|' + x)
            keys.append(key)
            if key in cache:
                preds.append(int(cache[key]['pred']))
            else:
                prompts.append([
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": f"KNOWN LABELS: {', '.join(known_labels)}\nPREDICTED LABEL: {lab}\nUTTERANCE: {x}\nAnswer: "}
                ])
                pending_pairs.append((key, x, lab))

        if prompts:
            outs = llm_chat_batch(
                args.llm_api_base, args.llm_model, api_key,
                prompts, temperature=args.llm_temperature
            )
            for (key, x, lab), out_text in zip(pending_pairs, outs):
                ans = _normalize_yesno(out_text)  # 1/0
                preds.append(ans)
                _append_cache(cache_path, key, {"text": x, "label": lab, "pred": ans})

    # 顺序对齐：按 texts 再读一遍
    final = []
    cache = _safe_load_cache(cache_path)
    for x, lab in zip(texts, pred_labels):
        key = _hash("confirm|" + args.llm_model + '|' + '|'.join(known_labels) + '|' + lab + '|' + x)
        final.append(int(cache[key]["pred"]))
    return final
