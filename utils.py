# -*- coding: utf-8 -*-
"""
通用工具：路径管理、去重、结果收集、阶段执行、单组合执行
"""
from __future__ import annotations
import csv
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, List
import yaml

# 这些全局路径由 set_paths() 在主程序中设置
RESULTS_DIR: Path = None
SUMMARY_CSV: Path = None
SEEN_JSON: Path = None
LOG_DIR: Path = None

def safe_equal(a, b):
    # 尝试把两个值都转成 float
    try:
        return float(a) == float(b)
    except (ValueError, TypeError):
        # 转换失败时，直接比较原值
        return a.lower() == b.lower()


def compare_common_keys(dict1: dict, dict2: dict, common_keys) -> bool:
    """
    比较两个字典在共有 key 上的 value 是否一致。
    - 对 int/float 支持跨类型比较（会尝试强制转换）。
    - 如果无法转换则返回 False。
    """
    common_keys = set(dict1.keys()) & set(dict2.keys()) & set(common_keys)
    for key in common_keys:
        v1, v2 = dict1[key], dict2[key]

        # 如果其中一个是数字类型，尝试把另一个转成 float 比较
        if isinstance(v1, (int, float)) or isinstance(v2, (int, float)):
            try:
                if float(v1) != float(v2):
                    print(f"不同值: key={key}, dict1={v1}, dict2={v2}")
                    return False
            except (ValueError, TypeError):
                print(f"无法转换: key={key}, dict1={v1}, dict2={v2}")
                return False
        else:
            # 普通比较
            if v1 != v2:
                print(f"不同值: key={key}, dict1={v1}, dict2={v2}")
                return False
    return True

def set_paths(results_dir: str, logs_dir: str, result_file: str):
    """由 YAML 指定的路径进行初始化"""
    global RESULTS_DIR, SUMMARY_CSV, SEEN_JSON, LOG_DIR
    RESULTS_DIR = Path(results_dir); RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_CSV = RESULTS_DIR / f"summary_{result_file}.csv" 
    SEEN_JSON = RESULTS_DIR / f"seen_index_{result_file}.json"
    LOG_DIR = Path(logs_dir); LOG_DIR.mkdir(parents=True, exist_ok=True)

def json_sha1(obj: Any) -> str:
    import hashlib
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def load_seen() -> Dict[str, Any]:
    if SEEN_JSON.exists():
        try:
            return json.loads(SEEN_JSON.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_seen(d: Dict[str, Any]) -> None:
    SEEN_JSON.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def ensure_summary_header():
    if not SUMMARY_CSV.exists():
        with SUMMARY_CSV.open("w", newline="") as f:
            csv.writer(f).writerow([
                "method","dataset","known_cls_ratio","labeled_ratio",
                "cluster_num_factor","seed","K",
                "ACC","H-Score","K-ACC","N-ACC","ARI","NMI","args"
            ])

def f2(x):
    try:
        return float(x)
    except Exception:
        return x

def i2(x):
    try:
        return int(float(x))
    except Exception:
        return 0

def summary_bucket_path(task:str, dataset:str, known:float, labeled:float) -> Path:
    return Path(f"results/{task}/{dataset}/{labeled}/{known}")

def args_equal_ignore_gpu(a:Dict[str,Any], b:Dict[str,Any]) -> bool:
    aa = {k:v for k,v in a.items() if k!="gpu_id"}
    bb = {k:v for k,v in b.items() if k!="gpu_id"}
    return json.dumps(aa, sort_keys=True, ensure_ascii=False) == json.dumps(bb, sort_keys=True, ensure_ascii=False)

def already_done_via_bucket(task:str, dataset:str, known:float, labeled:float, new_args:Dict[str,Any]) -> bool:
    """读取 results/{task}/{dataset}/{labeled}/{known} 下任意 CSV，若存在 args 全等（忽略 gpu_id）则跳过"""
    bucket = summary_bucket_path(task, dataset, known, labeled)
    if not bucket.exists():
        return False
    for p in bucket.glob("*.csv"):
        try:
            with p.open("r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    j = row.get("args"); 
                    if not j: 
                        continue
                    try:
                        old = json.loads(j)
                    except Exception:
                        continue
                    if args_equal_ignore_gpu(old, new_args):
                        return True
        except Exception:
            continue
    return False

def collect_latest_result(default_outputs_glob:str, args_json:Dict[str,Any]) -> Optional[dict]:
    """先按 outputs/{task}/{method}/**/results.csv 收集；若没有，再按 bucket 兜底"""
    from glob import glob
    candidates = sorted(glob(default_outputs_glob, recursive=True), key=lambda p: os.path.getmtime(p), reverse=True)
    for p in candidates:
        try:
            with open(p, "r", newline="") as f:
                rows = list(csv.DictReader(f))
                if not rows:
                    continue
                row = rows[-1]
                col_list = ["method","dataset","known_cls_ratio","labeled_ratio","cluster_num_factor","seed","K",
                          "ACC","F1","K-F1","N-F1","args"] if args_json['task'] == 'openset' else ["method","dataset","known_cls_ratio","labeled_ratio","cluster_num_factor","seed","K",
                          "ACC","H-Score","K-ACC","N-ACC","ARI","NMI","args"]
                for k in col_list:
                    row.setdefault(k, args_json.get(k, "" if k!="args" else json.dumps(args_json, ensure_ascii=False)))
                if not row.get("args"):
                    row["args"] = json.dumps(args_json, ensure_ascii=False)
                return row
        except Exception:
            continue
    # 兜底 bucket
    bucket = summary_bucket_path(args_json["task"], args_json["dataset"], args_json["known_cls_ratio"], args_json["labeled_ratio"])
    for p in sorted(bucket.glob("*.csv"), key=lambda pp: os.path.getmtime(pp), reverse=True):
        try:
            with open(p, "r", newline="") as f:
                rows = list(csv.DictReader(f))
                if rows:
                    row = rows[-1]
                    row.setdefault("args", json.dumps(args_json, ensure_ascii=False))
                    return row
        except Exception:
            continue
    return None

def write_summary(row:dict):
    ensure_summary_header()
    with SUMMARY_CSV.open("a", newline="") as f:
        content = [v if i in ['method', 'dataset', 'args'] else f2(v) for i,v in row.items()]
        csv.writer(f).writerow(content)

        # csv.writer(f).writerow([
        #     row.get("method",""),
        #     row.get("dataset",""),
        #     f2(row.get("known_cls_ratio","")),
        #     f2(row.get("labeled_ratio","")),
        #     f2(row.get("cluster_num_factor","")),
        #     i2(row.get("seed","")),
        #     i2(row.get("K","")),
        #     f2(row.get("ACC","")),
        #     f2(row.get("H-Score","")),
        #     f2(row.get("K-ACC","")),
        #     f2(row.get("N-ACC","")),
        #     f2(row.get("ARI","")),
        #     f2(row.get("NMI","")),
        #     row.get("args",""),
        # ])
    # seen = load_seen(); seen[key_hash] = dedup_key; save_seen(seen)
    print(f"[OK] Appended to {SUMMARY_CSV}")

def make_base_args(task:str, method:str, dataset:str, known:float, labeled:float,
                   fold_idx:int, seed:int, c_factor:float, gpu_id: Optional[int],
                   per_method_cfg: Optional[str], output_base: str,
                   # ★ 新增
                   num_pretrain_epochs: int, num_train_epochs: int) -> Dict[str, Any]:
    m_upper = method.upper()
    out_base = output_base or f"./outputs/{task}/{method}"
    subname = f"{dataset}_{known}_{labeled}_fold{fold_idx}_{seed}"
    return {
        "task": task,
        "config": per_method_cfg or "",
        "dataset": dataset,
        "known_cls_ratio": float(known),
        "labeled_ratio": float(labeled),
        "fold_idx": int(fold_idx),
        "seed": int(seed),
        "gpu_id": (gpu_id if gpu_id is not None else -1),
        "method": m_upper,
        "data_dir": "./data",
        "output_base_dir": out_base,
        "output_subdir": subname,
        "cluster_num_factor": float(c_factor),
        "result_dir": f"{out_base}/{subname}",
        "results_file_name": "results.csv",
        "K": 0,
        # ★ 新增：写入到 args，便于结果重现/去重（忽略 gpu_id 仍然成立）
        "num_pretrain_epochs": int(num_pretrain_epochs),
        "num_train_epochs": int(num_train_epochs),
    }

def run_stage(cli:List[str], args_json:Dict[str,Any], gpu_id: Optional[int], dry_run: bool, log_file: Path) -> int:
    env = os.environ.copy()
    env["ARGS_JSON"] = json.dumps(args_json, ensure_ascii=False)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if dry_run:
        print("[DRY-RUN]", " ".join(cli))
        print("          ARGS_JSON=", (env["ARGS_JSON"][:240]+"...") if len(env["ARGS_JSON"])>240 else env["ARGS_JSON"])
        return 0
    with log_file.open("a", encoding="utf-8") as lf:
        lf.write(f"# CMD: {' '.join(cli)}\n# ARGS_JSON: {env['ARGS_JSON']}\n\n")
        lf.flush()
        proc = subprocess.Popen(cli, stdout=lf, stderr=subprocess.STDOUT, env=env)
        return proc.wait()
    

def run_combo(method:str, dataset:str, known:float, labeled:float, fold_idx:int, seed:int, c_factor:float,
              gpu_id: Optional[int],
              # ★ 新增形参
              num_pretrain_epochs: int, num_train_epochs: int,
              dry_run: bool, only_collect: bool) -> Optional[dict]:
    from cli_gcd import METHOD_REGISTRY_GCD  # 避免循环导入
    from cli_openset import METHOD_REGISTRY_OPENSET  # 避免循环导入
    if method in METHOD_REGISTRY_GCD:
        spec = METHOD_REGISTRY_GCD.get(method)
    else:
        spec = METHOD_REGISTRY_OPENSET.get(method)
        
    if not spec:
        print(f"[WARN] Unknown method: {method}"); return None

    task = spec["task"]
    args_json = make_base_args(
        task, method, dataset, known, labeled, fold_idx, seed, c_factor, gpu_id,
        spec.get("config",""), spec.get("output_base", f"./outputs/{task}/{method}"),
        # ★ 传入
        num_pretrain_epochs=num_pretrain_epochs,
        num_train_epochs=num_train_epochs,
    )
    
    import pandas as pd
    save_result_file = f"results/{args_json['task']}/{method}/results.csv"
    if os.path.exists(save_result_file):

        ### 识别去重机制
        with open(args_json['config'], "r", encoding="utf-8") as f:
            configs = yaml.safe_load(f)
        dataset_name = args_json["dataset"]
        ds_cfg = {}
        if "dataset_specific_configs" in configs:
            ds_cfg = configs["dataset_specific_configs"].get(dataset_name, {})
        base_cfg = {k: v for k, v in configs.items() if k != "dataset_specific_configs"}
        run_args = {**base_cfg, **ds_cfg, **args_json}


        save_result_df = pd.read_csv(save_result_file)
        save_result_df = save_result_df[~save_result_df['args'].isna()]
        save_result_df['args'] = save_result_df['args'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        save_result_df['fold_idx'] = save_result_df['args'].apply(lambda x: int(x['fold_idx']))
        save_result_df['num_train_epochs'] = save_result_df['args'].apply(lambda x: int(x['num_train_epochs']))
        filter_list = ['method', 'dataset', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor', 'seed', 'fold_idx', 'num_train_epochs']
        for col in filter_list:
            save_result_df = save_result_df[save_result_df[col].apply(lambda x: safe_equal(x, args_json[col]))]
            if len(save_result_df) == 0:
                cur_values = list(save_result_df[col].unique())
                print(f"[Not Exist]  {method} {dataset} kr={known} lr={labeled} fold={fold_idx} seed={seed} | seen matched: {col}: args: {args_json[col]}; the current list: {cur_values}")
                break

        if len(save_result_df) > 0:
            print(f"[SKIP] seen matched: {method} {dataset} kr={known} lr={labeled} fold={fold_idx} seed={seed}")
            return None
            for items in save_result_df['args']:
                if compare_common_keys(run_args, items):
                    print(f"[SKIP] seen matched: {method} {dataset} kr={known} lr={labeled} fold={fold_idx} seed={seed}")
                    return None

    print(f"[RUN ] {dataset} | {method} | kr={known} lr={labeled} fold={fold_idx} seed={seed} cf={c_factor} | gpu={gpu_id}")

    # seen 索引
    # # 去重键（忽略 gpu_id）
    # dedup_key = {
    #     "method": args_json["method"], "dataset": dataset,
    #     "known_cls_ratio": known, "labeled_ratio": labeled,
    #     "cluster_num_factor": c_factor, "fold_idx": fold_idx, "seed": seed,
    #     "_args_wo_gpu": {k:v for k,v in args_json.items() if k!="gpu_id"},
    # }
    # key_hash = json_sha1(dedup_key)
    # seen = load_seen()
    # if key_hash in seen and not only_collect:
    #     print(f"[SKIP] seen matched: {method} {dataset} kr={known} lr={labeled} fold={fold_idx} seed={seed}")
    #     return None

    # # bucket 去重
    # if already_done_via_bucket(task, dataset, known, labeled, args_json) and not only_collect:
    #     print(f"[SKIP] bucket matched: {method} {dataset} kr={known} lr={labeled} fold={fold_idx} seed={seed}")
    #     return None

    # # 只收集
    # if only_collect:
    #     row = collect_latest_result(f"./results/{task}/{method}/results.csv", args_json)
    #     if row:
    #         write_summary(row, dedup_key, key_hash)
    #     else:
    #         print(f"[WARN] No results for {method} {dataset} (collect mode)")
    #     return row

    # # 执行阶段
    # log_file = LOG_DIR / f"{task}_{method}_{dataset}_{known}_{labeled}_c{c_factor}_fold{fold_idx}_seed{seed}_{int(time.time())}.log"
    # for idx, st in enumerate(spec["stages"], 1):
    #     cli_builder = st["cli_builder"]
    #     cli = cli_builder(args_json, idx)
    #     ret = run_stage(cli, args_json, gpu_id, dry_run, log_file)
    #     if ret != 0:
    #         print(f"[FAIL] stage{idx} ret={ret} | see {log_file}")
    #         return None

    # 分层日志目录：logs/{task}/{method}/{dataset}/krX/lrY/foldZ/seedS/
    log_dir = (
        LOG_DIR
        / args_json["task"]
        / method
        / args_json["dataset"]
        / f'kr{args_json["known_cls_ratio"]}'
        / f'lr{args_json["labeled_ratio"]}'
        / f'fold{args_json["fold_idx"]}'
        / f'seed{args_json["seed"]}'
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    # （可选）汇总一个 all.log，同时每个阶段有独立日志
    all_log = log_dir / "all.log"
    with all_log.open("a", encoding="utf-8") as lf:
        lf.write(f"# START COMBO {time.strftime('%F %T')} | {task=} {method=} {dataset=} kr={known} lr={labeled} fold={fold_idx} seed={seed} cf={c_factor}\n")

    for idx, st in enumerate(spec["stages"], 1):
        cli_builder = st["cli_builder"]
        cli = cli_builder(args_json, idx)
        stage_log = log_dir / f"stage{idx}.log"

        print(f"[stage_log ] {stage_log}")

        ret = run_stage(cli, args_json, gpu_id, dry_run, stage_log)
        # 也把阶段结果尾巴拼进 all.log 方便一处查看
        with all_log.open("a", encoding="utf-8") as lf:
            lf.write(f"# STAGE {idx} -> ret={ret}\n")
        if ret != 0:
            print(f"[FAIL] stage{idx} ret={ret} | see {stage_log}")
            return None

    with all_log.open("a", encoding="utf-8") as lf:
        lf.write(f"# END COMBO {time.strftime('%F %T')}\n\n")

    # # 收集与汇总
    row = collect_latest_result(f"./results/{task}/{method}/results.csv", args_json)
    if not row:
        print(f"[WARN] Finished but no results found for {method} {dataset}")
        return None
    write_summary(row)
    return row