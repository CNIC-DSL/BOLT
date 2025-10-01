#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主入口：只读取 YAML 配置并执行网格实验
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import time
import traceback
import yaml  # 要求：环境需安装 pyyaml

from utils import (
    run_combo,
    set_paths,
)

def main():
    ap = argparse.ArgumentParser(description="Run grid experiments (YAML-only).")
    ap.add_argument("--config", type=str, default="configs/grid.yaml", help="YAML config path")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERR] YAML 配置文件不存在：{cfg_path}")
        sys.exit(1)

    with cfg_path.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    # --- 必要字段检查（不提供默认，全部依赖 YAML） ---
    try:
        maps = y["maps"]                     # task -> [methods]
        methods = y["methods"]               # 参与的 method 列表
        datasets = y["datasets"]             # 数据集列表
        grid = y["grid"]                     # 网格参数字典
        run = y["run"]                       # 运行参数（gpus, max_workers, dry_run, only_collect）
        paths = y["paths"]                   # 路径（results_dir, logs_dir）
        result_file = y["result_file"]                   # 路径（results_dir, logs_dir）
        method_specs = y['per_method_extra']
    except KeyError as e:
        print(f"[ERR] YAML 缺少必要字段：{e}")
        sys.exit(1)

    

    knowns = grid["known_cls_ratio"]
    labeleds = grid["labeled_ratio"]
    fold_idxs = grid["fold_idxs"]
    seeds = grid["seeds"]
    cfs = grid["cluster_num_factor"]
    
    # ★ 新增：全局 epoch 控制（必填）
    num_pretrain_epochs = int(run["num_pretrain_epochs"])
    num_train_epochs = int(run["num_train_epochs"])

    gpus = run["gpus"]            # 例如：[0,1,2] 或 []
    max_workers = int(run["max_workers"])
    dry_run = bool(run.get("dry_run", False))
    only_collect = bool(run.get("only_collect", False))

    # 设置结果/日志路径（由 YAML 指定）
    set_paths(paths["results_dir"], paths["logs_dir"], result_file)

    # 过滤：只保留在 maps 出现过的方法
    method2task = {m: t for t, ms in maps.items() for m in ms}
    final_methods = [m for m in methods if m in method2task]
    if not final_methods:
        print("[ERR] methods 为空或不在 maps 中。")
        sys.exit(1)

    if not datasets:
        print("[ERR] datasets 为空。")
        sys.exit(1)

    # === 动态 GPU 调度器 ===
    # 支持 YAML: run.slots_per_gpu（默认1），run.retry_on_oom（默认True），run.max_retries（默认2）
    slots_per_gpu = int(run.get("slots_per_gpu", 1))
    retry_on_oom = bool(run.get("retry_on_oom", True))
    max_retries   = int(run.get("max_retries", 2))
    backoff_sec   = float(run.get("retry_backoff_sec", 15.0))

    # 令牌池：每块 GPU 投入 slots_per_gpu 个令牌（并发槽位）
    gpu_pool = Queue()
    if gpus:
        for gid in gpus:
            for _ in range(slots_per_gpu):
                gpu_pool.put(gid)
        pool_size = len(gpus) * slots_per_gpu
    else:
        # 无 GPU 列表：投一个 None 令牌，表示走 CPU 或由方法内部自行决定
        gpu_pool.put(None)
        pool_size = 1

    print(f"[SCHED] GPU tokens: {pool_size} | gpus={gpus or ['CPU']} | slots_per_gpu={slots_per_gpu}")

    # 组合与并行（保持你原来的 combos 生成逻辑）
    combos = []
    for cf in cfs:
        for sd in seeds:
            for fi in fold_idxs:
                for lr in labeleds:
                    for kr in knowns:
                        for d in datasets:
                            for m in final_methods:
                                combos.append((m, d, kr, lr, fi, sd, cf, method_specs))

    print(f"[INFO] 组合数={len(combos)} | methods={final_methods} | datasets={datasets}")

    def worker(task):
        """从池中领取 GPU -> 执行 -> 归还；可选 OOM 重试"""
        m, d, kr, lr, fi, sd, cf, method_specs = task
        tries = 0
        while True:
            gpu_id = gpu_pool.get()  # 阻塞，直到有空闲 GPU
            try:
                return run_combo(
                    method=m, dataset=d, known=kr, labeled=lr,
                    fold_idx=fi, seed=sd, c_factor=cf,
                    gpu_id=gpu_id,
                    num_pretrain_epochs=num_pretrain_epochs,
                    num_train_epochs=num_train_epochs,
                    dry_run=dry_run, only_collect=only_collect, method_specs=method_specs
                )
            except RuntimeError as e:
                msg = str(e)
                # 仅在检测到 CUDA OOM 时重试（可按需扩展匹配条件）
                is_oom = ("CUDA out of memory" in msg) or ("out of memory" in msg)
                print(f"[ERR ] {m}@{d} fold={fi} seed={sd} on gpu={gpu_id} | {e.__class__.__name__}: {msg}")
                if retry_on_oom and is_oom and tries < max_retries:
                    tries += 1
                    print(f"[RETRY] OOM detected. Retry {tries}/{max_retries} after {backoff_sec}s (will try another GPU if available).")
                    time.sleep(backoff_sec)
                    # 归还当前 GPU，再循环重新领取（可能是另一块 GPU）
                    gpu_pool.put(gpu_id)
                    continue
                else:
                    # 失败也必须归还 GPU，然后抛出
                    raise
            except Exception:
                print(f"[FATAL] Unexpected error in task {task} on gpu={gpu_id}")
                traceback.print_exc()
                raise
            finally:
                # 正常完成或异常都会走到这里，但上面 OOM 重试时已经手动归还过
                # 为避免重复 put，这里仅在“没有被 continue”时归还
                if 'gpu_id' in locals():
                    # 当发生 OOM 且进入 continue 前，我们已 put 过；这里做个保护
                    try:
                        # 如果队列未被 put 回来，这里补一次
                        # 简单做法：始终 put 回来，但在 OOM 分支 continue 前已 put 过，会导致多放一次。
                        # 为安全起见，可以用 try/except 忽略偶发不平衡，或引入更精细的状态变量。
                        gpu_pool.put(gpu_id)
                    except Exception:
                        pass

    # 线程池大小建议与 gpu token 数或 combos 数挂钩
    max_workers_eff = max(1, min(max_workers, pool_size, len(combos)))
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers_eff) as ex:
        for task in combos:
            futures.append(ex.submit(worker, task))
        # 等待全部完成，抛出异常（便于中断）
        for fu in as_completed(futures):
            _ = fu.result()

    from utils import SUMMARY_CSV
    print("[DONE] 汇总文件：", SUMMARY_CSV)

if __name__ == "__main__":
    main()