from __future__ import annotations
import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import time
import traceback
import yaml

from utils import (
    run_combo,
    set_paths,
)


def main():
    ap = argparse.ArgumentParser(description="Run grid experiments (YAML-only).")
    ap.add_argument(
        "--config", type=str, default="configs/grid.yaml", help="YAML config path"
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERR] YAML 配置文件不存在：{cfg_path}")
        sys.exit(1)

    with cfg_path.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    try:
        maps = y["maps"]
        methods = y["methods"]
        datasets = y["datasets"]
        grid = y["grid"]
        run = y["run"]
        paths = y["paths"]
        result_file = y["result_file"]
        method_specs = y["per_method_extra"]
    except KeyError as e:
        print(f"[ERR] YAML 缺少必要字段：{e}")
        sys.exit(1)

    knowns = grid["known_cls_ratio"]
    labeleds = grid["labeled_ratio"]
    fold_idxs = grid["fold_idxs"]
    fold_nums = grid["fold_nums"]
    fold_types = grid["fold_types"]
    seeds = grid["seeds"]
    cfs = grid["cluster_num_factor"]

    num_pretrain_epochs = int(run["num_pretrain_epochs"])
    num_train_epochs = int(run["num_train_epochs"])

    gpus = run["gpus"]
    max_workers = int(run["max_workers"])
    dry_run = bool(run.get("dry_run", False))
    only_collect = bool(run.get("only_collect", False))

    set_paths(paths["results_dir"], paths["logs_dir"], result_file)

    method2task = {m: t for t, ms in maps.items() for m in ms}
    final_methods = [m for m in methods if m in method2task]
    if not final_methods:
        print("[ERR] methods 为空或不在 maps 中。")
        sys.exit(1)

    if not datasets:
        print("[ERR] datasets 为空。")
        sys.exit(1)

    slots_per_gpu = int(run.get("slots_per_gpu", 1))
    retry_on_oom = bool(run.get("retry_on_oom", True))
    max_retries = int(run.get("max_retries", 2))
    backoff_sec = float(run.get("retry_backoff_sec", 15.0))

    gpu_pool = Queue()
    if gpus:
        for gid in gpus:
            for _ in range(slots_per_gpu):
                gpu_pool.put(gid)
        pool_size = len(gpus) * slots_per_gpu
    else:

        gpu_pool.put(None)
        pool_size = 1

    print(
        f"[SCHED] GPU tokens: {pool_size} | gpus={gpus or ['CPU']} | slots_per_gpu={slots_per_gpu}"
    )

    combos = []
    for cf in cfs:
        for sd in seeds:
            for ft in fold_types:
                for fi in fold_idxs:
                    for fn in fold_nums:
                        for lr in labeleds:
                            for kr in knowns:
                                for d in datasets:
                                    for m in final_methods:
                                        combos.append(
                                            (
                                                m,
                                                d,
                                                kr,
                                                lr,
                                                ft,
                                                fn,
                                                fi,
                                                sd,
                                                cf,
                                                method_specs,
                                            )
                                        )

    print(
        f"[INFO] 组合数={len(combos)} | methods={final_methods} | datasets={datasets}"
    )

    def worker(task):
        m, d, kr, lr, ft, fn, fi, sd, cf, method_specs = task
        tries = 0
        while True:
            gpu_id = gpu_pool.get()
            try:
                return run_combo(
                    method=m,
                    dataset=d,
                    known=kr,
                    labeled=lr,
                    fold_type=ft,
                    fold_num=fn,
                    fold_idx=fi,
                    seed=sd,
                    c_factor=cf,
                    gpu_id=gpu_id,
                    num_pretrain_epochs=num_pretrain_epochs,
                    num_train_epochs=num_train_epochs,
                    dry_run=dry_run,
                    only_collect=only_collect,
                    method_specs=method_specs,
                )
            except RuntimeError as e:
                msg = str(e)

                is_oom = ("CUDA out of memory" in msg) or ("out of memory" in msg)
                print(
                    f"[ERR ] {m}@{d} fold={fi} seed={sd} on gpu={gpu_id} | {e.__class__.__name__}: {msg}"
                )
                if retry_on_oom and is_oom and tries < max_retries:
                    tries += 1
                    print(
                        f"[RETRY] OOM detected. Retry {tries}/{max_retries} after {backoff_sec}s (will try another GPU if available)."
                    )
                    time.sleep(backoff_sec)

                    gpu_pool.put(gpu_id)
                    continue
                else:

                    raise
            except Exception:
                print(f"[FATAL] Unexpected error in task {task} on gpu={gpu_id}")
                traceback.print_exc()
                raise
            finally:

                if "gpu_id" in locals():

                    try:

                        gpu_pool.put(gpu_id)
                    except Exception:
                        pass

    max_workers_eff = max(1, min(max_workers, pool_size, len(combos)))
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers_eff) as ex:
        for task in combos:
            futures.append(ex.submit(worker, task))

        for fu in as_completed(futures):
            _ = fu.result()

    from utils import SUMMARY_CSV

    print("[DONE] 汇总文件：", SUMMARY_CSV)


if __name__ == "__main__":
    main()
