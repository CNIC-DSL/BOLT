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
import math

from datetime import datetime

RESULTS_DIR: Path = None
SUMMARY_CSV: Path = None
SEEN_JSON: Path = None
LOG_DIR: Path = None

WORKDIR: Path = None
DATA_DIR: Path = None
MODEL_DIR: Path = None


import json
import math


def _is_numlike(x):

    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_missing(x):

    if x is None:
        return True
    return isinstance(x, float) and math.isnan(x)


def safe_equal(a, b):

    if _is_missing(a) and _is_missing(b):
        return True

    if _is_missing(a) or _is_missing(b):
        return False

    if _is_numlike(a) and _is_numlike(b):
        return float(a) == float(b)

    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()

    if isinstance(a, str) and _is_numlike(b):
        try:
            return float(a) == float(b)
        except ValueError:
            return False
    if isinstance(b, str) and _is_numlike(a):
        try:
            return float(b) == float(a)
        except ValueError:
            return False

    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)

    try:
        aj = json.dumps(a, sort_keys=True, ensure_ascii=False)
        bj = json.dumps(b, sort_keys=True, ensure_ascii=False)
        return aj == bj
    except (TypeError, ValueError):
        pass

    return str(a) == str(b)


def compare_common_keys(dict1: dict, dict2: dict, common_keys) -> bool:
    common_keys = set(dict1.keys()) & set(dict2.keys()) & set(common_keys)
    for key in common_keys:
        v1, v2 = dict1[key], dict2[key]

        if isinstance(v1, (int, float)) or isinstance(v2, (int, float)):
            try:
                if float(v1) != float(v2):
                    print(f"不同值: key={key}, dict1={v1}, dict2={v2}")
                    return False
            except (ValueError, TypeError):
                print(f"无法转换: key={key}, dict1={v1}, dict2={v2}")
                return False
        else:

            if v1 != v2:
                print(f"不同值: key={key}, dict1={v1}, dict2={v2}")
                return False
    return True


# def set_paths(results_dir: str, logs_dir: str, result_file: str):
#     global RESULTS_DIR, SUMMARY_CSV, SEEN_JSON, LOG_DIR
#     RESULTS_DIR = Path(results_dir)
#     RESULTS_DIR.mkdir(parents=True, exist_ok=True)
#     SUMMARY_CSV = RESULTS_DIR / f"summary_{result_file}.csv"
#     SEEN_JSON = RESULTS_DIR / f"seen_index_{result_file}.json"
#     LOG_DIR = Path(logs_dir)
#     LOG_DIR.mkdir(parents=True, exist_ok=True)
def set_paths(results_dir: str, logs_dir: str, result_file: str, workdir: str, data_dir: str, model_dir: str):
    global RESULTS_DIR, SUMMARY_CSV, SEEN_JSON, LOG_DIR, WORKDIR, DATA_DIR, MODEL_DIR
    WORKDIR = Path(workdir).resolve()
    DATA_DIR = Path(data_dir).resolve()
    MODEL_DIR = Path(model_dir).resolve()

    RESULTS_DIR = Path(results_dir).resolve()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_CSV = RESULTS_DIR / f"summary_{result_file}.csv"
    SEEN_JSON = RESULTS_DIR / f"seen_index_{result_file}.json"

    LOG_DIR = Path(logs_dir).resolve()
    LOG_DIR.mkdir(parents=True, exist_ok=True)


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
            csv.writer(f).writerow(
                [
                    "method",
                    "dataset",
                    "known_cls_ratio",
                    "labeled_ratio",
                    "cluster_num_factor",
                    "seed",
                    "K",
                    "ACC",
                    "H-Score",
                    "K-ACC",
                    "N-ACC",
                    "ARI",
                    "NMI",
                    "args",
                ]
            )


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


def summary_bucket_path(task: str, dataset: str, known: float, labeled: float) -> Path:
    return Path(f"results/{task}/{dataset}/{labeled}/{known}")


def args_equal_ignore_gpu(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    aa = {k: v for k, v in a.items() if k != "gpu_id"}
    bb = {k: v for k, v in b.items() if k != "gpu_id"}
    return json.dumps(aa, sort_keys=True, ensure_ascii=False) == json.dumps(
        bb, sort_keys=True, ensure_ascii=False
    )


def already_done_via_bucket(
    task: str, dataset: str, known: float, labeled: float, new_args: Dict[str, Any]
) -> bool:
    bucket = summary_bucket_path(task, dataset, known, labeled)
    if not bucket.exists():
        return False
    for p in bucket.glob("*.csv"):
        try:
            with p.open("r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    j = row.get("args")
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


def collect_latest_result(
    default_outputs_glob: str, args_json: Dict[str, Any]
) -> Optional[dict]:
    from glob import glob

    candidates = sorted(
        glob(default_outputs_glob, recursive=True),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    for p in candidates:
        try:
            with open(p, "r", newline="") as f:
                rows = list(csv.DictReader(f))
                if not rows:
                    continue
                row = rows[-1]
                col_list = (
                    [
                        "method",
                        "dataset",
                        "known_cls_ratio",
                        "labeled_ratio",
                        "cluster_num_factor",
                        "seed",
                        "K",
                        "ACC",
                        "F1",
                        "K-F1",
                        "N-F1",
                        "args",
                    ]
                    if args_json["task"] == "openset"
                    else [
                        "method",
                        "dataset",
                        "known_cls_ratio",
                        "labeled_ratio",
                        "cluster_num_factor",
                        "seed",
                        "K",
                        "ACC",
                        "H-Score",
                        "K-ACC",
                        "N-ACC",
                        "ARI",
                        "NMI",
                        "args",
                    ]
                )
                for k in col_list:
                    row.setdefault(
                        k,
                        args_json.get(
                            k,
                            (
                                ""
                                if k != "args"
                                else json.dumps(args_json, ensure_ascii=False)
                            ),
                        ),
                    )
                if not row.get("args"):
                    row["args"] = json.dumps(args_json, ensure_ascii=False)
                return row
        except Exception:
            continue

    bucket = summary_bucket_path(
        args_json["task"],
        args_json["dataset"],
        args_json["known_cls_ratio"],
        args_json["labeled_ratio"],
    )
    for p in sorted(
        bucket.glob("*.csv"), key=lambda pp: os.path.getmtime(pp), reverse=True
    ):
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


def write_summary(row: dict):
    ensure_summary_header()
    with SUMMARY_CSV.open("a", newline="") as f:
        content = [
            v if i in ["method", "dataset", "args"] else f2(v) for i, v in row.items()
        ]
        csv.writer(f).writerow(content)

    print(f"[OK] Appended to {SUMMARY_CSV}")
    print(datetime.now().strftime("%H:%M:%S"))



def make_base_args(
    task: str,
    method: str,
    dataset: str,
    known: float,
    labeled: float,
    fold_type: str,
    fold_num: int,
    fold_idx: int,
    seed: int,
    c_factor: float,
    gpu_id: Optional[int],
    per_method_cfg: Optional[str],
    output_base: str,
    num_pretrain_epochs: int,
    num_train_epochs: int,
    method_specs: dict,
) -> Dict[str, Any]:

    out_base = output_base or f"./outputs/{task}/{method}"
    subname = f"{dataset}_{known}_{labeled}_{fold_type}_{fold_idx}_{seed}"
    args_json = {
        "task": task,
        "config": per_method_cfg or "",
        "dataset": dataset,
        "known_cls_ratio": float(known),
        "labeled_ratio": float(labeled),
        "fold_idx": int(fold_idx),
        "fold_num": int(fold_num),
        "fold_type": fold_type,
        "seed": int(seed),
        "gpu_id": (gpu_id if gpu_id is not None else -1),
        "method": method,
        "data_dir": "./data",
        "output_base_dir": out_base,
        "output_subdir": subname,
        "cluster_num_factor": float(c_factor),
        "result_dir": f"{out_base}/{subname}",
        "results_file_name": "results.csv",
        "K": 0,
        "num_pretrain_epochs": int(num_pretrain_epochs),
        "num_train_epochs": int(num_train_epochs),
        "model_dir": str(MODEL_DIR) if MODEL_DIR is not None else "",
    }
    if method in method_specs:
        for i, v in method_specs[method].items():
            args_json[i] = v

    return args_json


# def run_stage(
#     cli: List[str],
#     args_json: Dict[str, Any],
#     gpu_id: Optional[int],
#     dry_run: bool,
#     log_file: Path,
# ) -> int:
#     env = os.environ.copy()
#     env["ARGS_JSON"] = json.dumps(args_json, ensure_ascii=False)
#     if gpu_id is not None:
#         env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#     if dry_run:
#         print("[DRY-RUN]", " ".join(cli))
#         print(
#             "          ARGS_JSON=",
#             (
#                 (env["ARGS_JSON"][:240] + "...")
#                 if len(env["ARGS_JSON"]) > 240
#                 else env["ARGS_JSON"]
#             ),
#         )
#         return 0
#     with log_file.open("a", encoding="utf-8") as lf:
#         lf.write(f"# CMD: {' '.join(cli)}\n# ARGS_JSON: {env['ARGS_JSON']}\n\n")
#         lf.flush()
#         proc = subprocess.Popen(cli, stdout=lf, stderr=subprocess.STDOUT, env=env)
#         return proc.wait()

def run_stage(
    cli: List[str],
    args_json: Dict[str, Any],
    gpu_id: Optional[int],
    dry_run: bool,
    log_file: Path,
) -> int:
    env = os.environ.copy()
    env["ARGS_JSON"] = json.dumps(args_json, ensure_ascii=False)

    # 三根目录（子进程也能读到）
    if WORKDIR is not None:
        env["BOLT_OUTPUT_DIR"] = str(WORKDIR)
    if DATA_DIR is not None:
        env["BOLT_DATA_DIR"] = str(DATA_DIR)
    if MODEL_DIR is not None:
        env["BOLT_MODEL_DIR"] = str(MODEL_DIR)

        # HuggingFace 统一落到 MODEL_DIR
        env.setdefault("HF_HOME", str(MODEL_DIR))
        env.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_DIR / "hub"))
        env.setdefault("TRANSFORMERS_CACHE", str(MODEL_DIR / "transformers"))
        env.setdefault("HF_DATASETS_CACHE", str(MODEL_DIR / "datasets"))

    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if dry_run:
        print("[DRY-RUN]", " ".join(cli))
        return 0

    with log_file.open("a", encoding="utf-8") as lf:
        lf.write(f"# CMD: {' '.join(cli)}\n# ARGS_JSON: {env['ARGS_JSON']}\n\n")
        lf.flush()
        proc = subprocess.Popen(
            cli,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(WORKDIR) if WORKDIR is not None else None,   # ✅关键：固定 cwd
        )
        return proc.wait()


def run_combo(
    method: str,
    dataset: str,
    known: float,
    labeled: float,
    fold_type: str,
    fold_num: int,
    fold_idx: int,
    seed: int,
    c_factor: float,
    gpu_id: Optional[int],
    num_pretrain_epochs: int,
    num_train_epochs: int,
    dry_run: bool,
    only_collect: bool,
    method_specs: dict,
) -> Optional[dict]:
    # from cli_gcd import METHOD_REGISTRY_GCD
    # from cli_openset import METHOD_REGISTRY_OPENSET
    from .cli_gcd import METHOD_REGISTRY_GCD
    from .cli_openset import METHOD_REGISTRY_OPENSET

    if method in METHOD_REGISTRY_GCD:
        spec = METHOD_REGISTRY_GCD.get(method)
    else:
        spec = METHOD_REGISTRY_OPENSET.get(method)

    if not spec:
        print(f"[WARN] Unknown method: {method}")
        return None

    task = spec["task"]
    args_json = make_base_args(
        task,
        method,
        dataset,
        known,
        labeled,
        fold_type,
        fold_num,
        fold_idx,
        seed,
        c_factor,
        gpu_id,
        spec.get("config", ""),
        spec.get("output_base", f"./outputs/{task}/{method}"),
        num_pretrain_epochs=num_pretrain_epochs,
        num_train_epochs=num_train_epochs,
        method_specs=method_specs,
    )

    import pandas as pd

    save_result_file = f"results/{args_json['task']}/{method}/results.csv"
    if os.path.exists(save_result_file):

        with open(args_json["config"], "r", encoding="utf-8") as f:
            configs = yaml.safe_load(f)
        dataset_name = args_json["dataset"]
        ds_cfg = {}
        if "dataset_specific_configs" in configs:
            ds_cfg = configs["dataset_specific_configs"].get(dataset_name, {})
        base_cfg = {k: v for k, v in configs.items() if k != "dataset_specific_configs"}
        run_args = {**base_cfg, **ds_cfg, **args_json}

        save_result_df = pd.read_csv(save_result_file)
        save_result_df = save_result_df[~save_result_df["args"].isna()]
        save_result_df["args"] = save_result_df["args"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        save_result_df["fold_idx"] = save_result_df["args"].apply(
            lambda x: int(x["fold_idx"])
        )
        save_result_df["fold_num"] = save_result_df["args"].apply(
            lambda x: int(x["fold_num"])
        )
        save_result_df["fold_type"] = save_result_df["args"].apply(
            lambda x: x["fold_type"]
        )
        if "reg_loss" in args_json:
            save_result_df["reg_loss"] = save_result_df["args"].apply(
                lambda x: x["reg_loss"]
            )
        save_result_df["num_train_epochs"] = save_result_df["args"].apply(
            lambda x: int(x["num_train_epochs"])
        )

        filter_list = [
            "method",
            "dataset",
            "known_cls_ratio",
            "labeled_ratio",
            "cluster_num_factor",
            "seed",
            "fold_idx",
            "fold_num",
            "fold_type",
            "num_train_epochs",
            "backbone",
            "reg_loss",
        ]

        filter_list = list(
            (set(save_result_df.columns) & set(filter_list)) & set(args_json.keys())
        )

        def _vector_mask(series: pd.Series, val):
            s = series

            if val is None or (isinstance(val, float) and pd.isna(val)):
                return s.isna()

            if isinstance(val, str):

                return s.astype(str).str.strip().str.lower().eq(val.strip().lower())

            if isinstance(val, (int, float)) and not isinstance(val, bool):
                s_num = pd.to_numeric(s, errors="coerce")
                return s_num.eq(float(val))

            if isinstance(val, bool):

                return s == val

            return s.apply(lambda x: safe_equal(x, val))

        for col in filter_list:
            val = args_json[col]
            series = save_result_df[col]
            mask = _vector_mask(series, val)

            save_result_df = save_result_df[mask.fillna(False)]
            if save_result_df.empty:

                current_values = list(pd.unique(series.dropna()))
                print(
                    f"[Not Exist]  {method} {dataset} kr={known} lr={labeled} fold={fold_idx} seed={seed} | "
                    f"seen matched: {col}: args: {val}; the current list: {current_values}"
                )
                break

        if len(save_result_df) > 0:

            return None
            for items in save_result_df["args"]:
                if compare_common_keys(run_args, items):
                    print(
                        f"[SKIP] seen matched: {method} {dataset} kr={known} lr={labeled} fold={fold_idx} seed={seed}"
                    )
                    return None

    print(
        f"[RUN ] {dataset} | {method} | kr={known} lr={labeled} fold_type={fold_type} fold_idx={fold_idx} seed={seed} cf={c_factor} | gpu={gpu_id}"
    )
    print(datetime.now().strftime("%H:%M:%S"))
    log_dir = (
        LOG_DIR
        / args_json["task"]
        / method
        / args_json["dataset"]
        / f'kr{args_json["known_cls_ratio"]}'
        / f'lr{args_json["labeled_ratio"]}'
        / f'{args_json["fold_type"]}_{args_json["fold_num"]}_{args_json["fold_idx"]}'
        / f'seed{args_json["seed"]}'
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    all_log = log_dir / "all.log"
    with all_log.open("a", encoding="utf-8") as lf:
        lf.write(
            f"# START COMBO {time.strftime('%F %T')} | {task=} {method=} {dataset=} kr={known} lr={labeled} fold={fold_idx} seed={seed} cf={c_factor}\n"
        )

    for idx, st in enumerate(spec["stages"], 1):
        cli_builder = st["cli_builder"]
        cli = cli_builder(args_json, idx)
        stage_log = log_dir / f"stage{idx}.log"

        print(f"[stage_log ] {stage_log}")

        ret = run_stage(cli, args_json, gpu_id, dry_run, stage_log)

        with all_log.open("a", encoding="utf-8") as lf:
            lf.write(f"# STAGE {idx} -> ret={ret}\n")
        if ret != 0:
            print(f"[FAIL] stage{idx} ret={ret} | see {stage_log}")
            return None

    with all_log.open("a", encoding="utf-8") as lf:
        lf.write(f"# END COMBO {time.strftime('%F %T')}\n\n")

    row = collect_latest_result(f"./results/{task}/{method}/results.csv", args_json)
    if not row:
        print(f"[WARN] Finished but no results found for {method} {dataset}")
        return None
    write_summary(row)
    return row
