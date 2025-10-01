# -*- coding: utf-8 -*-
"""
OpenSet 系列：按模型定制的 Python CLI 生成器与注册表（不再调用 shell）
从 scripts/openset/*.sh 中抽取到的 python 入口与参数，映射为等价的 Python 命令行。
"""
from __future__ import annotations
import sys
from typing import Any, Dict, List, Callable

CliBuilder = Callable[[Dict[str, Any], int], List[str]]  # (args_json, stage_idx) -> argv list


# ----------------------------- 公共参数拼装 -----------------------------
def _maybe(v, flag: str) -> List[str]:
    """如果 v 不为 None/空串，则返回 [flag, str(v)]，否则返回空列表。"""
    if v is None:
        return []
    s = str(v)
    return [flag, s] if len(s) > 0 else []


def _epoch_flags(args_json:Dict[str,Any], is_pretrain: bool) -> List[str]:
    """根据阶段返回统一的 epoch 参数"""
    if is_pretrain:
        return ["--num_pretrain_epochs", str(args_json["num_pretrain_epochs"]), "--num_train_epochs", str(args_json["num_train_epochs"])]
    else:
        return ["--num_train_epochs", str(args_json["num_train_epochs"])]
    
def _common_flags(args_json: Dict[str, Any]) -> List[str]:
    """
    OpenSet 通用：config / dataset(or dataset_name) / seed / gpu_id / known ratio(rate)
    注：部分方法用 --dataset，个别（plm_ood）用 --dataset_name；下方各方法单独处理。
    """
    # 额外自定义 flags（例如某些模型特有参数），允许用户从外部传入列表
    extra = list(args_json.get("extra_flags", []))
    output_dir = f"outputs/openset/{args_json['method']}/{args_json['dataset']}_{args_json['labeled_ratio']}_{args_json['known_cls_ratio']}_{args_json['fold_idx']}_{args_json['seed']}"
    # model_name_or_path = "./pretrained_models/bert-base-chinese" if args_json["dataset"] in ['ecdt', 'thucnews'] else "./pretrained_models/bert-base-uncased"
    return [
        "--config", str(args_json["config"]),
        "--seed", str(args_json["seed"]),
        "--gpu_id", str(args_json["gpu_id"]),
        "--dataset", args_json["dataset"],
        "--known_cls_ratio", str(args_json["known_cls_ratio"]),
        "--labeled_ratio", str(args_json["labeled_ratio"]),
        "--fold_idx", str(args_json["fold_idx"]),
        "--output_dir", str(output_dir),
        *extra,
    ]


# ----------------------------- 各方法 CLI 构造 -----------------------------
def cli_ab(args_json: Dict[str, Any], stage: int) -> List[str]:
    """
    入口：code/openset/baselines/AB/code/run.py
    shell 中参数包含：--config --dataset --emb_name --seed --known_cls_ratio --gpu_id --output_dir
    """
    emb_name = args_json.get("emb_name", "sbert")  # 若 YAML 未给，给个兜底
    out_dir = args_json.get(
        "output_dir",
        f'./outputs/openset/ab/{args_json["dataset"]}_{emb_name}_{args_json["known_cls_ratio"]}_{args_json["seed"]}'
    )
    return [
        sys.executable, "code/openset/baselines/AB/code/run.py",
        "--emb_name", emb_name,
        "--output_dir", out_dir,
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]


def cli_adb(args_json: Dict[str, Any], stage: int) -> List[str]:
    """
    入口：code/openset/baselines/ADB/ADB.py
    """
    out_dir = args_json.get(
        "output_dir",
        f'./outputs/openset/adb/{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["seed"]}'
    )
    return [
        sys.executable, "code/openset/baselines/ADB/ADB.py",
        "--output_dir", out_dir,
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]


def cli_clap_stage1(args_json: Dict[str, Any], stage: int) -> List[str]:
    """
    CLAP 第一阶段（finetune）：
    入口：code/openset/baselines/CLAP/finetune/run_kccl.py
    需要：--config --dataset --seed --known_cls_ratio --gpu_id --output_dir
    """
    out_dir = args_json.get(
        "output_dir",
        f'./outputs/openset/clap/{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["seed"]}'
    )
    return [
        sys.executable, "code/openset/baselines/CLAP/finetune/run_kccl.py",
        "--output_dir", out_dir,
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]


def cli_clap_stage2(args_json: Dict[str, Any], stage: int) -> List[str]:
    """
    CLAP 第二阶段（boundary adjustment）：
    入口：code/openset/baselines/CLAP/boundary_adjustment/run_adbes.py
    需要承接 stage1 的输出：--pretrain_dir / --output_dir
    """
    out_dir = args_json.get(
        "output_dir",
        f'./outputs/openset/clap/{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["seed"]}'
    )
    # 若 stage1 另行指定了 finetuned 模型保存目录，可通过 args_json["finetuned_model_path"] 覆盖
    pretrain_dir = args_json.get("finetuned_model_path", out_dir)
    return [
        sys.executable, "code/openset/baselines/CLAP/boundary_adjustment/run_adbes.py",
        
        
        "--output_dir", out_dir,
        "--pretrain_dir", pretrain_dir,
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]


def cli_deepunk(args_json: Dict[str, Any], stage: int) -> List[str]:
    """
    入口：code/openset/baselines/DeepUnk/experiment.py
    shell 中是 known ratio -> --known_cls_ratio
    """
    out_dir = args_json.get(
        "output_dir",
        f'./outputs/openset/deepunk/{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["seed"]}'
    )
    return [
        sys.executable, "code/openset/baselines/DeepUnk/experiment.py",
        
        
        "--output_dir", out_dir,
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]


def cli_doc(args_json: Dict[str, Any], stage: int) -> List[str]:
    """
    入口：code/openset/baselines/DOC/DOC.py
    """
    out_dir = args_json.get(
        "output_dir",
        f'./outputs/openset/doc/{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["seed"]}'
    )
    return [
        sys.executable, "code/openset/baselines/DOC/DOC.py",
        
        
        "--output_dir", out_dir,
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]


def cli_dyen(args_json: Dict[str, Any], stage: int) -> List[str]:
    """
    入口：code/openset/baselines/DyEn/run_main.py
    """
    out_dir = args_json.get(
        "output_dir",
        f'./outputs/openset/dyen/{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["seed"]}'
    )
    return [
        sys.executable, "code/openset/baselines/DyEn/run_main.py",
        
        
        "--output_dir", out_dir,
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]


def cli_knncon(args_json: Dict[str, Any], stage: int) -> List[str]:
    """
    入口：code/openset/baselines/KnnCon/run_main.py
    """
    return [
        sys.executable, "code/openset/baselines/KnnCon/run_main.py",
        
        
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]


def cli_plm_ood_pre(args_json: Dict[str, Any], stage: int) -> List[str]:
    """
    PLM_OOD 第一阶段（pretrain）：
    入口：code/openset/plm_ood/pretrain.py
    shell 中使用 --dataset_name 与 --rate（而不是 --dataset / --known_cls_ratio）
    个别额外参数（如 reg_loss）可通过 args_json["reg_loss"] 透传。
    """
    reg_loss = args_json.get("reg_loss", None)
    argv = [
        sys.executable, "code/openset/plm_ood/pretrain.py",
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]
    argv += _maybe(reg_loss, "--reg_loss")
    return argv


def cli_plm_ood_run(args_json: Dict[str, Any], stage: int) -> List[str]:
    """
    PLM_OOD 第二阶段（train_ood）：
    入口：code/openset/plm_ood/train_ood.py
    """
    reg_loss = args_json.get("reg_loss", None)
    argv = [
        sys.executable, "code/openset/plm_ood/train_ood.py",
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]
    argv += _maybe(reg_loss, "--reg_loss")
    return argv


def cli_scl(args_json: Dict[str, Any], stage: int) -> List[str]:
    """
    入口：code/openset/baselines/SCL/train.py
    shell 中包含了 --cont_loss 与 --sup_cont 常开，我们也默认打开；
    如需关闭，可通过 extra_flags 在外层移除或覆盖。
    """
    return [
        sys.executable, "code/openset/baselines/SCL/train.py",
        
        
        "--cont_loss",
        "--sup_cont",
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False)
    ]


# ----------------------------- 注册表（仅 openset） -----------------------------
METHOD_REGISTRY_OPENSET: Dict[str, Dict[str, Any]] = {
    # 单阶段
    "ab": {
        "task": "openset",
        "stages": [{"entry": "code/openset/baselines/AB/code/run.py", "cli_builder": cli_ab}],
        "config": "configs/openset/ab.yaml",
        "output_base": "./outputs/openset/ab",
    },
    "adb": {
        "task": "openset",
        "stages": [{"entry": "code/openset/baselines/ADB/ADB.py", "cli_builder": cli_adb}],
        "config": "configs/openset/adb.yaml",
        "output_base": "./outputs/openset/adb",
    },
    "clap": {
        "task": "openset",
        "stages": [
            {"entry": "code/openset/baselines/CLAP/finetune/run_kccl.py", "cli_builder": cli_clap_stage1},
            {"entry": "code/openset/baselines/CLAP/boundary_adjustment/run_adbes.py", "cli_builder": cli_clap_stage2},
        ],
        "config": "configs/openset/clap.yaml",
        "output_base": "./outputs/openset/clap",
    },
    "deepunk": {
        "task": "openset",
        "stages": [{"entry": "code/openset/baselines/DeepUnk/experiment.py", "cli_builder": cli_deepunk}],
        "config": "configs/openset/deepunk.yaml",
        "output_base": "./outputs/openset/deepunk",
    },
    "doc": {
        "task": "openset",
        "stages": [{"entry": "code/openset/baselines/DOC/DOC.py", "cli_builder": cli_doc}],
        "config": "configs/openset/doc.yaml",
        "output_base": "./outputs/openset/doc",
    },
    "dyen": {
        "task": "openset",
        "stages": [{"entry": "code/openset/baselines/DyEn/run_main.py", "cli_builder": cli_dyen}],
        "config": "configs/openset/dyen.yaml",
        "output_base": "./outputs/openset/dyen",
    },
    "knncon": {
        "task": "openset",
        "stages": [{"entry": "code/openset/baselines/KnnCon/run_main.py", "cli_builder": cli_knncon}],
        "config": "configs/openset/knncon.yaml",
        "output_base": "./outputs/openset/knncon",
    },
    "plm_ood": {
        "task": "openset",
        "stages": [
            {"entry": "code/openset/plm_ood/pretrain.py", "cli_builder": cli_plm_ood_pre},
            {"entry": "code/openset/plm_ood/train_ood.py", "cli_builder": cli_plm_ood_run},
        ],
        "config": "configs/openset/plm_ood.yaml",
        "output_base": "./outputs/openset/plm_ood",
    },
    "scl": {
        "task": "openset",
        "stages": [{"entry": "code/openset/baselines/SCL/train.py", "cli_builder": cli_scl}],
        "config": "configs/openset/scl.yaml",
        "output_base": "./outputs/openset/scl",
    },
}