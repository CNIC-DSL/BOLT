from __future__ import annotations
import sys
from typing import Any, Dict, List, Callable

CliBuilder = Callable[[Dict[str, Any], int], List[str]]


def _maybe(v, flag: str) -> List[str]:
    if v is None:
        return []
    s = str(v)
    return [flag, s] if len(s) > 0 else []


def _epoch_flags(args_json: Dict[str, Any], is_pretrain: bool) -> List[str]:
    if is_pretrain:
        return [
            "--num_pretrain_epochs",
            str(args_json["num_pretrain_epochs"]),
            "--num_train_epochs",
            str(args_json["num_train_epochs"]),
        ]
    else:
        return ["--num_train_epochs", str(args_json["num_train_epochs"])]


def _common_flags(args_json: Dict[str, Any]) -> List[str]:

    extra = list(args_json.get("extra_flags", []))
    output_dir = f"outputs/openset/{args_json['method']}/{args_json['dataset']}_{args_json['labeled_ratio']}_{args_json['known_cls_ratio']}_{args_json['fold_type']}_{args_json['fold_num']}_{args_json['fold_idx']}_{args_json['seed']}"
    bert_model = (
        "./pretrained_models/bert-base-chinese"
        if args_json["dataset"] in ["ecdt", "thucnews"]
        else "./pretrained_models/bert-base-uncased"
    )

    if args_json["method"].lower() in [
        "ab",
        "deepunk",
        "doc",
        "plm_ood",
        "plm_ood-llm",
        "clap",
    ]:
        return [
            "--config",
            str(args_json["config"]),
            "--seed",
            str(args_json["seed"]),
            "--gpu_id",
            str(args_json["gpu_id"]),
            "--dataset",
            args_json["dataset"],
            "--known_cls_ratio",
            str(args_json["known_cls_ratio"]),
            "--labeled_ratio",
            str(args_json["labeled_ratio"]),
            "--fold_idx",
            str(args_json["fold_idx"]),
            "--fold_num",
            str(args_json["fold_num"]),
            "--fold_type",
            str(args_json["fold_type"]),
            "--output_dir",
            str(output_dir),
            *extra,
        ]
    else:
        return [
            "--config",
            str(args_json["config"]),
            "--seed",
            str(args_json["seed"]),
            "--gpu_id",
            str(args_json["gpu_id"]),
            "--dataset",
            args_json["dataset"],
            "--known_cls_ratio",
            str(args_json["known_cls_ratio"]),
            "--labeled_ratio",
            str(args_json["labeled_ratio"]),
            "--fold_idx",
            str(args_json["fold_idx"]),
            "--fold_num",
            str(args_json["fold_num"]),
            "--fold_type",
            str(args_json["fold_type"]),
            "--output_dir",
            str(output_dir),
            "--bert_model",
            bert_model,
            *extra,
        ]


def cli_ab(args_json: Dict[str, Any], stage: int) -> List[str]:
    emb_name = args_json.get("emb_name", "sbert")
    out_dir = args_json.get(
        "output_dir",
        f'./outputs/openset/ab/{args_json["dataset"]}_{emb_name}_{args_json["known_cls_ratio"]}_{args_json["seed"]}',
    )
    return [
        sys.executable,
        "code/openset/baselines/AB/code/run.py",
        "--emb_name",
        emb_name,
        "--output_dir",
        out_dir,
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]


def cli_adb(args_json: Dict[str, Any], stage: int) -> List[str]:
    out_dir = args_json.get(
        "output_dir",
        f'./outputs/openset/adb/{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["seed"]}',
    )
    return [
        sys.executable,
        "code/openset/baselines/ADB/ADB.py",
        "--output_dir",
        out_dir,
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]


def cli_adb_llm(args_json: Dict[str, Any], stage: int) -> List[str]:
    out_dir = args_json.get(
        "output_dir",
        f'./outputs/openset/adb-llm/{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["seed"]}',
    )
    return [
        sys.executable,
        "code/openset/baselines/ADB-llm/ADB.py",
        "--output_dir",
        out_dir,
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]


def cli_clap_stage1(args_json: Dict[str, Any], stage: int) -> List[str]:
    return [
        sys.executable,
        "code/openset/baselines/CLAP/finetune/run_kccl.py",
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]


def cli_clap_stage2(args_json: Dict[str, Any], stage: int) -> List[str]:
    return [
        sys.executable,
        "code/openset/baselines/CLAP/boundary_adjustment/run_adbes.py",
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]


def cli_deepunk(args_json: Dict[str, Any], stage: int) -> List[str]:
    out_dir = args_json.get(
        "output_dir",
        f'./outputs/openset/deepunk/{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["seed"]}',
    )
    return [
        sys.executable,
        "code/openset/baselines/DeepUnk/experiment.py",
        "--output_dir",
        out_dir,
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]


def cli_doc(args_json: Dict[str, Any], stage: int) -> List[str]:
    out_dir = args_json.get(
        "output_dir",
        f'./outputs/openset/doc/{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["seed"]}',
    )
    return [
        sys.executable,
        "code/openset/baselines/DOC/DOC.py",
        "--output_dir",
        out_dir,
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]


def cli_dyen(args_json: Dict[str, Any], stage: int) -> List[str]:
    out_dir = args_json.get(
        "output_dir",
        f'./outputs/openset/dyen/{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["seed"]}',
    )
    return [
        sys.executable,
        "code/openset/baselines/DyEn/run_main.py",
        "--output_dir",
        out_dir,
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]


def cli_knncon(args_json: Dict[str, Any], stage: int) -> List[str]:
    return [
        sys.executable,
        "code/openset/baselines/KnnCon/run_main.py",
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]


def cli_plm_ood_pre(args_json: Dict[str, Any], stage: int) -> List[str]:
    reg_loss = args_json.get("reg_loss", None)
    argv = [
        sys.executable,
        "code/openset/plm_ood/pretrain.py",
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]
    argv += _maybe(reg_loss, "--reg_loss")
    return argv


def cli_plm_ood_run(args_json: Dict[str, Any], stage: int) -> List[str]:
    reg_loss = args_json.get("reg_loss", None)
    argv = [
        sys.executable,
        "code/openset/plm_ood/train_ood.py",
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]
    argv += _maybe(reg_loss, "--reg_loss")
    return argv


def cli_plm_ood_llm_pre(args_json: Dict[str, Any], stage: int) -> List[str]:
    reg_loss = args_json.get("reg_loss", None)
    if args_json.get("eval_only", False):
        return [sys.executable, "-c", "print('skip plm_ood pretrain (eval_only=True)')"]
    argv = [
        sys.executable,
        "code/openset/plm_ood-llm/pretrain.py",
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]
    argv += _maybe(reg_loss, "--reg_loss")
    return argv


def cli_plm_ood_llm_run(args_json: Dict[str, Any], stage: int) -> List[str]:
    reg_loss = args_json.get("reg_loss", None)
    argv = [
        sys.executable,
        "code/openset/plm_ood-llm/train_ood.py",
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]
    argv += _maybe(reg_loss, "--reg_loss")

    argv += _maybe(args_json.get("backbone"), "--backbone")
    argv += _maybe(args_json.get("model_path"), "--model_path")

    vec = args_json.get("vector_path") or args_json.get("reuse_vectors_from")
    if vec:
        argv += ["--vector_path", str(vec), "--case_path", str(vec)]

    if args_json.get("llm_ood", False):
        argv += ["--llm_ood"]
    argv += _maybe(args_json.get("llm_api_base"), "--llm_api_base")
    argv += _maybe(args_json.get("llm_model"), "--llm_model")
    argv += _maybe(args_json.get("llm_api_key_env"), "--llm_api_key_env")
    argv += _maybe(args_json.get("llm_temperature"), "--llm_temperature")
    argv += _maybe(args_json.get("llm_batch_size"), "--llm_batch_size")
    argv += _maybe(args_json.get("detector_for_table"), "--detector_for_table")
    argv += _maybe(args_json.get("llm_cache_path"), "--llm_cache_path")

    argv += _maybe(args_json.get("ood_threshold"), "--ood_threshold")
    return argv


def cli_scl(args_json: Dict[str, Any], stage: int) -> List[str]:
    return [
        sys.executable,
        "code/openset/baselines/SCL/train.py",
        "--cont_loss",
        "--sup_cont",
        *_common_flags(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
    ]


METHOD_REGISTRY_OPENSET: Dict[str, Dict[str, Any]] = {
    "ab": {
        "task": "openset",
        "stages": [
            {"entry": "code/openset/baselines/AB/code/run.py", "cli_builder": cli_ab}
        ],
        "config": "configs/openset/ab.yaml",
        "output_base": "./outputs/openset/ab",
    },
    "adb": {
        "task": "openset",
        "stages": [
            {"entry": "code/openset/baselines/ADB/ADB.py", "cli_builder": cli_adb}
        ],
        "config": "configs/openset/adb.yaml",
        "output_base": "./outputs/openset/adb",
    },
    "adb-llm": {
        "task": "openset",
        "stages": [
            {
                "entry": "code/openset/baselines/ADB-llm/ADB.py",
                "cli_builder": cli_adb_llm,
            }
        ],
        "config": "configs/openset/adb-llm.yaml",
        "output_base": "./outputs/openset/adb-llm",
    },
    "clap": {
        "task": "openset",
        "stages": [
            {
                "entry": "code/openset/baselines/CLAP/finetune/run_kccl.py",
                "cli_builder": cli_clap_stage1,
            },
            {
                "entry": "code/openset/baselines/CLAP/boundary_adjustment/run_adbes.py",
                "cli_builder": cli_clap_stage2,
            },
        ],
        "config": "configs/openset/clap.yaml",
        "output_base": "./outputs/openset/clap",
    },
    "deepunk": {
        "task": "openset",
        "stages": [
            {
                "entry": "code/openset/baselines/DeepUnk/experiment.py",
                "cli_builder": cli_deepunk,
            }
        ],
        "config": "configs/openset/deepunk.yaml",
        "output_base": "./outputs/openset/deepunk",
    },
    "doc": {
        "task": "openset",
        "stages": [
            {"entry": "code/openset/baselines/DOC/DOC.py", "cli_builder": cli_doc}
        ],
        "config": "configs/openset/doc.yaml",
        "output_base": "./outputs/openset/doc",
    },
    "dyen": {
        "task": "openset",
        "stages": [
            {
                "entry": "code/openset/baselines/DyEn/run_main.py",
                "cli_builder": cli_dyen,
            }
        ],
        "config": "configs/openset/dyen.yaml",
        "output_base": "./outputs/openset/dyen",
    },
    "knncon": {
        "task": "openset",
        "stages": [
            {
                "entry": "code/openset/baselines/KnnCon/run_main.py",
                "cli_builder": cli_knncon,
            }
        ],
        "config": "configs/openset/knncon.yaml",
        "output_base": "./outputs/openset/knncon",
    },
    "plm_ood": {
        "task": "openset",
        "stages": [
            {
                "entry": "code/openset/plm_ood/pretrain.py",
                "cli_builder": cli_plm_ood_pre,
            },
            {
                "entry": "code/openset/plm_ood/train_ood.py",
                "cli_builder": cli_plm_ood_run,
            },
        ],
        "config": "configs/openset/plm_ood.yaml",
        "output_base": "./outputs/openset/plm_ood",
    },
    "plm_ood-llm": {
        "task": "openset",
        "stages": [
            {
                "entry": "code/openset/plm_ood-llm/pretrain.py",
                "cli_builder": cli_plm_ood_llm_pre,
            },
            {
                "entry": "code/openset/plm_ood-llm/train_ood.py",
                "cli_builder": cli_plm_ood_llm_run,
            },
        ],
        "config": "configs/openset/plm_ood-llm.yaml",
        "output_base": "./outputs/openset/plm_ood-llm",
    },
    "scl": {
        "task": "openset",
        "stages": [
            {"entry": "code/openset/baselines/SCL/train.py", "cli_builder": cli_scl}
        ],
        "config": "configs/openset/scl.yaml",
        "output_base": "./outputs/openset/scl",
    },
}
