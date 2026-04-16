from __future__ import annotations
import os
import sys
from typing import Any, Dict, List, Callable

CliBuilder = Callable[[Dict[str, Any], int], List[str]]


def _common_env(args_json: Dict[str, Any]) -> List[str]:
    bert_model = (
        "./pretrained_models/bert-base-chinese"
        if args_json["dataset"] in ["ecdt", "thucnews"]
        else "./pretrained_models/bert-base-uncased"
    )
    return [
        "--bert_model",
        bert_model,
        "--config",
        str(args_json["config"]),
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
        "--seed",
        str(args_json["seed"]),
    ]


def _epoch_flags(args_json: Dict[str, Any], is_pretrain: bool) -> List[str]:
    return [
        "--num_pretrain_epochs",
        str(args_json["num_pretrain_epochs"]),
        "--num_train_epochs",
        str(args_json["num_train_epochs"]),
    ]
    if is_pretrain:
        return [
            "--num_pretrain_epochs",
            str(args_json["num_pretrain_epochs"]),
            "--num_train_epochs",
            str(args_json["num_train_epochs"]),
        ]
    else:
        return ["--num_train_epochs", str(args_json["num_train_epochs"])]


def cli_tan(args_json: Dict[str, Any], stage: int) -> List[str]:
    pre = f'./outputs/gcd/tan/premodel_{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["labeled_ratio"]}_{args_json["fold_type"]}_{args_json["fold_num"]}{args_json["fold_idx"]}_seed{args_json["seed"]}'
    return [
        sys.executable,
        "code/gcd/baselines/TAN/run.py",
        *_common_env(args_json),
        "--gpu_id",
        str(args_json["gpu_id"]),
        "--pretrain_dir",
        pre,
        "--pretrain",
        *_epoch_flags(args_json, is_pretrain=True),
        "--save_model",
        "--freeze_bert_parameters",
    ]


def cli_loop(args_json: Dict[str, Any], stage: int) -> List[str]:
    pre = f'outputs/gcd/loop/premodel_{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["labeled_ratio"]}_{args_json["fold_type"]}_{args_json["fold_num"]}_{args_json["fold_idx"]}_seed{args_json["seed"]}'
    save = f'outputs/gcd/loop/model_{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["labeled_ratio"]}_{args_json["fold_type"]}_{args_json["fold_num"]}_{args_json["fold_idx"]}_seed{args_json["seed"]}'
    return [
        sys.executable,
        "code/gcd/baselines/LOOP/run.py",
        *_common_env(args_json),
        "--gpu_id",
        str(args_json["gpu_id"]),
        "--pretrain_dir",
        pre,
        "--save_model_path",
        save,
        *_epoch_flags(args_json, is_pretrain=False),
        "--save_premodel",
        "--save_model",
    ]


def cli_glean(args_json: Dict[str, Any], stage: int) -> List[str]:
    pre = f'./outputs/gcd/glean/premodel_{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["labeled_ratio"]}_{args_json["fold_type"]}_{args_json["fold_num"]}_{args_json["fold_idx"]}_seed{args_json["seed"]}'
    save = f'./outputs/gcd/glean/model_{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["labeled_ratio"]}_{args_json["fold_type"]}_{args_json["fold_num"]}_{args_json["fold_idx"]}_seed{args_json["seed"]}'
    cli = [
        sys.executable,
        "code/gcd/baselines/Glean/run.py",
        *_common_env(args_json),
        "--gpu_id",
        str(args_json["gpu_id"]),
        *_epoch_flags(args_json, is_pretrain=False),
        "--save_premodel",
        "--save_model",
        "--feedback_cache",
        "--flag_demo",
        "--flag_filtering",
        "--flag_demo_c",
        "--flag_filtering_c",
        "--pretrain_dir",
        pre,
        "--save_model_path",
        save,
    ]

    if "OPENAI_API_KEY" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
    return cli


def cli_geoid(args_json: Dict[str, Any], stage: int) -> List[str]:
    return [
        sys.executable,
        "code/gcd/baselines/GeoID/run.py",
        *_common_env(args_json),
        *_epoch_flags(args_json, is_pretrain=False),
        "--report_pretrain",
    ]


def cli_dpn(args_json: Dict[str, Any], stage: int) -> List[str]:
    return [
        sys.executable,
        "code/gcd/baselines/DPN/run.py",
        *_common_env(args_json),
        "--gpu_id",
        str(args_json["gpu_id"]),
        *_epoch_flags(args_json, is_pretrain=True),
        "--freeze_bert_parameters",
        "--save_model",
        "--pretrain",
    ]


def cli_deepaligned(args_json: Dict[str, Any], stage: int) -> List[str]:
    return [
        sys.executable,
        "code/gcd/baselines/DeepAligned-Clustering/run.py",
        *_common_env(args_json),
        "--gpu_id",
        str(args_json["gpu_id"]),
        "--freeze_bert_parameters",
        *_epoch_flags(args_json, is_pretrain=True),
        "--save_model",
        "--pretrain",
    ]


def cli_alup(args_json: Dict[str, Any], stage: int) -> List[str]:
    base = f'{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["labeled_ratio"]}_{args_json["fold_type"]}_{args_json["fold_num"]}_{args_json["fold_idx"]}_{args_json["seed"]}'
    pre_sub = f"pretrain/{base}"
    fin_sub = f"finetune/{base}"
    if stage == 1:
        return [
            sys.executable,
            "code/gcd/baselines/ALUP/run.py",
            *_common_env(args_json),
            "--gpu_id",
            str(args_json["gpu_id"]),
            "--do_pretrain_and_contrastive",
            *_epoch_flags(args_json, is_pretrain=True),
            "--output_subdir",
            pre_sub,
        ]
    else:
        return [
            sys.executable,
            "code/gcd/baselines/ALUP/run.py",
            *_common_env(args_json),
            "--gpu_id",
            str(args_json["gpu_id"]),
            "--do_al_finetune",
            *_epoch_flags(args_json, is_pretrain=False),
            "--pretrained_stage1_subdir",
            pre_sub,
            "--output_subdir",
            fin_sub,
            "--save_results",
        ]


def cli_sdc_pre(args_json: Dict[str, Any], stage: int) -> List[str]:
    pre = f'outputs/gcd/sdc/premodels/{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["labeled_ratio"]}_{args_json["fold_type"]}_{args_json["fold_num"]}{args_json["fold_idx"]}_seed{args_json["seed"]}'
    return [
        sys.executable,
        "code/gcd/baselines/SDC/pretrain.py",
        *_common_env(args_json),
        "--gpu_id",
        str(args_json["gpu_id"]),
        "--pretrain_dir",
        pre,
        *_epoch_flags(args_json, is_pretrain=True),
        "--pretrain",
        "--save_model",
    ]


def cli_sdc_run(args_json: Dict[str, Any], stage: int) -> List[str]:
    pre = f'outputs/gcd/sdc/premodels/{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["labeled_ratio"]}_{args_json["fold_type"]}_{args_json["fold_num"]}{args_json["fold_idx"]}_seed{args_json["seed"]}'
    train = f'outputs/gcd/sdc/models/{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["labeled_ratio"]}_{args_json["fold_type"]}_{args_json["fold_num"]}{args_json["fold_idx"]}_seed{args_json["seed"]}'
    return [
        sys.executable,
        "code/gcd/baselines/SDC/run.py",
        *_common_env(args_json),
        "--gpu_id",
        str(args_json["gpu_id"]),
        "--pretrain_dir",
        pre,
        "--train_dir",
        train,
        *_epoch_flags(args_json, is_pretrain=False),
        "--save_model",
    ]


def cli_plm_pre(args_json: Dict[str, Any], stage: int) -> List[str]:
    return [
        sys.executable,
        "code/gcd/plm_gcd/pretrain.py",
        *_common_env(args_json),
        "--gpu_id",
        str(args_json["gpu_id"]),
        *_epoch_flags(args_json, is_pretrain=False),
    ]


def cli_plm_run(args_json: Dict[str, Any], stage: int) -> List[str]:
    return [
        sys.executable,
        "code/gcd/plm_gcd/run.py",
        *_common_env(args_json),
        "--gpu_id",
        str(args_json["gpu_id"]),
        *_epoch_flags(args_json, is_pretrain=False),
    ]


def cli_simple_openset(entry: str) -> CliBuilder:
    def _f(args_json: Dict[str, Any], stage: int) -> List[str]:
        return [
            sys.executable,
            entry,
            "--config",
            str(args_json["config"]),
            "--dataset",
            args_json["dataset"],
            "--gpu_id",
            str(args_json["gpu_id"]),
            *_epoch_flags(args_json, is_pretrain=False),
        ]

    return _f


def cli_tlsa(args_json: Dict[str, Any], stage: int) -> List[str]:
    return [
        sys.executable,
        "code/gcd/baselines/TLSA/main.py",
        *_common_env(args_json),
        "--gpu_id",
        str(args_json["gpu_id"]),
        *_epoch_flags(args_json, is_pretrain=False),
        "--save_results_path",
        "./results/gcd/tlsa",
        "--output_dir",
        str(args_json["result_dir"]),
        "--use_known_labeled_data",
        "--use_novel_clustered_data",
    ]


def cli_llm4openssl(args_json: Dict[str, Any], stage: int) -> List[str]:
    """Call original LLM4OpenSSL main.py (train or eval-test)."""
    mode = "train" if stage == 1 else "eval-test"
    basedir = os.path.abspath("code/gcd/baselines/LLM4OpenSSL")
    bolt_root = os.path.abspath(".")
    model = args_json.get("model_name_or_path", "Meta-Llama-3.1-8B-Instruct")
    py_cmd = (
        f"{sys.executable} main.py"
        f" --base_dir {bolt_root}"
        f" --dataset_name {args_json['dataset']}"
        f" --rate {args_json['known_cls_ratio']}"
        f" --labeled_ratio {args_json['labeled_ratio']}"
        f" --seed {args_json['seed']}"
        f" --gpu_id {args_json['gpu_id']}"
        f" --mode {mode}"
        f" --data_root_dir {bolt_root}/data"
        f" --model_name_or_path {model}"
        f" --num_train_epochs {args_json.get('num_train_epochs', 12)}"
        f" --fold_idx {args_json['fold_idx']}"
        f" --fold_num {args_json['fold_num']}"
        f" --fold_type {args_json['fold_type']}"
    )
    for key in [
        "per_device_train_batch_size", "per_device_eval_batch_size",
        "gradient_accumulation_steps", "learning_rate", "linear_learning_rate",
        "num_semi_warmup_epochs", "num_gen_warmup_epochs", "is_semi", "is_mlp",
        "cca_loss_func", "cca_k", "cca_loss_weight",
    ]:
        if key in args_json:
            py_cmd += f" --{key} {args_json[key]}"
    return ["bash", "-c", f"cd {basedir} && {py_cmd}"]


METHOD_REGISTRY_GCD: Dict[str, Dict[str, Any]] = {
    "tan": {
        "task": "gcd",
        "stages": [{"entry": "code/gcd/baselines/TAN/run.py", "cli_builder": cli_tan}],
        "config": "configs/gcd/tan.yaml",
        "output_base": "./outputs/gcd/tan",
    },
    "loop": {
        "task": "gcd",
        "stages": [
            {"entry": "code/gcd/baselines/LOOP/run.py", "cli_builder": cli_loop}
        ],
        "config": "configs/gcd/loop.yaml",
        "output_base": "./outputs/gcd/loop",
    },
    "glean": {
        "task": "gcd",
        "stages": [
            {"entry": "code/gcd/baselines/Glean/run.py", "cli_builder": cli_glean}
        ],
        "config": "configs/gcd/glean.yaml",
        "output_base": "./outputs/gcd/glean",
    },
    "geoid": {
        "task": "gcd",
        "stages": [
            {"entry": "code/gcd/baselines/GeoID/run.py", "cli_builder": cli_geoid}
        ],
        "config": "configs/gcd/geoid.yaml",
        "output_base": "./outputs/gcd/geoid",
    },
    "dpn": {
        "task": "gcd",
        "stages": [{"entry": "code/gcd/baselines/DPN/run.py", "cli_builder": cli_dpn}],
        "config": "configs/gcd/dpn.yaml",
        "output_base": "./outputs/gcd/dpn",
    },
    "deepaligned": {
        "task": "gcd",
        "stages": [
            {
                "entry": "code/gcd/baselines/DeepAligned-Clustering/run.py",
                "cli_builder": cli_deepaligned,
            }
        ],
        "config": "configs/gcd/deepaligned.yaml",
        "output_base": "./outputs/gcd/deepaligned",
    },
    "alup": {
        "task": "gcd",
        "stages": [
            {"entry": "code/gcd/baselines/ALUP/run.py", "cli_builder": cli_alup},
            {"entry": "code/gcd/baselines/ALUP/run.py", "cli_builder": cli_alup},
        ],
        "config": "configs/gcd/alup.yaml",
        "output_base": "./outputs/gcd/alup",
    },
    "sdc": {
        "task": "gcd",
        "stages": [
            {"entry": "code/gcd/baselines/SDC/pretrain.py", "cli_builder": cli_sdc_pre},
            {"entry": "code/gcd/baselines/SDC/run.py", "cli_builder": cli_sdc_run},
        ],
        "config": "configs/gcd/sdc.yaml",
        "output_base": "./outputs/gcd/sdc",
    },
    "plm_gcd": {
        "task": "gcd",
        "stages": [
            {"entry": "code/gcd/plm_gcd/pretrain.py", "cli_builder": cli_plm_pre},
            {"entry": "code/gcd/plm_gcd/run.py", "cli_builder": cli_plm_run},
        ],
        "config": "configs/gcd/plm_gcd.yaml",
        "output_base": "./outputs/gcd/plm_gcd",
    },
    "tlsa": {
        "task": "gcd",
        "stages": [
            {"entry": "code/gcd/baselines/TLSA/main.py", "cli_builder": cli_tlsa},
        ],
        "config": "configs/gcd/tlsa.yaml",
        "output_base": "./outputs/gcd/tlsa",
    },
    "llm4openssl": {
        "task": "gcd",
        "stages": [
            {
                "entry": "code/gcd/baselines/LLM4OpenSSL/main.py",
                "cli_builder": cli_llm4openssl,
            },
            {
                "entry": "code/gcd/baselines/LLM4OpenSSL/main.py",
                "cli_builder": cli_llm4openssl,
            },
        ],
        "config": "configs/gcd/llm4openssl.yaml",
        "output_base": "./outputs/gcd/llm4openssl",
    },
}
