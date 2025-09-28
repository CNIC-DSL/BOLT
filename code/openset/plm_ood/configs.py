# configs.py (重构后)

import argparse
import os
import logging

def create_parser():
    """
    定义所有命令行参数并返回一个ArgumentParser对象。
    这个函数只负责“定义”，不负责“解析”。
    """
    parser = argparse.ArgumentParser()
    
    # --- 基础参数定义 ---
    parser.add_argument("--dataset_name", default="stackoverflow", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--reg_loss", default="npo", type=str, choices=['normal', 'vos', 'npo'])
    parser.add_argument("--rate", default=0.25, type=float)
    parser.add_argument("--labeled_ratio", default=1.0, type=float)
    parser.add_argument("--n_epochs", default=20, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--fold_idx", default=0, type=int)
    parser.add_argument("--fold_num", default=5, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--root", default="data", type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--backbone", default="Meta-Llama-3.1-8B-Instruct", type=str)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--early_stop_patience", type=int, default=3, help="Patience for early stopping.")
    parser.add_argument("--early_stop_delta", type=float, default=0.0, help="Minimum change to qualify as an improvement for early stopping.")
    return parser

def finalize_config(args):
    """
    接收已经解析完毕的args对象，完成派生路径的计算和日志的设置。
    """
    # --- 派生参数计算 (动态生成路径) ---
    args.data_identity = f"{args.dataset_name}_{args.labeled_ratio}_{args.rate}_{args.fold_num}_{args.fold_idx}"
    run_identity = f"{args.data_identity}{args.backbone}_{args.seed}"
    
    base_path = os.path.join(args.output_dir, args.reg_loss, args.dataset_name, str(args.labeled_ratio), run_identity)

    args.metric_dir = os.path.join(base_path, "metrics")
    args.log_dir = os.path.join(base_path, "logs")
    args.checkpoint_path = base_path
    args.case_path = os.path.join(base_path, "case_study")
    args.vector_path = os.path.join(base_path, "case_study")

    # 创建目录
    os.makedirs(args.metric_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.case_path, exist_ok=True)

    args.metric_file = f"{args.metric_dir}/epoch_{args.n_epochs}_seed_{args.seed}.csv"

    # --- model_path 备用逻辑 ---
    if args.model_path is None:
        logging.warning("model_path not specified, generating a fallback path. It is recommended to specify this in the YAML file.")
        args.model_path = f"./pretrained_models/{args.backbone}"
    else:
        logging.info(f"Using model_path specified in config: {args.model_path}")

    # --- 日志设置 ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{args.log_dir}/epoch_{args.n_epochs}_seed_{args.seed}.log", mode="w"),
            logging.StreamHandler()
        ],
        force=True
    )

    logging.info(f"Final configuration: {args}")
    
    return args