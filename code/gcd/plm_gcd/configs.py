# configs.py
import argparse
import os
import logging
import sys
import yaml
import torch

# 1. 定义所有参数 (这是原有的逻辑，我们保留它)
parser = argparse.ArgumentParser()
# --- 新增我们框架所需的参数 ---
parser.add_argument("--config", type=str, default=None, help="Path to the YAML config file.")
# --- 保留并检查所有原有参数 ---
parser.add_argument("--dataset", default="banking", type=str)
parser.add_argument("--data_dir", default="./data", type=str) # 修正默认路径
parser.add_argument("--known_cls_ratio", default=0.25, type=float)
parser.add_argument("--num_pretrain_epochs", default=20, type=int)
parser.add_argument("--num_train_epochs", default=20, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--gpu_id", default='0', type=str)
# device 不再需要，可以由 gpu_id 生成
# parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--root", default="data", type=str) # 这个参数似乎没用到，但暂时保留
parser.add_argument("--output_dir", default="outputs", type=str)
parser.add_argument("--save_results_path", default="results", type=str)
parser.add_argument("--backbone", default="Meta-Llama-3.1-8B-Instruct", type=str)
parser.add_argument("--bert_model", default="./pretrained_models/bert-base-uncased", type=str)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--eval_batch_size", default=128, type=int)
parser.add_argument("--labeled_ratio", default=0.1, type=float)
parser.add_argument("--fold_idx", default=0, type=int)
parser.add_argument("--fold_num", default=5, type=int)
parser.add_argument("--es_patience", type=int, default=3,
                    help="Early stopping patience (in eval steps or epochs when evaluation_strategy='epoch').")
parser.add_argument("--es_min_delta", type=float, default=0.0,
                    help="Minimum improvement to qualify as better.")
parser.add_argument("--metric_for_best", type=str, default="accuracy",
                    help="Metric name from compute_metrics to select best model.")
parser.add_argument("--fold_type", type=str, default="fold", help="", choices=['imbalance_fold', 'fold'])


# 2. 解析初始参数 (主要是为了获取 --config 路径)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# 3. 定义并执行配置注入
def apply_config_updates(args, config_dict, parser):
    """我们标准的配置更新函数"""
    type_map = {action.dest: action.type for action in parser._actions}
    for key, value in config_dict.items():
        if f'--{key}' in sys.argv or not hasattr(args, key):
            continue
        expected_type = type_map.get(key)
        if expected_type and value is not None:
            try:
                # 对布尔值进行特殊处理
                if expected_type is bool:
                    value = str(value).lower() in ('true', '1', 't', 'yes')
                else:
                    value = expected_type(value)
            except (ValueError, TypeError):
                print(f"Warning: Could not cast YAML value '{value}' for key '{key}' to type {expected_type}.")
        setattr(args, key, value)

if args.config:
    with open(args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)
    apply_config_updates(args, yaml_config, parser)
    if 'dataset_specific_configs' in yaml_config:
        dataset_configs = yaml_config['dataset_specific_configs'].get(args.dataset, {})
        apply_config_updates(args, dataset_configs, parser)

# 4. 基于最终的 args，完成所有依赖于参数的设置 (路径生成、环境设置等)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用统一的 output_dir 生成所有子路径
args.model_path = f"./pretrained_models/{args.backbone}"

args.output_dir = f"{args.output_dir}/gcd/plm_gcd/{args.dataset}_{args.known_cls_ratio}_{args.seed}_{args.backbone}"
args.checkpoint_path = os.path.join(args.output_dir, 'checkpoints')
args.log_dir = os.path.join(args.output_dir, 'logs')
args.case_path = os.path.join(args.output_dir, 'case_study')
args.metric_dir = args.save_results_path

# 创建目录
os.makedirs(args.metric_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.case_path, exist_ok=True)
os.makedirs(args.checkpoint_path, exist_ok=True) # checkpoint 目录也需要创建

# metric_file 路径也应使用新的根目录
args.metric_file = os.path.join(args.metric_dir, 'results.csv')

# 5. 配置日志记录器
# 移除之前的 handler，确保日志配置只执行一次
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO, # 通常设置为 INFO 级别
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(args.log_dir, "run.log"), mode="a"), # 用追加模式，记录两阶段日志
        logging.StreamHandler()
    ]
)

logging.info("="*20 + " New Run Initialized " + "="*20)
logging.info(f"Arguments loaded and processed: {args}")

# --- 至此，当其他文件 import args 时，它已经是完全配置好的了 ---
