#!/usr/bin/env python3
import os
import sys
import time
import yaml
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_gpu_before_torch():
    """Set GPU env before importing torch."""
    gpu_id = None
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu_id' and i + 1 < len(sys.argv):
            gpu_id = sys.argv[i + 1]
            break

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[INFO] Set CUDA_VISIBLE_DEVICES = {gpu_id} before importing torch")

setup_gpu_before_torch()
import torch

from config import get_config, validate_config
from dataloader import TTCLIPDataManager
from trainer import TTCLIPTrainer
from utils import set_seed, setup_logging


def apply_config_updates(args, config_dict, parser):
    """Apply YAML config values to args, without overriding CLI-provided values."""
    type_map = {action.dest: action.type for action in parser._actions}
    for key, value in config_dict.items():
        if f"--{key}" not in sys.argv and hasattr(args, key):
            expected_type = type_map.get(key)
            if expected_type and value is not None:
                try:
                    value = expected_type(value)
                except (TypeError, ValueError):
                    pass
            setattr(args, key, value)


def main():
    """Main entry point."""
    parser = get_config()
    command_args = parser.parse_args()

    # Load YAML config (BOLT framework pattern)
    if command_args.config:
        with open(command_args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
        apply_config_updates(command_args, yaml_config, parser)
        if "dataset_specific_configs" in yaml_config:
            ds_cfg = yaml_config["dataset_specific_configs"].get(command_args.dataset, {})
            apply_config_updates(command_args, ds_cfg, parser)

    # Set LLM environment variables from config
    if getattr(command_args, 'llm_model_name', None):
        os.environ["LLM_MODEL"] = command_args.llm_model_name
    if getattr(command_args, 'api_base', None):
        os.environ["LLM_BASE_URL"] = command_args.api_base
    api_key = getattr(command_args, 'api_key', None)
    if api_key and api_key != "YOUR_API_KEY":
        os.environ["LLM_API_KEY"] = api_key

    args = validate_config(command_args)
    set_seed(args.seed)
    log_dir = getattr(args, 'output_dir', None) or args.save_results_path
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logging(log_file)

    logger.info("TLSA Training Configuration")

    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")

    try:
        logger.info("Loading and preparing data...")
        data_manager = TTCLIPDataManager(args)

        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Total labels: {len(data_manager.all_labels)}")
        logger.info(f"Known labels: {len(data_manager.known_labels)}")
        logger.info(f"Known ratio: {args.known_cls_ratio}")
        logger.info(f"Labeled ratio: {args.labeled_ratio}")
        logger.info(f"Training samples - Labeled: {len(data_manager.train_labeled_examples)}, "
                   f"Unlabeled: {len(data_manager.train_unlabeled_examples)}")

        logger.info("Initializing trainer...")
        trainer = TTCLIPTrainer(args, data_manager, device)

        trainer.train()

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    main()
