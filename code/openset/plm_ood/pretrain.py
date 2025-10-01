# pretrain.py (最终修正版)

import os
os.environ["WANDB_DISABLED"]="true"
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig
import logging
from sklearn.metrics import classification_report
import numpy as np
from transformers import EarlyStoppingCallback

from utils import get_best_checkpoint, create_model
# --- 改动 1: 导入新的数据加载函数 ---
from load_dataset import load_and_prepare_datasets
from reg_trainer import RegTrainer
# from configs import get_plm_ood_config
from configs import create_parser, finalize_config
import yaml
import sys

def run_pretraining(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    """
    预训练的主函数，包含了所有的核心逻辑。
    """
    # --- 改动 2: 在函数开头调用函数，获取所有数据对象 ---
    logging.info("Loading and preparing datasets for pretraining...")
    data = load_and_prepare_datasets(args)
    # 从字典中解包出我们需要的变量
    tokenizer = data['tokenizer']
    train_dataset = data['train_dataset']
    dataset_in_eval = data['dataset_in_eval']
    collate_batch = data['collate_batch'] # 注意：collate_fn 也从这里获取
    logging.info("Datasets loaded successfully.")
    # --- 改动结束 ---
    
    # 检查点存在则退出
    if os.path.exists(args.checkpoint_path) and get_best_checkpoint(args.checkpoint_path) is not None:
        logging.warning(f"Checkpoint already exists at {args.checkpoint_path}. Exiting pretraining.")
        return

    def compute_metrics(eval_predictions):
        preds, golds = eval_predictions
        preds = np.argmax(preds, axis=1)
        # metrics = classification_report(preds, golds, output_dict=True)
        metrics = classification_report(golds, preds, output_dict=True)
        metrics['macro avg'].update({'accuracy': metrics['accuracy']})
        return metrics['macro avg']

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"] if 'bert' in args.model_path else ["q_proj", "v_proj"]
    )
    tokenizer = data['tokenizer']
    base_model = create_model(model_path=args.model_path, num_labels=args.num_labels, tokenizer=tokenizer)
    model = get_peft_model(base_model, peft_config)
    
    # --- 改动 3: 使用从 'data' 字典中获取的对象 ---
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    early_stop_patience = getattr(args, "early_stop_patience", 3)
    early_stop_delta = getattr(args, "early_stop_delta", 0.0)


    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=args.checkpoint_path,
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        save_strategy='epoch',
        save_total_limit=1,
        # —— 为早停配套的设置 ——
        load_best_model_at_end=True,       # 训练结束回滚到最佳模型
        metric_for_best_model="f1-score",  # 与 compute_metrics 的键一致
        greater_is_better=True,            # f1 越大越好
        report_to=[],                      # 彻底关闭外部上报（已禁用 wandb）
    )

    # 定义Trainer
    trainer = RegTrainer(
        reg_loss=args.reg_loss,
        num_labels=args.num_labels,
        device=args.device,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dataset_in_eval,
        compute_metrics=compute_metrics,
        data_collator=collate_batch,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stop_patience,     # 连续若干 eval 指标无提升则停止
                early_stopping_threshold=early_stop_delta        # 最小提升幅度（小数，比如 1e-4）
            )
        ],
    )
    # --- 改动结束 ---

    logging.info("Evaluating before training...")
    results = trainer.evaluate()
    logging.info(f"Initial results: {results}")

    logging.info("Starting training...")
    trainer.train()

    logging.info("Evaluating after training...")
    results = trainer.evaluate()
    logging.info(f"Final results: {results}")

def apply_config_updates(args, config_dict, parser):
    """
    使用配置字典中的值更新 args 对象，同时进行类型转换。
    命令行中显式给出的参数不会被覆盖。
    """
    # 创建一个从 dest 到 action.type 的映射
    type_map = {action.dest: action.type for action in parser._actions}

    for key, value in config_dict.items():
        # 检查参数是否在命令行中被用户显式提供
        if f'--{key}' not in sys.argv and hasattr(args, key):
            # 获取该参数预期的类型
            expected_type = type_map.get(key)
            # 如果有预期类型且值不为None，则进行类型转换
            if expected_type and value is not None:
                value = expected_type(value)
            setattr(args, key, value)

if __name__ == '__main__':
    # 1. 从 configs.py 获取基础的解析器
    parser = create_parser()
    
    # 2. 【核心】只在这里添加 --config 参数
    parser.add_argument("--config", type=str, help="Path to the YAML config file")
    
    # 3. 解析参数 (命令行参数优先)
    args = parser.parse_args()
    
    # 4. 加载 YAML 文件并智能覆盖
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        apply_config_updates(args, yaml_config, parser)
        
    # 5. 调用 finalize_config 完成路径和日志的设置
    config_args = finalize_config(args)
    
    # 6. 使用最终的 config_args 运行主程序
    # 在 pretrain.py 中:
    run_pretraining(config_args) 
    # 在 train_ood.py 中:
    # run_ood_evaluation(config_args)