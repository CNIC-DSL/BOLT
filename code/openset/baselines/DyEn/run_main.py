import os
import logging
import shutil
import sys

import torch
import fitlog

import transformers
import transformers.utils.logging
from transformers import (
    AutoConfig,
    AutoTokenizer,
    set_seed, BertConfig,
)
from transformers.trainer_utils import is_main_process

import eval as my_eval
from handle_data import load_datasets, convert_to_nums, data_collator
from my_args import DataTrainingArguments, FitLogArguments, OtherArguments
from my_trainer import SimpleTrainer
from transformers.training_args import TrainingArguments
from models import (
    BertForSequenceClassificationWithPabee
)
from my_hf_argparser import HfArgumentParser
import train_step_freelb, train_step_plain
import yaml
import json

logger = logging.getLogger(__name__)
torch.set_num_threads(6)

# --- 新增：定义一个辅助函数来处理配置更新 ---
def apply_yaml_to_dataclasses(yaml_config, dataclass_tuple, cli_args):
    """
    将 YAML 配置应用到 dataclass 对象上，但跳过命令行中已明确指定的参数。
    """
    arg_list = [arg.lstrip('-') for arg in cli_args if arg.startswith('--')]
    
    for key, value in yaml_config.items():
        if key in arg_list:
            # 如果参数在命令行中指定，则跳过，保留命令行优先级
            continue
        
        # 遍历所有 dataclass 对象，找到持有该属性的对象并更新
        for dc_obj in dataclass_tuple:
            if hasattr(dc_obj, key):
                setattr(dc_obj, key, value)
                break # 找到后即跳出内层循环


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script_v0.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # https://zhuanlan.zhihu.com/p/296535876
    # HfArgumentParser可以将类对象中的实例属性转换成转换为解析参数。
    parser = HfArgumentParser((OtherArguments, DataTrainingArguments, TrainingArguments, FitLogArguments))
    other_args: OtherArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments
    fitlog_args: FitLogArguments

    # assert len(sys.argv) == 2 and sys.argv[1].endswith(".json")
    # other_args, data_args, training_args, fitlog_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    other_args, data_args, training_args, fitlog_args = parser.parse_args_into_dataclasses()

    # # TODO: DEBUG
    # training_args.do_train = False
    # other_args.scale = 1.5
    # other_args.scale_ood = 1.3

    all_args_tuple = (other_args, data_args, training_args, fitlog_args)
    if other_args.config:
        with open(other_args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # 3. 应用通用配置 (最低优先级)
        # 我们传入 sys.argv 来判断哪些参数是命令行指定的
        apply_yaml_to_dataclasses(yaml_config, all_args_tuple, sys.argv)
        
        # 4. (推荐) 应用数据集专属配置 (第二优先级)
        if 'dataset_specific_configs' in yaml_config:
            dataset_name = data_args.dataset  # 获取当前数据集名称
            if dataset_name in yaml_config['dataset_specific_configs']:
                dataset_specific_config = yaml_config['dataset_specific_configs'][dataset_name]
                apply_yaml_to_dataclasses(dataset_specific_config, all_args_tuple, sys.argv)

    # 校验防呆
    assert other_args.loss_type in ["original", "increase", "ce_and_div_drop-last-layer", "ce_and_div"]

    # trainer 执行 DataLoader 转化，有一步 _remove_unused_columns
    # 什么 sb 默认啊
    training_args.remove_unused_columns = False

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 直接使用我们YAML中定义的标准输出路径
    model_output_root = training_args.output_dir
    os.makedirs(model_output_root, exist_ok=True)

    # if training_args.do_train:
    #     json_file_name = '_' + '_'.join([data_args.dataset, str(data_args.known_cls_ratio), str(training_args.seed)]) + '.json'
    #     shutil.copy2(os.path.abspath(sys.argv[1]), os.path.join(model_output_root, json_file_name))

    # ################################################################################
    # # 设置 fitlog
    # fitlog.set_log_dir(other_args.fitlog_dir)
    # fitlog_args_dict = {
    #     "seed": training_args.seed,
    #     "warmup_steps": training_args.warmup_steps,
    #     "task_name": f'{data_args.data}-{data_args.known_ratio}-{training_args.seed}'

    # }
    # fitlog_args_name = [i for i in dir(fitlog_args) if i[0] != "_"]
    # for args_name in fitlog_args_name:
    #     args_value = getattr(fitlog_args, args_name)
    #     if args_value is not None:
    #         fitlog_args_dict[args_name] = args_value
    # fitlog.add_hyper(fitlog_args_dict)

    # ################################################################################

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    ################################################################################################
    datasets = load_datasets(data_args)

    # 有多少 label

    # 获取 num_all_labels（其中加上了 oos）
    # 这里是本项目自定义的
    # Trying to have good defaults here, don't hesitate to tweak to your needs.
    is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
    assert not is_regression
    # A useful fast method:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    label_list = datasets["train"].unique("label")
    label_list += ['oos']
    num_all_labels = len(label_list)

    tokenizer: transformers.PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        other_args.bert_model,
        cache_dir=other_args.cache_dir,
        # local_files_only=True,
        use_fast=True,
    )

    datasets = convert_to_nums(data_args, datasets, label_list, tokenizer)

    ################################################################################################
    # 得到 model（这里暂不支持加载已经训练好的）

    pertained_config: BertConfig = AutoConfig.from_pretrained(
        other_args.bert_model,
        finetuning_task=None,
        # local_files_only=True,
        cache_dir=other_args.cache_dir,
    )

    # model = pjmodel.BertForSequenceClassificationWithPabee.from_pretrained(other_args.bert_model, num_ind_labels=num_all_labels-1)
    model = BertForSequenceClassificationWithPabee(pertained_config=pertained_config, other_args=other_args, num_ind_labels=num_all_labels-1)

    #####################################################################################
    # 训练

    model_file_name = '_' + '_'.join([data_args.dataset, str(data_args.known_cls_ratio), str(training_args.seed)]) + '.pt'

    if other_args.adv_k > 0:
        train_step = train_step_freelb.FreeLB(
            adv_k=other_args.adv_k,
            adv_lr=other_args.adv_lr,
            adv_init_mag=other_args.adv_init_mag,
            adv_max_norm=other_args.adv_max_norm
        )
    else:
        train_step = train_step_plain.TrainStep()
    trainer = SimpleTrainer(
        num_train_epochs=training_args.num_train_epochs,
        clip=other_args.clip,
        model_path_=os.path.join(model_output_root, model_file_name),  # 保存模型
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid_seen"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_step=train_step
    )
    if training_args.do_train:
        file_postfix = '_'.join([data_args.dataset, str(data_args.known_cls_ratio), str(training_args.seed)]) + '.csv'

        trainer.train_ce_loss()
   
    else:
        model.load_state_dict(torch.load(os.path.join(model_output_root, model_file_name)))
        model.to(training_args.device)

    ################################################################################################
    # Evaluation (已根据SOP进行标准化改造 - 最终版)
    if training_args.do_predict:
        from sklearn.metrics import classification_report
        import pandas as pd

        # --- 步骤1：重新定义 kwargs 字典，为所有评估器准备通用参数 ---
        file_postfix = '_'.join([data_args.dataset, str(data_args.known_cls_ratio), str(training_args.seed)]) + '.csv'
        
        valid_all_dataloader = trainer.get_eval_dataloader(datasets['valid_all'])
        valid_dataloader = trainer.get_eval_dataloader(datasets['valid_seen'])
        train_dataloader = trainer.get_train_dataloader()
        test_dataloader = trainer.get_test_dataloader(datasets["test"])

        model_forward_cache = {}

        kwargs = dict(
            model=model,
            root=model_output_root,
            file_postfix=file_postfix,
            dataset_name=data_args.dataset, # 使用标准化的参数名
            device=training_args.device,
            num_labels=num_all_labels,
            tuning='valid',
            scale_ind=other_args.scale,
            scale_ood=other_args.scale_ood,
            valid_all_dataloader=valid_all_dataloader,
            valid_dataloader=valid_dataloader,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            model_forward_cache=model_forward_cache
        )

        # --- 步骤2：循环运行每一种评估方法 ---
        evaluator_classes = [
            (my_eval.KnnEvaluator, "knn"),
            (my_eval.MspEvaluator, "msp"),
            (my_eval.MaxLogitEvaluator, "maxLogit"),
            (my_eval.EnergyEvaluator, "energy"),
            (my_eval.EntropyEvaluator, "entropy"),
            (my_eval.OdinEvaluator, "odin"),
            (my_eval.MahaEvaluator, "maha"),
            (my_eval.LofCosineEvaluator, "lof_cosine"),
            (my_eval.LofEuclideanEvaluator, "lof_euclidean")
        ]
        
        id_to_label = {i: v for i, v in enumerate(label_list)}

        for eval_class, method_name in evaluator_classes:
            print(f"\n---> Running Evaluation for method: {method_name}")
            
            # 修正：不再修改共享的kwargs，而是只在需要时传递额外参数
            if method_name in ["energy", "odin"]:
                temp = 1 if method_name == "energy" else 100
                evaluator = eval_class(temperature=temp, **kwargs)
            else:
                evaluator = eval_class(**kwargs)
            
            y_pred_ids, y_true_ids = evaluator.eval()
            
            y_true_labels = [id_to_label.get(label_id, 'oos') for label_id in y_true_ids]
            y_pred_labels = [id_to_label.get(label_id, 'oos') for label_id in y_pred_ids]

            report = classification_report(y_true_labels, y_pred_labels, output_dict=True, zero_division=0)
            
            final_results = {}
            final_results['dataset'] = data_args.dataset
            final_results['seed'] = training_args.seed
            final_results['known_cls_ratio'] = data_args.known_cls_ratio
            final_results['ood_method'] = method_name
            final_results['ACC'] = report['accuracy']
            final_results['F1'] = report['macro avg']['f1-score']
            
            seen_class_labels = [l for l in label_list if l != 'oos']
            known_f1_scores = [report[label]['f1-score'] for label in seen_class_labels if label in report]
            final_results['K-F1'] = sum(known_f1_scores) / len(known_f1_scores) if known_f1_scores else 0.0
            final_results['N-F1'] = report['oos']['f1-score'] if 'oos' in report else 0.0
            # final_results['args'] = json.dumps(vars(training_args), ensure_ascii=False)
            def safe_args_to_json(*args_objs):
                safe_dict = {}
                for args_obj in args_objs:
                    if args_obj is None:
                        continue
                    for k, v in vars(args_obj).items():
                        try:
                            json.dumps(v)  # 测试能不能序列化
                            safe_dict[k] = v
                        except (TypeError, OverflowError):
                            safe_dict[k] = str(v)  # 无法直接序列化就转字符串
                return json.dumps(safe_dict, ensure_ascii=False)

            # 用法：把两个 args 一起传进去
            final_results['args'] = safe_args_to_json(other_args, training_args, data_args)

            # metric_dir = os.path.join(training_args.output_dir, 'metrics')
            os.makedirs(other_args.save_results_path, exist_ok=True)
            results_path = os.path.join(other_args.save_results_path, 'results.csv')

            df_to_save = pd.DataFrame([final_results])

            # merged_args = merge_args_dicts(other_args, training_args, cfg)

            df_to_save["method"] = "DyEn"

            # 需要的列
            cols = ["method","dataset","known_cls_ratio","labeled_ratio","cluster_num_factor",
                    "seed","ACC","F1","K-F1","N-F1","args"]

            # 从 other_args / training_args 取值的最小函数（优先 other_args）
            val = lambda c: next((getattr(src, c) for src in (other_args, training_args, data_args) if hasattr(src, c)), None)

            # 若列已存在则保留原列；否则用 val(c) 创建并广播
            for c in cols:
                df_to_save[c] = df_to_save.get(c, val(c))

            # 调整列顺序
            df_to_save = df_to_save[cols]

            if not os.path.exists(results_path):
                df_to_save.to_csv(results_path, index=False)
            else:
                existing_df = pd.read_csv(results_path)
                # new_row_df = pd.DataFrame([final_results])
                updated_df = pd.concat([existing_df, df_to_save], ignore_index=True)
                updated_df.to_csv(results_path, index=False)
            
            print(f"Saved results for {method_name} to {results_path}")

    return None


if __name__ == "__main__":
    main()
