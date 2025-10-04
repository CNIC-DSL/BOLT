import os
from typing import List

import torch
import numpy as np
import pandas as pd
import transformers
from datasets import Dataset, DatasetDict

from my_args import DataTrainingArguments


def data_collator(features):
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    first = features[0]
    batch = {}
    if "original_text" in first:
        batch["original_text"] = [f["original_text"] for f in features]
    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


def convert_to_nums(data_args: DataTrainingArguments, datasets: DatasetDict, label_list: List[int], tokenizer: transformers.PreTrainedTokenizerBase) -> DatasetDict:
    # 查看 input 有什么特征（这里只有一个 text，作为输入特征；bert 是输入两句话的，另一句就是 None 了）
    # Preprocessing the datasets
    # 这里是本项目自定义的

    # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    # non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
    # if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
    #     sentence1_key, sentence2_key = "sentence1", "sentence2"
    # else:
    #     if len(non_label_column_names) >= 2:
    #         sentence1_key, sentence2_key = non_label_column_names[:2]
    #     else:
    #         sentence1_key, sentence2_key = non_label_column_names[0], None

    # --- 修正：绕过脆弱的自动列名检测，直接指定文本列 ---
    # 我们的数据集是单句分类，文本列永远是 'text'
    sentence1_key, sentence2_key = "text", None
    # --- 修正结束 ---

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
        max_length = data_args.max_seq_length
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
        max_length = None

    # 校验并获得 label_to_id
    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None

    label_to_id = {v: i for i, v in enumerate(label_list)}

    #####################################################################################
    # 数据编码（把 label text 转为 int 这些，方便后续转为 Tensor）
    def preprocess_function(examples):
        """
        字段有 input_ids；token_type_ids；attention_mask;label; sent_id; original_text

        input_ids；token_type_ids 就是 bert 里面是第几句话；attention_mask 标记每句话真实长度

        labels: List[int]
        sent_id: List[int]
        original_text: List[str]
        """
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[label] for label in examples["label"]]
        result["sent_id"] = [index for index, i in enumerate(examples["label"])]
        result["original_text"] = examples[sentence1_key]
        return result

    # # map: Apply a function to all the elements in the table
    datasets = datasets.map(preprocess_function, batched=True, batch_size=None, load_from_cache_file=not data_args.overwrite_cache)
    return datasets


def load_datasets(data_args: DataTrainingArguments) -> DatasetDict:
    """
    (已根据SOP进行标准化改造)
    Loads train/validation/test data from standardized directories and determines
    known classes based on fold parameters.
    """
    # 1. 根据SOP标准从YAML参数构建文件路径
    origin_train_path = os.path.join(data_args.data_dir, data_args.dataset, 'origin_data', 'train.tsv')
    origin_valid_path = os.path.join(data_args.data_dir, data_args.dataset, 'origin_data', 'dev.tsv')
    origin_test_path = os.path.join(data_args.data_dir, data_args.dataset, 'origin_data', 'test.tsv')

    labeled_train_path = os.path.join(data_args.data_dir, data_args.dataset, 'labeled_data', str(data_args.labeled_ratio), 'train.tsv')
    labeled_valid_path = os.path.join(data_args.data_dir, data_args.dataset, 'labeled_data', str(data_args.labeled_ratio), 'dev.tsv')

    # 2. 使用pandas加载标准化的TSV文件
    origin_train_df = pd.read_csv(origin_train_path, sep='\t', dtype=str)
    origin_valid_df = pd.read_csv(origin_valid_path, sep='\t', dtype=str)
    df_test = pd.read_csv(origin_test_path, sep='\t', dtype=str)

    labeled_train_df = pd.read_csv(labeled_train_path, sep='\t', dtype=str)
    labeled_valid_df = pd.read_csv(labeled_valid_path, sep='\t', dtype=str)
    
    # 3. 组合文本和标签信息，生成后续步骤所需的DataFrame
    df_train = labeled_train_df
    df_train['text'] = origin_train_df['text']
    
    df_valid = labeled_valid_df
    df_valid['text'] = origin_valid_df['text']

    # 4. (核心修改) 从标准化的.list文件加载已知类，替换掉旧的随机选择和idx.txt逻辑
    known_label_path = os.path.join(
        data_args.data_dir,
        data_args.dataset,
        'label',
        f'{data_args.fold_type}{data_args.fold_num}',
        f'part{data_args.fold_idx}',
        f'label_known_{data_args.known_cls_ratio}.list'
    )
    seen_labels = pd.read_csv(known_label_path, header=None)[0].tolist()

    # 5. (保留原逻辑) 使用新的 seen_labels 列表来筛选和划分数据
    # 训练集和验证集：必须是已知类别 & 已标注
    df_train_seen: pd.DataFrame = df_train[(df_train.label.isin(seen_labels)) & (df_train['labeled'].astype(bool))]
    df_valid_seen: pd.DataFrame = df_valid[(df_valid.label.isin(seen_labels)) & (df_valid['labeled'].astype(bool))]
    df_valid_oos: pd.DataFrame = df_valid[~df_valid.label.isin(seen_labels)]

    df_valid_oos.loc[:, "label"] = 'oos'
    df_test.loc[~df_test.label.isin(seen_labels), "label"] = 'oos'

    df_valid_all = pd.concat([df_valid_seen, df_valid_oos])

    # 6. (保留原逻辑) 将处理好的DataFrame转换为Hugging Face的Dataset对象
    data = {
        "train": Dataset.from_pandas(df_train_seen, preserve_index=False),
        "valid_seen": Dataset.from_pandas(df_valid_seen, preserve_index=False),
        "valid_oos": Dataset.from_pandas(df_valid_oos, preserve_index=False),
        "valid_all": Dataset.from_pandas(df_valid_all, preserve_index=False),
        "test": Dataset.from_pandas(df_test, preserve_index=False),
    }

    datasets = DatasetDict(data)
    return datasets
