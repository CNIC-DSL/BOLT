import os
from typing import List

import torch
import numpy as np
import pandas as pd
import transformers
from datasets import Dataset, DatasetDict

from my_args import DataTrainingArguments


def data_collator(features):

    first = features[0]
    batch = {}
    if "original_text" in first:
        batch["original_text"] = [f["original_text"] for f in features]

    if "label" in first and first["label"] is not None:
        label = (
            first["label"].item()
            if isinstance(first["label"], torch.Tensor)
            else first["label"]
        )
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor(
                [f["label_ids"] for f in features], dtype=dtype
            )

    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


def convert_to_nums(
    data_args: DataTrainingArguments,
    datasets: DatasetDict,
    label_list: List[int],
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> DatasetDict:

    sentence1_key, sentence2_key = "text", None

    if data_args.pad_to_max_length:
        padding = "max_length"
        max_length = data_args.max_seq_length
    else:

        padding = False
        max_length = None

    label_to_id = None

    label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):

        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_length, truncation=True
        )

        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[label] for label in examples["label"]]
        result["sent_id"] = [index for index, i in enumerate(examples["label"])]
        result["original_text"] = examples[sentence1_key]
        return result

    datasets = datasets.map(
        preprocess_function,
        batched=True,
        batch_size=None,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    return datasets


def load_datasets(data_args: DataTrainingArguments) -> DatasetDict:

    origin_train_path = os.path.join(
        data_args.data_dir, data_args.dataset, "origin_data", "train.tsv"
    )
    origin_valid_path = os.path.join(
        data_args.data_dir, data_args.dataset, "origin_data", "dev.tsv"
    )
    origin_test_path = os.path.join(
        data_args.data_dir, data_args.dataset, "origin_data", "test.tsv"
    )

    labeled_train_path = os.path.join(
        data_args.data_dir,
        data_args.dataset,
        "labeled_data",
        str(data_args.labeled_ratio),
        "train.tsv",
    )
    labeled_valid_path = os.path.join(
        data_args.data_dir,
        data_args.dataset,
        "labeled_data",
        str(data_args.labeled_ratio),
        "dev.tsv",
    )

    origin_train_df = pd.read_csv(origin_train_path, sep="\t", dtype=str)
    origin_valid_df = pd.read_csv(origin_valid_path, sep="\t", dtype=str)
    df_test = pd.read_csv(origin_test_path, sep="\t", dtype=str)

    labeled_train_df = pd.read_csv(labeled_train_path, sep="\t", dtype=str)
    labeled_valid_df = pd.read_csv(labeled_valid_path, sep="\t", dtype=str)

    df_train = labeled_train_df
    df_train["text"] = origin_train_df["text"]

    df_valid = labeled_valid_df
    df_valid["text"] = origin_valid_df["text"]

    known_label_path = os.path.join(
        data_args.data_dir,
        data_args.dataset,
        "label",
        f"{data_args.fold_type}{data_args.fold_num}",
        f"part{data_args.fold_idx}",
        f"label_known_{data_args.known_cls_ratio}.list",
    )
    seen_labels = pd.read_csv(known_label_path, header=None)[0].tolist()

    df_train_seen: pd.DataFrame = df_train[
        (df_train.label.isin(seen_labels)) & (df_train["labeled"].astype(bool))
    ]
    df_valid_seen: pd.DataFrame = df_valid[
        (df_valid.label.isin(seen_labels)) & (df_valid["labeled"].astype(bool))
    ]
    df_valid_oos: pd.DataFrame = df_valid[~df_valid.label.isin(seen_labels)]

    df_valid_oos.loc[:, "label"] = "oos"
    df_test.loc[~df_test.label.isin(seen_labels), "label"] = "oos"

    df_valid_all = pd.concat([df_valid_seen, df_valid_oos])

    data = {
        "train": Dataset.from_pandas(df_train_seen, preserve_index=False),
        "valid_seen": Dataset.from_pandas(df_valid_seen, preserve_index=False),
        "valid_oos": Dataset.from_pandas(df_valid_oos, preserve_index=False),
        "valid_all": Dataset.from_pandas(df_valid_all, preserve_index=False),
        "test": Dataset.from_pandas(df_test, preserve_index=False),
    }

    datasets = DatasetDict(data)
    return datasets
