import torch
import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer


def load_and_prepare_datasets(args):
    known_label_list = pd.read_csv(
        f"{args.data_dir}/{args.dataset}/label/{args.fold_type}{args.fold_num}/part{args.fold_idx}/label_known_{args.known_cls_ratio}.list",
        header=None,
    )[0].tolist()

    args.num_labels = len(known_label_list)

    origin_train_data = pd.read_csv(
        f"{args.data_dir}/{args.dataset}/origin_data/train.tsv", sep="\t"
    )
    origin_eval_data = pd.read_csv(
        f"{args.data_dir}/{args.dataset}/origin_data/dev.tsv", sep="\t"
    )
    origin_test_data = pd.read_csv(
        f"{args.data_dir}/{args.dataset}/origin_data/test.tsv", sep="\t"
    )

    train_data = pd.read_csv(
        f"{args.data_dir}/{args.dataset}/labeled_data/{args.labeled_ratio}/train.tsv",
        sep="\t",
    )
    train_data["text"] = origin_train_data["text"]
    train_data = train_data[
        (train_data["label"].isin(known_label_list)) & (train_data["labeled"])
    ].drop("labeled", axis=1)

    eval_data = pd.read_csv(
        f"{args.data_dir}/{args.dataset}/labeled_data/{args.labeled_ratio}/dev.tsv",
        sep="\t",
    )
    eval_data["text"] = origin_eval_data["text"]
    data_in_eval = eval_data[
        (eval_data["label"].isin(known_label_list)) & (eval_data["labeled"])
    ].drop("labeled", axis=1)

    test_data = pd.read_csv(
        f"{args.data_dir}/{args.dataset}/origin_data/test.tsv", sep="\t"
    )
    data_in_test = test_data[test_data["label"].isin(known_label_list)]
    data_out_test = test_data[~test_data["label"].isin(known_label_list)]
    data_out_test["label"] = "ood"

    train_data["labels"] = train_data["label"].apply(
        lambda x: known_label_list.index(x) if x in known_label_list else -1
    )
    data_in_eval["labels"] = data_in_eval["label"].apply(
        lambda x: known_label_list.index(x) if x in known_label_list else -1
    )
    data_in_test["labels"] = data_in_test["label"].apply(
        lambda x: known_label_list.index(x) if x in known_label_list else -1
    )
    data_out_test["labels"] = data_out_test["label"].apply(
        lambda x: known_label_list.index(x) if x in known_label_list else -1
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.add_special_tokens({"eos_token": "[END]"})

    train_dataset = Dataset.from_pandas(train_data.reset_index(drop=True))
    dataset_in_eval = Dataset.from_pandas(data_in_eval.reset_index(drop=True))
    dataset_in_test = Dataset.from_pandas(data_in_test.reset_index(drop=True))
    dataset_out_test = Dataset.from_pandas(data_out_test.reset_index(drop=True))

    def tokenize_function(example):
        return tokenizer(
            example["text"], padding="max_length", truncation=True, max_length=60
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    dataset_in_test = dataset_in_test.map(tokenize_function, batched=True)
    dataset_in_eval = dataset_in_eval.map(tokenize_function, batched=True)
    dataset_out_test = dataset_out_test.map(tokenize_function, batched=True)

    train_dataset = train_dataset.remove_columns(["text", "label"])
    dataset_in_test = dataset_in_test.remove_columns(["text", "label"])
    dataset_in_eval = dataset_in_eval.remove_columns(["text", "label"])
    dataset_out_test = dataset_out_test.remove_columns(["text", "label"])

    test_dataset = concatenate_datasets([dataset_in_test, dataset_out_test])

    train_dataset.set_format("torch")
    dataset_in_eval.set_format("torch")
    test_dataset.set_format("torch")

    def collate_batch(batch, max_len=512):
        ans = {}
        max_len = max([i["input_ids"].shape[0] for i in batch])
        for key in batch[0]:
            if key in ["input_ids", "attention_mask", "token_type_ids"]:
                padded = [
                    i[key].tolist()
                    + (max_len - len(i[key]))
                    * [0 if key != "input_ids" else tokenizer.pad_token_id]
                    for i in batch
                ]
                ans[key] = torch.tensor(padded, dtype=torch.long)
            else:
                ans[key] = torch.stack([i[key] for i in batch])
        return ans

    loader_in_train = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    loader_in_eval = DataLoader(
        dataset_in_eval,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.eval_batch_size,
        collate_fn=collate_batch,
    )

    return {
        "train_dataset": train_dataset,
        "dataset_in_eval": dataset_in_eval,
        "dataset_in_test": dataset_in_test,
        "dataset_out_test": dataset_out_test,
        "test_dataset": test_dataset,
        "loader_in_train": loader_in_train,
        "loader_in_eval": loader_in_eval,
        "test_loader": test_loader,
        "tokenizer": tokenizer,
        "collate_batch": collate_batch,
    }
