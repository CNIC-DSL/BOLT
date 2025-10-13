import csv
import sys
import logging
import os
import random
import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


class BaseDataNew(object):

    def __init__(self, args):

        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.logger_name = args.logger_name
        args.max_seq_length = args.max_seq_length

        all_label_path = os.path.join(self.data_dir, "label", "label.list")
        self.all_label_list = pd.read_csv(all_label_path, header=None)[0].tolist()

        self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor)

        self.known_label_list = pd.read_csv(
            f"{args.data_dir}/{args.dataset}/label/{args.fold_type}{args.fold_num}/part{args.fold_idx}/label_known_{args.known_cls_ratio}.list",
            header=None,
        )[0].tolist()
        self.n_known_cls = len(self.known_label_list)

        self.known_lab = [self.all_label_list.index(a) for a in self.known_label_list]

        self.train_labeled_examples, self.train_unlabeled_examples = self.get_examples(
            args, mode="train", separate=True
        )
        self.eval_examples = self.get_examples(args, mode="eval", separate=False)
        self.test_examples = self.get_examples(args, mode="test", separate=False)
        self.test_known_examples, self.test_unknown_examples = self.get_examples(
            args, mode="test", separate=True
        )

    def get_examples(self, args, mode, separate=False):

        if mode == "train":

            origin_data_path = os.path.join(self.data_dir, "origin_data", "train.tsv")
            labeled_info_path = os.path.join(
                self.data_dir, "labeled_data", str(args.labeled_ratio), "train.tsv"
            )

            origin_data = pd.read_csv(origin_data_path, sep="\t")
            labeled_info = pd.read_csv(labeled_info_path, sep="\t")

            merged_data = labeled_info
            merged_data["text"] = origin_data["text"]

            is_labeled_known = (merged_data["label"].isin(self.known_label_list)) & (
                merged_data["labeled"]
            )

            labeled_df = merged_data[is_labeled_known]
            unlabeled_df = merged_data[~is_labeled_known]

            train_labeled_examples = []
            for i, row in labeled_df.iterrows():
                guid = f"train_labeled-{i}"
                train_labeled_examples.append(
                    InputExample(guid=guid, text_a=row["text"], label=row["label"])
                )

            train_unlabeled_examples = []
            for i, row in unlabeled_df.iterrows():
                guid = f"train_unlabeled-{i}"
                train_unlabeled_examples.append(
                    InputExample(guid=guid, text_a=row["text"], label=row["label"])
                )

            return train_labeled_examples, train_unlabeled_examples

        else:
            examples = self.read_data(self.data_dir, mode)
            if mode == "eval":

                eval_examples = [
                    ex for ex in examples if ex.label in self.known_label_list
                ]
                return eval_examples

            elif mode == "test":
                if not separate:
                    return examples
                else:

                    test_known_examples = [
                        ex for ex in examples if ex.label in self.known_label_list
                    ]
                    test_unknown_examples = [
                        ex for ex in examples if ex.label not in self.known_label_list
                    ]
                    return test_known_examples, test_unknown_examples

    def read_data(self, data_dir, mode):
        if mode == "train":

            file_path = os.path.join(data_dir, "origin_data", "train.tsv")
        elif mode == "eval":

            file_path = os.path.join(data_dir, "origin_data", "dev.tsv")
        elif mode == "test":

            file_path = os.path.join(data_dir, "origin_data", "test.tsv")
        else:
            raise NotImplementedError(f"Mode {mode} not found")

        lines = self.read_tsv(file_path)
        examples = self.create_examples(lines, mode)
        return examples

    def read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def create_examples(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 2:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
            )
        return examples

    def get_labels(self, data_dir):
        docs = os.listdir(data_dir)
        if "train.tsv" in docs:
            import pandas as pd

            test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
            labels = [str(label).lower() for label in test["label"]]
            labels = np.unique(np.array(labels))
        elif "dataset.json" in docs:
            with open(os.path.join(data_dir, "dataset.json"), "r") as f:
                dataset = json.load(f)
                dataset = dataset[list(dataset.keys())[0]]
            labels = []
            for dom in dataset:
                for ind, data in enumerate(dataset[dom]):
                    label = data[1][0]
                    labels.append(str(label).lower())
            labels = np.unique(np.array(labels))
        return labels

    def convert_examples_to_features(
        self, examples, label_list, max_seq_length, tokenizer
    ):
        label_map = {}
        for i, label in enumerate(label_list):
            label_map[label] = i

        features = []
        for ex_index, example in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)

                self.truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:

                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[: (max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1] * len(input_ids)

            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            label_id = label_map[example.label]

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                )
            )
        return features

    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop(0)
            else:
                tokens_b.pop()

    def difference(self, a, b):
        _b = set(b)
        return [item for item in a if item not in _b]


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


if __name__ == "__main__":

    config_path = "../methods/intent_generation/config.yaml"

    from utils import load_yaml_config
    from easydict import EasyDict

    configs = load_yaml_config(config_path)

    args = EasyDict(configs)

    base_data = BaseData(args)

    dataloader = base_data.eval_dataloader

    for idx, batch in enumerate(dataloader):
        print(batch)
        exit()
