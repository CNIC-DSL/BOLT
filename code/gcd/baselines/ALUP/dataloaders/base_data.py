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
       
        all_label_path = os.path.join(self.data_dir, 'label', 'label.list')
        self.all_label_list = pd.read_csv(all_label_path, header=None)[0].tolist()

        # calculate the number of known classes
        self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor)
        # self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)

        # conduct the split of known and unknown classes
        # self.known_label_list = pd.read_csv(f"{self.data_dir}/label/label_{args.known_cls_ratio}.list", header=None)[0].tolist()
        self.known_label_list = pd.read_csv(f'{args.data_dir}/{args.dataset}/label/{args.fold_type}{args.fold_num}/part{args.fold_idx}/label_known_{args.known_cls_ratio}.list', header=None)[0].tolist()
        self.n_known_cls = len(self.known_label_list)


        # self.known_train_sample = pd.read_csv(f"{self.data_dir}/labeled_data/train_{args.labeled_ratio}.tsv", sep='\t')
        # self.known_train_sample = self.known_train_sample[self.known_train_sample['label'].isin(self.known_label_list)]

        # self.known_eval_sample = pd.read_csv(f"{self.data_dir}/labeled_data/dev_{args.labeled_ratio}.tsv", sep='\t')
        # self.known_eval_sample = self.known_eval_sample[self.known_eval_sample['label'].isin(self.known_label_list)]

        # self.known_lab = [int(np.where(self.all_label_list== a)[0]) for a in self.known_label_list]
        self.known_lab = [self.all_label_list.index(a) for a in self.known_label_list]

        # create examples
        self.train_labeled_examples, self.train_unlabeled_examples = self.get_examples(args, mode='train', separate=True)
        self.eval_examples = self.get_examples(args, mode='eval', separate=False)
        self.test_examples = self.get_examples(args, mode='test', separate=False)
        self.test_known_examples, self.test_unknown_examples = self.get_examples(args, mode='test', separate=True)


    def get_examples(self, args, mode, separate=False):
        """
        采用Glean的双重过滤逻辑来划分训练集。
        eval和test模式保持ALUP原有的简单过滤逻辑。
        """
        # --- 对于训练集，执行Glean的双重过滤 ---
        if mode == 'train':
            # 1. 定义并读取两个核心文件
            origin_data_path = os.path.join(self.data_dir, 'origin_data', 'train.tsv')
            labeled_info_path = os.path.join(self.data_dir, 'labeled_data', str(args.labeled_ratio), 'train.tsv')

            # 使用pandas读取
            origin_data = pd.read_csv(origin_data_path, sep='\t')
            labeled_info = pd.read_csv(labeled_info_path, sep='\t')
            
            # 2. 合并信息：将原始文本添加到标签信息中
            # 假设两个文件的行数和顺序完全对应
            merged_data = labeled_info
            merged_data['text'] = origin_data['text']
            
            # 3. 创建双重过滤的布尔掩码 (boolean mask)
            # 条件：标签必须在已知类别列表里 & labeled 字段必须为 True
            is_labeled_known = (merged_data['label'].isin(self.known_label_list)) & (merged_data['labeled'])
            
            # 4. 根据掩码分割为“有标签”和“无标签”两个 DataFrame
            labeled_df = merged_data[is_labeled_known]
            unlabeled_df = merged_data[~is_labeled_known]

            # 5. 将两个 DataFrame 分别转换为 InputExample 对象列表 (保持ALUP的输出格式)
            train_labeled_examples = []
            for i, row in labeled_df.iterrows():
                guid = f"train_labeled-{i}"
                train_labeled_examples.append(InputExample(guid=guid, text_a=row['text'], label=row['label']))
            
            train_unlabeled_examples = []
            for i, row in unlabeled_df.iterrows():
                guid = f"train_unlabeled-{i}"
                train_unlabeled_examples.append(InputExample(guid=guid, text_a=row['text'], label=row['label']))
                
            return train_labeled_examples, train_unlabeled_examples

        # --- 对于验证集和测试集，使用ALUP原有的、正确的加载逻辑 ---
        else: # mode is 'eval' or 'test'
            examples = self.read_data(self.data_dir, mode)
            if mode == 'eval':
                # 验证集：只包含已知类别的样本
                eval_examples = [ex for ex in examples if ex.label in self.known_label_list]
                return eval_examples

            elif mode == 'test':
                if not separate:
                    return examples
                else:
                    # 分别返回已知和未知意图的测试样本
                    test_known_examples = [ex for ex in examples if ex.label in self.known_label_list]
                    test_unknown_examples = [ex for ex in examples if ex.label not in self.known_label_list]
                    return test_known_examples, test_unknown_examples


    def read_data(self, data_dir, mode):
        """
        read data from data_dir, with paths pointing to the 'origin_data' subdirectory.
        """
        if mode == 'train':
            # 注意：新的get_examples train模式不再调用此分支，但为保持一致性我们仍修正它
            file_path = os.path.join(data_dir, "origin_data", "train.tsv")
        elif mode == 'eval':
            # 修正eval模式的路径
            file_path = os.path.join(data_dir, "origin_data", "dev.tsv")
        elif mode == 'test':
            # 修正test模式的路径
            file_path = os.path.join(data_dir, "origin_data", "test.tsv")
        else:
            raise NotImplementedError(f"Mode {mode} not found")
        
        lines = self.read_tsv(file_path)
        examples = self.create_examples(lines, mode)
        return examples
        

    def read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
        

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 2:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
        
            
    def get_labels(self, data_dir):
        """See base class."""
        docs = os.listdir(data_dir)
        if "train.tsv" in docs:
            import pandas as pd
            test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
            labels = [str(label).lower() for label in test['label']]
            labels = np.unique(np.array(labels))
        elif "dataset.json" in docs:
            with open(os.path.join(data_dir, "dataset.json"), 'r') as f:
                dataset = json.load(f)
                dataset = dataset[list(dataset.keys())[0]]
            labels = []
            for dom in dataset:
                for ind, data in enumerate(dataset[dom]):
                    label = data[1][0]
                    labels.append(str(label).lower())
            labels = np.unique(np.array(labels))
        return labels
    

    def convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""
        label_map = {}
        for i, label in enumerate(label_list):
            label_map[label] = i

        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self.truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            label_id = label_map[example.label]

            features.append(
                InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id))
        return features


    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop(0)  # For dialogue context
            else:
                tokens_b.pop()

    
    def difference(self, a, b):
        _b = set(b)
        return [item for item in a if item not in _b]


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        self.configs:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """
        base features for a single training example
    """
    def __init__(self, input_ids, input_mask, segment_ids, label_id):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        

if __name__ == "__main__":
    
    config_path = '../methods/intent_generation/config.yaml'

    from utils import load_yaml_config
    from easydict import EasyDict
    configs = load_yaml_config(config_path)

    args = EasyDict(configs)

    base_data = BaseData(args)

    dataloader = base_data.eval_dataloader

    for idx, batch in enumerate(dataloader):
        print(batch)
        exit()

    






        