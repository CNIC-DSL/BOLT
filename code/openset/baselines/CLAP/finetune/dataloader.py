

import torch
# import json_lines
import json
from util import *
from collections import Counter
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Data:

    def __init__(self, args):
        set_seed(args.seed)
        
        # ========================================================================
        # --- 全新的、符合SOP标准的数据加载与处理逻辑 ---
        # ========================================================================
        
        # 1. 根据SOP标准从YAML参数构建文件路径
        origin_train_path = os.path.join(args.data_dir, args.dataset, 'origin_data', 'train.tsv')
        origin_eval_path = os.path.join(args.data_dir, args.dataset, 'origin_data', 'dev.tsv')
        origin_test_path = os.path.join(args.data_dir, args.dataset, 'origin_data', 'test.tsv')
        
        labeled_train_path = os.path.join(args.data_dir, args.dataset, 'labeled_data', str(args.labeled_ratio), 'train.tsv')
        
        # 2. 使用pandas加载标准化的TSV文件
        origin_train_df = pd.read_csv(origin_train_path, sep='\t')
        origin_eval_df = pd.read_csv(origin_eval_path, sep='\t')
        origin_test_df = pd.read_csv(origin_test_path, sep='\t')
        labeled_train_df = pd.read_csv(labeled_train_path, sep='\t')

        labeled_eval_path = os.path.join(args.data_dir, args.dataset, 'labeled_data', str(args.labeled_ratio), 'dev.tsv')
        labeled_eval_df = pd.read_csv(labeled_eval_path, sep='\t')
        
        # 3. 组合文本和标签信息
        df_train = labeled_train_df
        df_train['text'] = origin_train_df['text']
        df_eval = labeled_eval_df
        df_eval['text'] = origin_eval_df['text']
        
        # 4. 从标准化的.list文件加载已知类
        known_label_path = os.path.join(
            args.data_dir,
            args.dataset,
            'label',
            f'{args.fold_type}{args.fold_num}',
            f'part{args.fold_idx}',
            f'label_known_{args.known_cls_ratio}.list'
        )
        self.known_label_list = pd.read_csv(known_label_path, header=None)[0].tolist()
        self.num_labels = len(self.known_label_list)

        # 5. 筛选数据，构建examples
        train_examples_df = df_train[(df_train.label.isin(self.known_label_list)) & (df_train['labeled'].astype(bool))]
        eval_examples_df = df_eval[(df_eval.label.isin(self.known_label_list)) & (df_eval['labeled'].astype(bool))]

        
        # 测试集包含所有类别，用于后续阶段
        test_examples_df = origin_test_df

        # 将DataFrame转换为旧代码期望的InputExample格式
        processor = DatasetProcessor()
        self.train_examples = processor._create_examples_from_df(train_examples_df, 'train')
        self.eval_examples = processor._create_examples_from_df(eval_examples_df, 'eval')
        self.test_examples = processor._create_examples_from_df(test_examples_df, 'test')

        if args.dataset == 'oos':
            self.unseen_token = 'oos'
        else:
            self.unseen_token = '<UNK>'
        self.unseen_token_id = self.num_labels
        self.label_list = self.known_label_list + [self.unseen_token]
        
        # 6. 获取DataLoader (保留原逻辑)
        self.train_dataloader = self.get_loader(self.train_examples, args, 'train')
        self.eval_dataloader = self.get_loader(self.eval_examples, args, 'eval')
        self.test_dataloader = self.get_loader(self.test_examples, args, 'test')

        # 7. 设置其他属性 (保留原逻辑)
        if args.dataset == 'oos':
            self.unseen_token = 'oos'
        else:
            self.unseen_token = '<UNK>'
        self.unseen_token_id = self.num_labels
        self.label_list = self.known_label_list + [self.unseen_token]

    def read_jsonl(self, args):
        examples = []
        i = 1
        path = os.path.join(args.data_dir, '{}_v3.jsonl'.format(args.dataset))
        with open(path, "r", encoding="utf-8") as f:
            # for item in json_lines.reader(f):
            for line in f:
                item = json.loads(line)
                for text in item['generate_other']:
                    if len(text) > 0:
                        guid = "%s-%s" % (args.adbes_type, i)
                        text_a = text
                        label = self.gen_label
                        examples.append(
                            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
                        i += 1
        print('{}, len: {}'.format(path, len(examples)))
        return examples

    def get_examples(self, processor, args, mode='train'):
        mode_ = mode
        ori_examples = processor.get_examples(self.data_dir, mode_)

        examples = []
        if mode == 'train':
            for example in ori_examples:
                if (example.label in self.known_label_list) and (np.random.uniform(0, 1) <= args.labeled_ratio):
                    examples.append(example)
        elif mode == 'eval':
            for example in ori_examples:
                if (example.label in self.known_label_list):
                    examples.append(example)
        elif mode == 'test':
            for example in ori_examples:
                if (example.label in self.label_list) and (example.label is not self.unseen_token):
                    examples.append(example)
                else:
                    example.label = self.unseen_token
                    examples.append(example)
        print('{}, {}, before: {}, after: {} \n'.format(self.data_dir, mode, len(ori_examples), len(examples)))
        return examples

    def get_loader(self, examples, args, mode='train', neg_gen_examples=None):
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

        if mode == 'k_positive' and neg_gen_examples is not None:
            examples += neg_gen_examples
        features = convert_examples_to_features(examples, self.label_list, args.max_seq_length, tokenizer, self.unseen_token_id)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        
        datatensor = TextDataset(input_ids, input_mask, segment_ids, label_ids, args.kccl_k, args.neg_num)



        # 自定义 collate_fn
        def collate_fn_stack_to_column(batch):
            # 将 batch 中的数据堆叠为一列
            collated_batch = {}

            # Extract keys dynamically from the first sample
            keys = batch[0].keys()

            for key in keys:
                # Stack the tensors along the first dimension for each key
                collated_batch[key] = torch.concat([item[key] for item in batch], dim=0)
            
            return collated_batch
        
        if mode == 'train':
            sampler = RandomSampler(datatensor)
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size=args.train_batch_size, collate_fn=collate_fn_stack_to_column)
        elif mode == 'eval' or mode == 'test':
            sampler = SequentialSampler(datatensor)
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn_stack_to_column)
        elif mode == 'k_positive':
            sampler = RandomSampler(datatensor)
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size=args.train_batch_size, collate_fn=collate_fn_stack_to_column)
        print('mode: {}, len: {}'.format(mode, len(datatensor)))
        return dataloader

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, k_pos=0, n_neg=0,
                 mode='train', neg_label=None
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.label2sid = dict()    
        self.k = k_pos
        self.n = n_neg
        self.mode = mode
        self.neg_label = neg_label

        if k_pos > 0:
            for sid, i in enumerate(label_ids.detach().cpu().numpy()):
                if i not in self.label2sid:
                    self.label2sid[i] = [sid]
                else:
                    self.label2sid[i].append(sid)

    def generate_postive_sample(self, label, self_index):
        if self.k > 0:
            index_list = [ind for ind in self.label2sid[label] if ind != self_index]
            if len(index_list) == 0:
                index_list = [self_index]
            if self.k <= len(index_list):
                return np.random.choice(index_list, size=self.k, replace=False)
            else:
                return np.random.choice(index_list, size=self.k, replace=True)
        else:
            return None

    def generate_negtive_sample(self, label):
        if self.n > 0:
            index_list = []
            for key, value in self.label2sid.items():
                if key != label:
                    index_list += value
            return np.random.choice(index_list, size=self.n, replace=False)
        else:
            return None

    def __getitem__(self, idx):
        sid = self.generate_postive_sample(self.label_ids[idx].item(), idx)
        if self.n > 0:
            nid = self.generate_negtive_sample(self.label_ids[idx].item())
            sids = np.append([idx], sid)
            sids = np.append(sids, nid)
        else:
            sids = np.append([idx], sid)
        input_ids = self.input_ids[sids]    
        input_mask = self.input_mask[sids]
        segment_ids = self.segment_ids[sids]
        label_ids = self.label_ids[sids]
        assert input_ids.shape[0] % (1 + self.k + self.n) == 0
        return {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids, "label_ids": label_ids}

    def __len__(self):
        return len(self.label_ids)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
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
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class DatasetProcessor(DataProcessor):

    def get_examples(self, data_dir, mode):
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'eval':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
        elif mode == 'mytest':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "mytest75.tsv")), "mytest")

    def get_labels(self, data_dir):
        """See base class."""
        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        labels = np.unique(np.array(test['label']))
        return labels

    def _create_examples_from_df(self, df, set_type):
        """从DataFrame创建examples的辅助函数"""
        examples = []
        for i, row in df.iterrows():
            guid = "%s-%s" % (set_type, i)
            text_a = row['text']
            label = row['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        labels = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 2:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            labels.append(label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        print('{}, len: {}, labels: {} \n'.format(set_type, len(dict(Counter(labels))), dict(Counter(labels))))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, unseen_token_id):
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
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else: 
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
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

        label_id = label_map.get(example.label, unseen_token_id)
        
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  
        else:
            tokens_b.pop()
