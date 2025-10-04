from util import *

import pandas as pd
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
class Data:
    
    def __init__(self, args):
        set_seed(args.seed)
        # max_seq_lengths = {'clinc':30, 'stackoverflow':45,'banking':55,'hwu':55}
        # args.max_seq_length = max_seq_lengths[args.dataset]

        processor = DatasetProcessor()
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        all_label_path = os.path.join(self.data_dir, 'label', 'label.list')
        self.all_label_list = pd.read_csv(all_label_path, header=None)[0].tolist()

        # self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)

        # self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))
        self.known_label_list = pd.read_csv(f'{args.data_dir}/{args.dataset}/label/{args.fold_type}{args.fold_num}/part{args.fold_idx}/label_known_{args.known_cls_ratio}.list', header=None)[0].tolist()

        self.n_known_cls = len(self.known_label_list)

        # self.known_lab = [i for i in range(len(self.known_label_list))]
        self.known_lab = [self.all_label_list.index(a) for a in self.known_label_list]
        self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor)
        
        self.train_labeled_examples, self.train_unlabeled_examples = self.get_examples(processor, args, 'train')
        print('num_labeled_samples',len(self.train_labeled_examples))
        print('num_unlabeled_samples',len(self.train_unlabeled_examples))
        self.eval_examples = self.get_examples(processor, args, 'eval')
        self.test_examples = self.get_examples(processor, args, 'test')
        self.train_labeled_dataloader = self.get_loader(self.train_labeled_examples, args, 'train')

        self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids = self.get_semi(self.train_labeled_examples, self.train_unlabeled_examples, args)
        self.train_semi_dataloader = self.get_semi_loader(self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids, args)

        self.eval_dataloader = self.get_loader(self.eval_examples, args, 'eval')
        self.test_dataloader = self.get_loader(self.test_examples, args, 'test')
        
    def get_examples(self, processor, args, mode='train'):
    # --- 对于训练集，执行严谨的双重过滤 ---
        if mode == 'train':
            # 1. 定义并读取两个核心文件
            origin_data_path = os.path.join(self.data_dir, 'origin_data', 'train.tsv')
            labeled_info_path = os.path.join(self.data_dir, 'labeled_data', str(args.labeled_ratio), 'train.tsv')

            origin_data = pd.read_csv(origin_data_path, sep='\t')
            labeled_info = pd.read_csv(labeled_info_path, sep='\t')
            
            # 2. 合并信息：将原始文本添加到标签信息中
            # 假设两个文件行数和顺序完全对应
            merged_data = labeled_info
            merged_data['text'] = origin_data['text']
            
            # 3. 创建双重过滤的布尔掩码 (boolean mask)
            # 条件：标签必须在已知类别列表里 & labeled 字段必须为 True
            is_labeled_known = (merged_data['label'].isin(self.known_label_list)) & (merged_data['labeled'])
            
            # 4. 根据掩码分割为“有标签”和“无标签”两个 DataFrame
            labeled_df = merged_data[is_labeled_known]
            unlabeled_df = merged_data[~is_labeled_known] # ~is_labeled_known 表示取反

            # 5. 将两个 DataFrame 分别转换为 InputExample 对象列表
            train_labeled_examples = []
            for i, row in labeled_df.iterrows():
                guid = f"train_labeled-{i}"
                train_labeled_examples.append(InputExample(guid=guid, text_a=row['text'], label=row['label']))
            
            train_unlabeled_examples = []
            for i, row in unlabeled_df.iterrows():
                guid = f"train_unlabeled-{i}"
                train_unlabeled_examples.append(InputExample(guid=guid, text_a=row['text'], label=row['label']))
                
            return train_labeled_examples, train_unlabeled_examples

        # --- 对于验证集和测试集，使用旧的、但正确的加载逻辑 ---
        # （因为它们不需要复杂的分割，只需读取 origin_data 即可）
        else: # mode is 'eval' or 'test'
            ori_examples = processor.get_examples(self.data_dir, mode)
            
            if mode == 'eval':
                # 验证集：只包含已知类别的样本
                eval_examples = []
                for example in ori_examples:
                    if example.label in self.known_label_list:
                        eval_examples.append(example)
                return eval_examples

            elif mode == 'test':
                # 测试集：包含所有样本
                return ori_examples

    def get_semi(self, labeled_examples, unlabeled_examples, args):
        
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)    
        labeled_features = convert_examples_to_features(labeled_examples, self.known_label_list, args.max_seq_length, tokenizer)
        unlabeled_features = convert_examples_to_features(unlabeled_examples, self.all_label_list, args.max_seq_length, tokenizer)

        labeled_input_ids = torch.tensor([f.input_ids for f in labeled_features], dtype=torch.long)
        labeled_input_mask = torch.tensor([f.input_mask for f in labeled_features], dtype=torch.long)
        labeled_segment_ids = torch.tensor([f.segment_ids for f in labeled_features], dtype=torch.long)
        labeled_label_ids = torch.tensor([f.label_id for f in labeled_features], dtype=torch.long)      

        unlabeled_input_ids = torch.tensor([f.input_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_input_mask = torch.tensor([f.input_mask for f in unlabeled_features], dtype=torch.long)
        unlabeled_segment_ids = torch.tensor([f.segment_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_label_ids = torch.tensor([-1 for f in unlabeled_features], dtype=torch.long)      

        semi_input_ids = torch.cat([labeled_input_ids, unlabeled_input_ids])
        semi_input_mask = torch.cat([labeled_input_mask, unlabeled_input_mask])
        semi_segment_ids = torch.cat([labeled_segment_ids, unlabeled_segment_ids])
        semi_label_ids = torch.cat([labeled_label_ids, unlabeled_label_ids])
        return semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids

    def get_semi_loader(self, semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids, args):
        semi_data = TensorDataset(semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids)
        semi_sampler = SequentialSampler(semi_data)
        semi_dataloader = DataLoader(semi_data, sampler=semi_sampler, batch_size = args.train_batch_size) 

        return semi_dataloader


    def get_loader(self, examples, args, mode = 'train'):
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)    
        
        if mode == 'train' or mode == 'eval':
            features = convert_examples_to_features(examples, self.known_label_list, args.max_seq_length, tokenizer)
        elif mode == 'test':
            features = convert_examples_to_features(examples, self.all_label_list, args.max_seq_length, tokenizer)

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
        
        if mode == 'train':
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size = args.train_batch_size)    
        elif mode == 'eval' or mode == 'test':
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size = args.eval_batch_size) 
        
        return dataloader


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
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                # line = [l.lower() for l in line]
                lines.append(line)
            return lines

class DatasetProcessor(DataProcessor):

    def get_examples(self, data_dir, mode):
        if mode == 'train':
            file_path = os.path.join(data_dir, "origin_data", "train.tsv")
        elif mode == 'eval':
            file_path = os.path.join(data_dir, "origin_data", "dev.tsv")
        elif mode == 'test':
            file_path = os.path.join(data_dir, "origin_data", "test.tsv")
        else:
            raise ValueError("Invalid mode %s" % mode)

        return self._create_examples(self._read_tsv(file_path), mode)

    def get_labels(self, data_dir):
        """See base class."""
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        labels = np.unique(np.array(test['label']))
            
        return labels

    def _create_examples(self, lines, set_type):
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

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
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
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
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
        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()
