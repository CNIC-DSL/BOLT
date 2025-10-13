from util import *

import pandas as pd


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Data:

    def __init__(self, args):
        set_seed(args.seed)

        processor = DatasetProcessor()
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        all_label_path = os.path.join(self.data_dir, "label", "label.list")
        self.all_label_list = pd.read_csv(all_label_path, header=None)[0].tolist()

        self.known_label_list = pd.read_csv(
            f"{args.data_dir}/{args.dataset}/label/{args.fold_type}{args.fold_num}/part{args.fold_idx}/label_known_{args.known_cls_ratio}.list",
            header=None,
        )[0].tolist()

        self.n_known_cls = len(self.known_label_list)

        self.known_lab = [self.all_label_list.index(a) for a in self.known_label_list]
        self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor)

        self.train_labeled_examples, self.train_unlabeled_examples = self.get_examples(
            processor, args, "train"
        )
        print("num_labeled_samples", len(self.train_labeled_examples))
        print("num_unlabeled_samples", len(self.train_unlabeled_examples))
        self.eval_examples = self.get_examples(processor, args, "eval")
        self.test_examples = self.get_examples(processor, args, "test")
        self.train_labeled_dataloader = self.get_loader(
            self.train_labeled_examples, args, "train"
        )

        (
            self.semi_input_ids,
            self.semi_input_mask,
            self.semi_segment_ids,
            self.semi_label_ids,
        ) = self.get_semi(
            self.train_labeled_examples, self.train_unlabeled_examples, args
        )
        self.train_semi_dataloader = self.get_semi_loader(
            self.semi_input_ids,
            self.semi_input_mask,
            self.semi_segment_ids,
            self.semi_label_ids,
            args,
        )

        self.eval_dataloader = self.get_loader(self.eval_examples, args, "eval")
        self.test_dataloader = self.get_loader(self.test_examples, args, "test")

    def get_examples(self, processor, args, mode="train"):

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
            ori_examples = processor.get_examples(self.data_dir, mode)

            if mode == "eval":

                eval_examples = []
                for example in ori_examples:
                    if example.label in self.known_label_list:
                        eval_examples.append(example)
                return eval_examples

            elif mode == "test":

                return ori_examples

    def get_semi(self, labeled_examples, unlabeled_examples, args):

        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        labeled_features = convert_examples_to_features(
            labeled_examples, self.known_label_list, args.max_seq_length, tokenizer
        )
        unlabeled_features = convert_examples_to_features(
            unlabeled_examples, self.all_label_list, args.max_seq_length, tokenizer
        )

        labeled_input_ids = torch.tensor(
            [f.input_ids for f in labeled_features], dtype=torch.long
        )
        labeled_input_mask = torch.tensor(
            [f.input_mask for f in labeled_features], dtype=torch.long
        )
        labeled_segment_ids = torch.tensor(
            [f.segment_ids for f in labeled_features], dtype=torch.long
        )
        labeled_label_ids = torch.tensor(
            [f.label_id for f in labeled_features], dtype=torch.long
        )

        unlabeled_input_ids = torch.tensor(
            [f.input_ids for f in unlabeled_features], dtype=torch.long
        )
        unlabeled_input_mask = torch.tensor(
            [f.input_mask for f in unlabeled_features], dtype=torch.long
        )
        unlabeled_segment_ids = torch.tensor(
            [f.segment_ids for f in unlabeled_features], dtype=torch.long
        )
        unlabeled_label_ids = torch.tensor(
            [-1 for f in unlabeled_features], dtype=torch.long
        )

        semi_input_ids = torch.cat([labeled_input_ids, unlabeled_input_ids])
        semi_input_mask = torch.cat([labeled_input_mask, unlabeled_input_mask])
        semi_segment_ids = torch.cat([labeled_segment_ids, unlabeled_segment_ids])
        semi_label_ids = torch.cat([labeled_label_ids, unlabeled_label_ids])
        return semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids

    def get_semi_loader(
        self, semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids, args
    ):
        semi_data = TensorDataset(
            semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids
        )
        semi_sampler = SequentialSampler(semi_data)
        semi_dataloader = DataLoader(
            semi_data, sampler=semi_sampler, batch_size=args.train_batch_size
        )

        return semi_dataloader

    def get_loader(self, examples, args, mode="train"):
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

        if mode == "train" or mode == "eval":
            features = convert_examples_to_features(
                examples, self.known_label_list, args.max_seq_length, tokenizer
            )
        elif mode == "test":
            features = convert_examples_to_features(
                examples, self.all_label_list, args.max_seq_length, tokenizer
            )

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        if mode == "train":
            sampler = RandomSampler(data)
            dataloader = DataLoader(
                data, sampler=sampler, batch_size=args.train_batch_size
            )
        elif mode == "eval" or mode == "test":
            sampler = SequentialSampler(data)
            dataloader = DataLoader(
                data, sampler=sampler, batch_size=args.eval_batch_size
            )

        return dataloader


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


class DataProcessor(object):
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:

                lines.append(line)
            return lines


class DatasetProcessor(DataProcessor):

    def get_examples(self, data_dir, mode):
        if mode == "train":
            file_path = os.path.join(data_dir, "origin_data", "train.tsv")
        elif mode == "eval":
            file_path = os.path.join(data_dir, "origin_data", "dev.tsv")
        elif mode == "test":
            file_path = os.path.join(data_dir, "origin_data", "test.tsv")
        else:
            raise ValueError("Invalid mode %s" % mode)

        return self._create_examples(self._read_tsv(file_path), mode)

    def get_labels(self, data_dir):
        import pandas as pd

        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        labels = np.unique(np.array(test["label"]))

        return labels

    def _create_examples(self, lines, set_type):
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


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    features = []
    for ex_index, example in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
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


def _truncate_seq_pair(tokens_a, tokens_b, max_length):

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()
