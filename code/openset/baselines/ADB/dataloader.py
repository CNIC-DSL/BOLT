from util import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Data:

    def __init__(self, args):
        set_seed(args.seed)
        args.max_seq_length = 512
        self.known_label_list = pd.read_csv(
            f"{args.data_dir}/{args.dataset}/label/{args.fold_type}{args.fold_num}/part{args.fold_idx}/label_known_{args.known_cls_ratio}.list",
            header=None,
        )[0].tolist()
        self.num_labels = len(self.known_label_list)

        if args.dataset == "oos":
            self.unseen_token = "oos"
        else:
            self.unseen_token = "<UNK>"

        self.unseen_token_id = self.num_labels
        self.label_list = self.known_label_list + [self.unseen_token]

        processor = DatasetProcessor()
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.train_examples = self.get_examples(processor, args, "train")
        self.eval_examples = self.get_examples(processor, args, "eval")
        self.test_examples = self.get_examples(processor, args, "test")

        self.train_dataloader = self.get_loader(self.train_examples, args, "train")
        self.eval_dataloader = self.get_loader(self.eval_examples, args, "eval")
        self.test_dataloader = self.get_loader(self.test_examples, args, "test")

    def get_examples(self, processor, args, mode="train"):
        if mode == "train" or mode == "eval":
            split_name = "dev" if mode == "eval" else "train"
            origin_data_path = os.path.join(
                self.data_dir, "origin_data", f"{split_name}.tsv"
            )
            labeled_info_path = os.path.join(
                self.data_dir,
                "labeled_data",
                str(args.labeled_ratio),
                f"{split_name}.tsv",
            )

            origin_data = pd.read_csv(origin_data_path, sep="\t")
            labeled_info = pd.read_csv(labeled_info_path, sep="\t")

            merged_data = labeled_info
            merged_data["text"] = origin_data["text"]

            if mode == "train":

                final_data = merged_data[
                    (merged_data["label"].isin(self.known_label_list))
                    & (merged_data["labeled"])
                ]
            else:
                final_data = merged_data[
                    (merged_data["label"].isin(self.known_label_list))
                ]

            examples = []
            for i, row in final_data.iterrows():
                guid = f"{mode}-{i}"
                examples.append(
                    InputExample(
                        guid=guid, text_a=row["text"], text_b=None, label=row["label"]
                    )
                )
            return examples

        elif mode == "test":
            data_path = os.path.join(self.data_dir, "origin_data")
            ori_examples = processor.get_examples(data_path, "test")

            examples = []
            for example in ori_examples:
                if example.label in self.known_label_list:
                    examples.append(example)
                else:
                    example.label = self.unseen_token
                    examples.append(example)
            return examples

    def get_loader(self, examples, args, mode="train"):
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        features = convert_examples_to_features(
            examples, self.label_list, args.max_seq_length, tokenizer
        )
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        datatensor = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        if mode == "train":
            sampler = RandomSampler(datatensor)
            dataloader = DataLoader(
                datatensor,
                sampler=sampler,
                batch_size=args.train_batch_size,
                drop_last=False,
            )
        elif mode == "eval" or mode == "test":
            sampler = SequentialSampler(datatensor)
            dataloader = DataLoader(
                datatensor,
                sampler=sampler,
                batch_size=args.eval_batch_size,
                drop_last=False,
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

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r") as f:
            csv.field_size_limit(500 * 1024 * 1024)
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, "utf-8") for cell in line)
                lines.append(line)
            return lines


class DatasetProcessor(DataProcessor):

    def get_examples(self, data_dir, mode):
        if mode == "train":
            file_path = os.path.join(data_dir, "train.tsv")
            return self._create_examples(self._read_tsv(file_path), "train")
        elif mode == "eval":
            file_path = os.path.join(data_dir, "dev.tsv")
            return self._create_examples(self._read_tsv(file_path), "eval")
        elif mode == "test":
            file_path = os.path.join(data_dir, "test.tsv")
            return self._create_examples(self._read_tsv(file_path), "test")

    def get_labels(self, data_dir):
        import pandas as pd

        file_path = os.path.join(data_dir, "train.tsv")
        train_df = pd.read_csv(file_path, sep="\t")
        labels = np.unique(np.array(train_df["label"]))
        return labels

    def _create_examples(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines[1:]):
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
