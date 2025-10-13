from utils.tools import *


class Data:

    def __init__(self, args):
        set_seed(args.seed)
        args.pretrain_dir = os.path.join(
            args.pretrain_dir,
            f"premodel_{args.dataset}_{args.known_cls_ratio}_{args.seed}",
        )

        processor = DatasetProcessor()
        args.cluster_num_factor = 1
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
        args.num_labeled_examples = len(self.train_labeled_examples)
        self.eval_examples = self.get_examples(processor, args, "eval")
        self.test_examples = self.get_examples(processor, args, "test")

        if self.n_known_cls > 0:
            self.train_labeled_dataloader = self.get_loader(
                self.train_labeled_examples, args, "train"
            )
        else:
            self.train_labeled_dataloader = None

        (
            self.semi_input_ids,
            self.semi_input_mask,
            self.semi_segment_ids,
            self.semi_label_ids,
            self.semi_idx_ids,
        ) = self.get_semi(
            self.train_labeled_examples, self.train_unlabeled_examples, args
        )

        self.train_semi_dataset, self.train_semi_dataloader = self.get_semi_loader(
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
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
        if self.n_known_cls > 0:
            labeled_features = convert_examples_to_features(
                labeled_examples, self.known_label_list, args.max_seq_length, tokenizer
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
        else:
            labeled_features = None

        unlabeled_features = convert_examples_to_features(
            unlabeled_examples, self.all_label_list, args.max_seq_length, tokenizer
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
            [f.label_id for f in unlabeled_features], dtype=torch.long
        )

        if self.n_known_cls > 0:
            semi_input_ids = torch.cat([labeled_input_ids, unlabeled_input_ids])
            semi_input_mask = torch.cat([labeled_input_mask, unlabeled_input_mask])
            semi_segment_ids = torch.cat([labeled_segment_ids, unlabeled_segment_ids])
            semi_label_ids = torch.cat([labeled_label_ids, unlabeled_label_ids])
        else:
            semi_input_ids = unlabeled_input_ids
            semi_input_mask = unlabeled_input_mask
            semi_segment_ids = unlabeled_segment_ids
            semi_label_ids = unlabeled_label_ids

        idx_list = np.array(
            [int(_.guid.split("-")[-1]) for _ in labeled_examples]
            + [int(_.guid.split("-")[-1]) for _ in unlabeled_examples]
        )

        return (
            semi_input_ids,
            semi_input_mask,
            semi_segment_ids,
            semi_label_ids,
            idx_list,
        )

    def get_semi_loader(
        self, semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids, args
    ):
        semi_data = TensorDataset(
            semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids
        )
        semi_sampler = SequentialSampler(semi_data)
        semi_dataloader = DataLoader(
            semi_data,
            sampler=semi_sampler,
            batch_size=args.train_batch_size,
            drop_last=False,
        )

        return semi_data, semi_dataloader

    def get_loader(self, examples, args, mode="train"):
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

        if mode == "train" or mode == "eval":
            features = convert_examples_to_features(
                examples, self.known_label_list, args.max_seq_length, tokenizer
            )
        elif mode == "test":
            features = convert_examples_to_features(
                examples, self.all_label_list, args.max_seq_length, tokenizer
            )
        else:
            raise NotImplementedError(f"Mode {mode} not found")

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        if mode == "train":
            sampler = RandomSampler(data)
            dataloader = DataLoader(
                data, sampler=sampler, batch_size=args.pretrain_batch_size
            )
        elif mode in ["eval", "test"]:
            sampler = SequentialSampler(data)
            dataloader = DataLoader(
                data, sampler=sampler, batch_size=args.eval_batch_size
            )
        else:
            raise NotImplementedError(f"Mode {mode} not found")

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
    for index, example in enumerate(examples):
        tokens = tokenizer(
            example.text_a,
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
        )

        input_ids = tokens["input_ids"]
        input_mask = tokens["attention_mask"]
        segment_ids = tokens["token_type_ids"]
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
