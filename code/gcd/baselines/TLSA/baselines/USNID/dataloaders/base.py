import numpy as np
import os
import logging
import pandas as pd

from .__init__ import max_seq_lengths, backbone_loader_map


class DataManager:

    def __init__(self, args, logger_name = 'Discovery'):

        self.logger = logging.getLogger(logger_name)
        args.max_seq_length = max_seq_lengths[args.dataset]
        self.data_dir = os.path.join(args.data_dir, args.dataset)

        self.all_label_list = self.get_labels(self.data_dir)
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        self.known_label_list = pd.read_csv(f"{self.data_dir}/label/label_{args.known_cls_ratio}.list", header=None)[0].tolist()

        # For non-banking datasets, all_label_list is already lowercased in get_labels; lowercase known labels too for correct index matching
        if args.dataset != 'banking':
            self.known_label_list = [str(label).lower() for label in self.known_label_list]

        self.known_lab = [int(np.where(self.all_label_list== a)[0]) for a in self.known_label_list]

        self.logger.info('The number of known intents is %s', self.n_known_cls)
        self.logger.info('Lists of known labels are: %s', str(self.known_label_list))

        args.num_labels = self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor)
        self.dataloader = self.get_loader(args, self.get_attrs())


    def get_labels(self, data_dir):
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        if os.path.basename(os.path.normpath(data_dir)) == 'banking':
            labels = np.unique(np.array(test['label']))
        else:
            labels = [str(label).lower() for label in test['label']]
            labels = np.unique(np.array(labels))

        return labels

    def get_loader(self, args, attrs):

        dataloader = backbone_loader_map[args.backbone](args, attrs)

        return dataloader

    def get_attrs(self):

        attrs = {}
        for name, value in vars(self).items():
            attrs[name] = value

        return attrs
