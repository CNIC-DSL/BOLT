# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import collections
import inspect
import math
import os
import re
import shutil
import fitlog
import warnings
import random
from tqdm import tqdm
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, NamedTuple
# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    hp_params,
    is_azureml_available,
    is_comet_available,
    is_mlflow_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
)
from utils import get_score

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.file_utils import WEIGHTS_NAME, is_datasets_available, is_in_notebook
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
# from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    get_tpu_sampler,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    HPSearchBackend,
    TrainOutput,
    default_compute_objective,
    set_seed,
)

import pandas as pd
from utils import *
from training_args import TrainingArguments, DataTrainingArguments, ModelArguments
from transformers.utils import logging

# Evaluation
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

_is_native_amp_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        pass
else:
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

# if is_torch_tpu_available():
#     import torch_xla.core.xla_model as xm
#     import torch_xla.debug.metrics as met
#     import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    from transformers.integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)


if is_wandb_available():
    from transformers.integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from transformers.integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)

if is_mlflow_available():
    from transformers.integrations import MLflowCallback

    DEFAULT_CALLBACKS.append(MLflowCallback)

if is_optuna_available():
    import optuna

if is_ray_available():
    pass

if is_azureml_available():
    from transformers.integrations import AzureMLCallback

    DEFAULT_CALLBACKS.append(AzureMLCallback)

# if is_fairscale_available():
#     pass

logger = logging.get_logger(__name__)


filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves']
filter_words = set(filter_words)


class MyEvalPrediction(NamedTuple):

    prediction_by_knn: Union[np.ndarray, Tuple[np.ndarray]]
    prediction_by_cls: Union[np.ndarray, Tuple[np.ndarray]]
    prediction_combine: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]


def l2norm(x: torch.Tensor):
    norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
    x = torch.div(x, norm)
    return x

class Evaluation:
    """
    This Train is simple version for Origin Trainer.
    This is to exce the origin contrastive learning.
    """
    def __init__(self,
                 model: Union[PreTrainedModel, torch.nn.Module] = None,
                 args: TrainingArguments = None,
                 data_args: DataTrainingArguments = None,
                 model_args: ModelArguments = None,
                 data_collator: Optional[DataCollator] = None,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None,
                 test_dataset: Optional[Dataset] = None,
                 tokenizer: Optional["PreTrainedTokenizerBase"] = None,
                 number_labels: Optional[int] = None,
                 eval_oos_dataset: Optional[Dataset] = None,
                 model_init: Callable[[], PreTrainedModel] = None,
                 compute_metrics: Optional[Callable[[MyEvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        if args is None:
            logger.info("No 'TrainingArgumenets' passed, using the current path as 'output_dir'" )
            args = TrainingArguments("tmp_trainer")
        self.args = args
        self.data_args = data_args
        self.model_args = model_args
        set_seed(self.args.seed)

        self.number_labels = number_labels
        self.OOS = False
        self.model = model
        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
        filepath = os.path.join(args.output_dir, 'model_best.pkl')
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.eval_oos_dataset = eval_oos_dataset
        self.tokenizer = tokenizer
        #self.label_names = default_label_names if self.args.label_names is None else self.args.label_names

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
                self.train_dataset, collections.abc.Sized
        ):
            return None
        # elif is_torch_tpu_available():
        #     return get_tpu_sampler(self.train_dataset)
        else:
            return (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.sampler.Sampler]:
        # if is_torch_tpu_available():
        #     return SequentialDistributedSampler(eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())
        if self.args.local_rank != -1:
            return SequentialDistributedSampler(eval_dataset)
        else:
            return SequentialSampler(eval_dataset)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        elif eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")
        elif is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            self._remove_unused_columns(eval_dataset, description="evaluation")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_eval_oos_dataloader(self, eval_oos_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_oos_dataset is None and self.eval_oos_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        elif eval_oos_dataset is not None and not isinstance(eval_oos_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")
        elif is_datasets_available() and isinstance(eval_oos_dataset, datasets.Dataset):
            self._remove_unused_columns(eval_oos_dataset, description="evaluation")
        eval_oos_dataset = eval_oos_dataset if eval_oos_dataset is not None else self.eval_oos_dataset
        eval_sampler = self._get_eval_sampler(eval_oos_dataset)

        return DataLoader(
            eval_oos_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                The test dataset to use. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if not isinstance(test_dataset, collections.abc.Sized):
            raise ValueError("test_dataset must implement __len__")
        elif is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            pass
            #self._remove_unused_columns(test_dataset, description="test")
        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            if self.sharded_dpp:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=AdamW,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            else:
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a :class:`~torch.utils.data.DataLoader` by accessing its dataset.

        Will raise an exception if the underlying dataset dese not implement method :obj:`__len__`
        """
        return len(dataloader.dataset)

    def evaluation(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
        ############## eval in validation ################################
        ############## for the theshold #################################
        predict = []
        target = []

        model = self.model.to(self.args.device)
        #model = torch.load(model_path, map_location=self.args.device)
        valid_loader = self.get_eval_dataloader()
        torch.no_grad()
        model.eval()

        # test in valid
        for step, inputs in enumerate(valid_loader):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)

            output = model(inputs, mode='validation')
            predict += output[1]
            target += output[0]

        f1 = metrics.f1_score(target, predict, average='macro')
        print(f"in-domain f1:{f1}")

        ################### predict ##########################################

        #valid_oos_loader = self.get_eval_oos_dataloader(self.eval_oos_dataset)
        valid_oos_loader = self.get_eval_oos_dataloader()
        train_loader = self.get_train_dataloader()
        test_loader = self.get_test_dataloader(self.test_dataset)

        feature_train = None
        feature_valid = None
        feature_valid_ood = None
        feature_test = None

        prob_train = None
        prob_valid = None
        prob_valid_ood = None
        prob_test = None

        with torch.no_grad():
            y_labels_train = None
            for step, inputs in enumerate(train_loader):
                if y_labels_train is None:
                    y_labels_train = inputs['labels']
                else:
                    y_labels_train = torch.cat((y_labels_train, inputs['labels']), dim=0)


                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.args.device)
                output = model(inputs, mode='test')
                if feature_train != None:
                    feature_train = torch.cat((feature_train, output[1]), dim=0)
                    prob_train = torch.cat((prob_train, output[0]), dim=0)
                else:
                    feature_train = output[1]
                    prob_train = output[0]

            valid_labels = None
            for step, inputs in enumerate(valid_loader):
                if valid_labels is None:
                    valid_labels = inputs['labels']
                else:
                    valid_labels = torch.cat((valid_labels, inputs['labels']), dim=0)

            for step, inputs in enumerate(valid_loader):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.args.device)
                output = model(inputs, mode='test')
                if feature_valid != None:
                    feature_valid = torch.cat((feature_valid, output[1]), dim=0)
                    prob_valid = torch.cat((prob_valid, output[0]), dim=0)
                else:
                    feature_valid = output[1]
                    prob_valid = output[0]

            valid_oos_labels = None
            for step, inputs in enumerate(valid_oos_loader):
                if valid_oos_labels is None:
                    valid_oos_labels = inputs['labels']
                else:
                    valid_oos_labels = torch.cat((valid_oos_labels, inputs['labels']), dim=0)

            for step, inputs in enumerate(valid_oos_loader):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.args.device)
                output = model(inputs, mode='test')
                if feature_valid_ood != None:
                    feature_valid_ood = torch.cat((feature_valid_ood, output[1]), dim=0)
                    prob_valid_ood = torch.cat((prob_valid_ood, output[0]), dim=0)
                else:
                    feature_valid_ood = output[1]
                    prob_valid_ood = output[0]


            valid_all_feature = torch.cat([feature_valid, feature_valid_ood], dim=0)
            valid_all_prob = torch.cat([prob_valid, prob_valid_ood], dim=0)
            valid_all_labels = torch.cat([valid_labels, valid_oos_labels], dim=0)


            labels_test = None
            for step, inputs in enumerate(test_loader):
                if labels_test is None:
                    labels_test = inputs['labels']
                else:
                    labels_test = torch.cat((labels_test, inputs['labels']), dim=0)

                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.args.device)

                output = model(inputs, mode='test')
                if feature_test != None:
                    feature_test = torch.cat((feature_test, output[1]), dim=0)
                    prob_test = torch.cat((prob_test, output[0]), dim=0)
                    #labels_test = torch.cat((labels_test, inputs['labels']), dim=0)
                else:
                    feature_test = output[1]
                    prob_test = output[0]
                    #labels_test = inputs['labels']

        feature_train = feature_train.cpu().detach().numpy()
        feature_valid = feature_valid.cpu().detach().numpy()
        feature_valid_ood = feature_valid_ood.cpu().detach().numpy()
        feature_test = feature_test.cpu().detach().numpy()
        prob_train = prob_train.cpu().detach().numpy()
        prob_valid = prob_valid.cpu().detach().numpy()
        prob_valid_ood = prob_valid_ood.cpu().detach().numpy()
        prob_test = prob_test.cpu().detach().numpy()

        valid_all_feature = valid_all_feature.cpu().detach().numpy()
        valid_all_prob = valid_all_prob.cpu().detach().numpy()
        valid_all_labels = valid_all_labels.cpu().detach().numpy()
        valid_labels = valid_labels.cpu().detach().numpy()

        if self.args.mode == 'find_threshold':
            # this setting refer to scl
            settings = ['gda_lsqr_' + str(10.0 + 1.0 * (i)) for i in range(20)]
        else:
            if isinstance(self.args.setting, str):
                settings = ['lof_cosine', 'lof_euclidean', 'gda']
                settings = ['lof_cosine', 'lof_euclidean']


        dis_metric = 'cosine'
        data_type = 'full'
        cl_rate = '0.1'
        for setting in settings:
            setting_fields = setting.split("_")
            ood_method = setting_fields[0]
            if len(setting_fields) > 1:
                dis_metric = setting_fields[1]
            if ood_method == 'lof':
                oos_index_test = []
                lof = LocalOutlierFactor(n_neighbors=20, metric=dis_metric, novelty=True,
                                         n_jobs=-1)
                lof.fit(feature_train)

                lof_score = lof.score_samples(feature_test)
                # lof_score_val = lof.score_samples(valid_all_feature)
                lof_score_val = lof.score_samples(feature_valid)
                for index, val in enumerate(labels_test):
                    if val == self.number_labels - 1:
                        oos_index_test.append(index)

                best_item = -1.5
                theshold_list = np.arange(-1.2, -10, -0.03)
                best_all_f1 = 0
                best_ind_f1 = 0
                best_oos_f1 = 0
                best_acc_ood = 0
                best_acc_in = 0
                for item in theshold_list:
                    # is_inlier = np.ones(len(valid_all_feature), dtype=int)

                    is_inlier = np.ones(len(feature_valid), dtype=int)

                    index_out = lof_score_val - item
                    is_inlier[index_out <= 0] = -1

                    #replace oos
                    # predict_label_val = np.argmax(valid_all_prob, axis=1)
                    
                    predict_label_val = np.argmax(prob_valid, axis=1)

                    for ind, val in enumerate(is_inlier):
                        if val == -1:
                            predict_label_val[ind] = self.number_labels - 1 # oos

                    classes = [i for i in range(self.number_labels)]
                    
                    # cm = confusion_matrix(valid_all_labels, predict_label_val)

                    cm = confusion_matrix(valid_labels, predict_label_val)

                    _, _, f_seen, acc_in, _, _, _, _, _, _ = get_score(cm, mode='valid')
                    # just only use in-domain data
                    if f_seen >= best_ind_f1 and acc_in >= best_acc_in:
                        best_item = item
                        best_ind_f1 = f_seen
                        best_acc_in = acc_in

                # if the dateset is clinc_full or clinc_small, the threashold of euclidean-based
                # should be different for cosine-based
                if "clinc" in self.args.dataset and "euclidean" in setting:
                    best_item -= -2.1
                is_inlier = np.ones(len(feature_test), dtype=int)
                # y_pred_lof = lof.predict(feature_test)
                index_out = lof_score - best_item
                is_inlier[index_out <= 0] = -1

                # replace oos
                predict_label_test = np.argmax(prob_test, axis=1)
                for ind, val in enumerate(is_inlier):
                    if val == -1:
                        predict_label_test[ind] = self.number_labels - 1  # oos

                classes = [i for i in range(self.number_labels)]
                cm = confusion_matrix(labels_test, predict_label_test)
                # f, f_seen, f_unseen, p_unseen, r_unseen = get_score(cm)
                print("this is %s" % setting)

                f, acc_all, f_seen, acc_in, p_seen, r_seen, f_unseen, acc_ood, p_unseen, r_unseen = get_score(cm)
                # ========================================================================
                # --- 全新的、标准化的结果保存逻辑 (最终版) ---
                # ========================================================================
                import pandas as pd
                import os
                
                # --- 步骤1：创建一个只包含最终指标的“单行”结果字典 ---
                final_results = {}

                # a. 添加实验参数元数据
                final_results['dataset'] = self.args.dataset
                final_results['seed'] = self.args.seed
                final_results['known_cls_ratio'] = self.args.known_cls_ratio
                final_results['ood_method'] = setting # 记录本次使用的是哪种OOD检测方法

                # b. 添加核心的总览性能指标 (f, acc_all等变量来自您提供的代码中的 get_score(cm) 的返回值)
                final_results['ACC'] = acc_all
                final_results['F1'] = f
                final_results['K-F1'] = f_seen
                final_results['N-F1'] = f_unseen
                # final_results['args'] = json.dumps(vars(self.args), ensure_ascii=False)
                def safe_args_to_json(args_obj):
                    safe_dict = {}
                    for k, v in vars(args_obj).items():
                        try:
                            json.dumps(v)  # 测试能否直接序列化
                            safe_dict[k] = v
                        except (TypeError, OverflowError):
                            safe_dict[k] = str(v)  # 不能序列化的转成字符串
                    return json.dumps(safe_dict, ensure_ascii=False)

                final_results['args'] = safe_args_to_json(self.args)

                # --- 步骤2：将“单行”结果追加保存到主 results.csv 文件 ---
                # 定义标准化的 metrics 输出目录和文件路径
                # metric_dir = os.path.join(self.args.output_dir, 'metrics')
                os.makedirs(self.data_args.save_results_path, exist_ok=True) # 关键：确保目录存在，解决OSError
                results_path = os.path.join(self.data_args.save_results_path, 'results.csv')

                df_to_save = pd.DataFrame([final_results])
                df_to_save['method'] = 'KnnCon'
                cols = ["method","dataset","known_cls_ratio","labeled_ratio","cluster_num_factor",
                        "seed","ACC","F1","K-F1","N-F1","args"]
                val = lambda c: next((getattr(src, c) for src in (self.data_args, self.model_args, self.args) if hasattr(src, c)), None)
                for c in cols:
                    df_to_save[c] = df_to_save.get(c, val(c))
                df_to_save = df_to_save[cols]

                if not os.path.exists(results_path):
                    # 如果文件不存在，直接创建并写入（包含表头）
                    df_to_save.to_csv(results_path, index=False)
                else:
                    # 如果文件存在，则读取->追加->写回
                    existing_df = pd.read_csv(results_path)
                    # new_row_df = pd.DataFrame([final_results])
                    updated_df = pd.concat([existing_df, df_to_save], ignore_index=True)
                    updated_df.to_csv(results_path, index=False)

                print(f"\nResults have been saved to: {results_path}")
                print("Appended new result row for setting '{}':".format(setting))
                print(pd.DataFrame([final_results]))
                # ========================================================================
                # --- 结果保存逻辑结束 ---
                # ======================================================================== 

