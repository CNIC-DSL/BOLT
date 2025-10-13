from typing import Tuple, List

from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

from my_args import OtherArguments


class Classifier(nn.Module):
    def __init__(self, config: BertConfig, num_ind_labels: int):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_ind_labels)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.tensor]:

        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        sentences_embeddings = self.dropout(pooled_output)

        logits = self.classifier(sentences_embeddings)
        return sentences_embeddings, logits


class BertForSequenceClassificationWithPabee(nn.Module):
    def __init__(
        self,
        pertained_config: BertConfig,
        other_args: OtherArguments,
        num_ind_labels: int,
    ):
        super().__init__()
        self.num_ind_labels = num_ind_labels

        self.bert: BertModel = BertModel.from_pretrained(
            other_args.bert_model, config=pertained_config
        )

        self.classifiers = nn.ModuleList(
            [
                Classifier(pertained_config, num_ind_labels)
                for _ in range(pertained_config.num_hidden_layers)
            ]
        )

        self.loss_type = other_args.loss_type
        self.diversity_loss_weight = other_args.diversity_loss_weight
        assert 1 >= self.diversity_loss_weight >= 0

    def forward_each_layer(
        self, query
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        query = {
            key: value
            for key, value in query.items()
            if key not in ["labels", "original_text", "sent_id"]
        }

        bert_output = self.bert(output_hidden_states=True, return_dict=True, **query)

        hidden_states = bert_output.hidden_states

        feats, logits = [], []

        for index, cur_layer_hidden_states in enumerate(hidden_states[1:]):

            cur_layer_pooled_output, cur_layer_logit = self.classifiers[index](
                cur_layer_hidden_states
            )

            feats.append(cur_layer_pooled_output)
            logits.append(cur_layer_logit)

        return feats, logits

    def forward(self, query, mode):
        labels = query["labels"]
        labels = labels.view(-1)

        feats, logits = self.forward_each_layer(query)

        if mode == "eval":

            return torch.stack(feats, dim=0), torch.stack(logits, dim=0)

        if mode == "train":
            if self.loss_type == "original":
                losses = 0

                for layer_index, cur_layer_logits in enumerate(logits):

                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        cur_layer_logits.view(-1, self.num_ind_labels), labels
                    )

                    losses += loss
                return losses
            if self.loss_type == "increase":
                total_loss = 0
                total_weights = 0
                for layer_index, cur_layer_logits in enumerate(logits):
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        cur_layer_logits.view(-1, self.num_ind_labels), labels
                    )

                    total_loss += loss * (layer_index + 1)
                    total_weights += layer_index + 1
                loss = total_loss / total_weights
                return loss

            if self.loss_type == "plain":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits[-1].view(-1, self.num_ind_labels), labels)
                return loss

            if self.loss_type == "ce_and_div":

                total_ce_loss = 0
                for layer_index, cur_layer_logits in enumerate(logits):
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        cur_layer_logits.view(-1, self.num_ind_labels), labels
                    )
                    total_ce_loss += loss

                total_diversity_loss = 0
                scores = [
                    torch.softmax(cur_layer_logits, dim=1)
                    for cur_layer_logits in logits
                ]
                for layer_index_i, cur_layer_logits in enumerate(logits[1:], start=1):
                    min_ce_loss = None
                    for layer_index_j in range(layer_index_i):
                        loss_fct = CrossEntropyLoss()
                        ce_loss_i_j = loss_fct(cur_layer_logits, scores[layer_index_j])
                        if min_ce_loss is None:
                            min_ce_loss = ce_loss_i_j
                        else:
                            min_ce_loss = min(min_ce_loss, ce_loss_i_j)
                    total_diversity_loss += min_ce_loss

                return total_ce_loss - self.diversity_loss_weight * total_diversity_loss

            if self.loss_type == "ce_and_div_drop-last-layer":

                total_ce_loss = 0
                for layer_index, cur_layer_logits in enumerate(logits):
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        cur_layer_logits.view(-1, self.num_ind_labels), labels
                    )
                    total_ce_loss += loss

                total_diversity_loss = 0
                scores = [
                    torch.softmax(cur_layer_logits, dim=1)
                    for cur_layer_logits in logits
                ]

                for layer_index_i, cur_layer_logits in enumerate(logits[1:-1], start=1):
                    min_ce_loss = None
                    for layer_index_j in range(layer_index_i):
                        loss_fct = CrossEntropyLoss()
                        ce_loss_i_j = loss_fct(cur_layer_logits, scores[layer_index_j])
                        if min_ce_loss is None:
                            min_ce_loss = ce_loss_i_j
                        else:
                            min_ce_loss = min(min_ce_loss, ce_loss_i_j)
                    total_diversity_loss += min_ce_loss

                return total_ce_loss - self.diversity_loss_weight * total_diversity_loss

        raise ImportError(f"unknown mode {mode} or loss_type {self.loss_type}")
