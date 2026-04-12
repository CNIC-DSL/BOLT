from utils.tools import *
from utils.contrastive import SupConLoss

class BertForModel(nn.Module):
    def __init__(self,model_name, num_labels, device=None):
        super(BertForModel, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.device = device
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.classifier = nn.Linear(768, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.backbone.to(self.device)
        self.classifier.to(self.device)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, output_hidden_states=False, output_attentions=False, **kwargs):
        """logits are not normalized by softmax in forward function"""
        # Ensure we are passing tensors to backbone
        if isinstance(input_ids, dict):
            input_dict = input_ids
        else:
            input_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

        outputs = self.backbone(**input_dict, output_hidden_states=True)
        CLSEmbedding = outputs.hidden_states[-1][:,0]
        CLSEmbedding = self.dropout(CLSEmbedding)
        logits = self.classifier(CLSEmbedding)
        output_dir = {"logits": logits}
        if output_hidden_states:
            output_dir["hidden_states"] = outputs.hidden_states[-1][:, 0]
        return output_dir

    def mlmForward(self, X, Y):
        outputs = self.backbone(**X, labels=Y)
        return outputs.loss

    def loss_ce(self, logits, Y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output

    def save_backbone(self, save_path):
        self.backbone.save_pretrained(save_path)


class CLBert(nn.Module):
    def __init__(self, model_name, device, num_labels, feat_dim=128):
        super(CLBert, self).__init__()
        self.model_name = model_name
        self.device = device
        self.num_labels = num_labels
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, feat_dim)
        )
        self.classifier = nn.Linear(feat_dim, self.num_labels)
        self.backbone.to(self.device)
        self.head.to(device)
        self.classifier.to(device)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, output_hidden_states=False, output_attentions=False, output_logits=False, **kwargs):
        """logits are not normalized by softmax in forward function"""
        if isinstance(input_ids, dict):
            input_dict = input_ids
        else:
            input_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

        outputs = self.backbone(**input_dict, output_hidden_states=True, output_attentions=True)
        cls_embed = outputs.hidden_states[-1][:,0]
        features = F.normalize(self.head(cls_embed), dim=1)
        output_dir = {"features": features}
        logits = self.classifier(self.head(cls_embed))
        output_dir["logits"] = logits
        if output_hidden_states:
            output_dir["hidden_states"] = cls_embed
        if output_attentions:
            output_dir["attentions"] = outputs.attentions
        return output_dir

    def loss_cl(self, embds, label=None, mask=None, temperature=0.07, base_temperature=0.07):
        """compute contrastive loss"""
        loss = SupConLoss(temperature=temperature, base_temperature=base_temperature)
        output = loss(embds, labels=label, mask=mask)
        return output

    def loss_ce(self, logits, Y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output

    def save_backbone(self, save_path):
        self.backbone.save_pretrained(save_path)
