# from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME, BertPreTrainedModel,BertModel
# from pytorch_pretrained_bert.tokenization import BertTokenizer

# BertPreTrainedModel.from_pretrained('../../llms/bert-base-uncased')


from transformers import AutoModel, AutoTokenizer
import torch
model = AutoModel.from_pretrained('../../llms/bert-base-chinese')
torch.save(model.state_dict(), '../../llms/bert-base-chinese/pytorch_model.bin')