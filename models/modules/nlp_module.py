from transformers import BertModel
import torch.nn as nn

class NLPModule(nn.Module):
    def __init__(self):
        super(NLPModule, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, text_input):
        return self.bert(**text_input).pooler_output
