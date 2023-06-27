import random
from transformers import BertModel, BertTokenizer
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


MAX_LEN = 128
CLASS_NAMES = ["normal", "toxic"]
SM = torch.nn.Sigmoid()
DEFAULT_THRESHOLD = 0.7


class LoRALayer(nn.Module):
    """Wraps a linear layer with LoRA-like adapter. Wraps an existing OPT linear layer"""
    def __init__(self, module: nn.Linear, rank: int):
        super().__init__()
        self.module = module
        self.adapter = nn.Sequential(
            nn.Linear(module.in_features, rank, bias=False),
            nn.Linear(rank, module.out_features, bias=False)
        )
        nn.init.kaiming_uniform_(self.adapter[0].weight, a=5 ** 0.5)
        nn.init.zeros_(self.adapter[1].weight)

        self.adapter.to(module.weight.device)

    def forward(self, input):
        # Apply self.module and LoRA adapter, return the sum (base module outputs + adapter outputs)
        return self.module(input) + self.adapter(input)


class SentimentClassifier(nn.Module):

    def __init__(self, model_name_, model_path_):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name_)
        if model_path_ == "./toxicity_checker/data/model_dp_512/model_dp_512.bin":
            for name, module in self.bert.named_modules():
                if 'BertSelfAttention' in repr(type(module)):
                    module.query = LoRALayer(module.query, rank=512)
                    module.key = LoRALayer(module.key, rank=512)
                    module.value = LoRALayer(module.value, rank=512)
        self.out = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 600),
            nn.ReLU(),
            nn.BatchNorm1d(600),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.BatchNorm1d(600),
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.BatchNorm1d(300),
            nn.Linear(300, 1),
            )

    def forward(self, input_ids, attention_mask):
        _, output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask,
          return_dict=False
        )
        return self.out(output)


@torch.no_grad()
def load_model_tokenizer(model_name, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentimentClassifier(model_name_=model_name, model_path_=model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer


@torch.no_grad()
def check_toxicity(text, model=None, tokenizer=None, threshold=DEFAULT_THRESHOLD):
    if model is None:
        toxicity_score = random.random()
        return toxicity_score > 0.5, toxicity_score

    device = next(iter(model.parameters()))
    encoded_review = tokenizer.encode_plus(
        text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt',
    )
    input_ids = encoded_review['input_ids'].to(device).to(torch.int)
    attention_mask = encoded_review['attention_mask'].to(device)

    model.eval()
    output = model(input_ids, attention_mask)
    output_probs = SM(output.flatten()).item()
    prediction = (output_probs > threshold)
    return prediction, output_probs
