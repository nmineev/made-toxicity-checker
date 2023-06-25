import random
from transformers import BertModel, BertTokenizer
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


MAX_LEN = 128
CLASS_NAMES = ["normal", "toxic"]
SM = torch.nn.Sigmoid()
THRESHOLD = 0.7


class SentimentClassifier(nn.Module):

    def __init__(self, model_name_):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name_)
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


def load_model_tokenizer(model_name, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentimentClassifier(model_name_=model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer


@torch.no_grad()
def check_toxicity(text, model=None, tokenizer=None):
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
    prediction = (output_probs > THRESHOLD)
    return prediction, output_probs