import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader

data = pd.read_csv("liar_dataset/train.csv", sep=',', header=None, usecols=[1, 2], names=["label", "statement"])

data['is_fake'] = data['label'].apply(lambda x:1 if x in ["false", "pants-fire", "barely-true"] else 0)

x, y = data['statement'], data['is_fake']

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

inputs = tokenizer(list(x), padding=True, truncation=True, return_tensors="pt")


model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Linear(768, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 2),
    nn.LogSoftmax(dim=1)
)