import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn import NLLLoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader


data = pd.read_csv("liar_dataset/train.csv", sep=',', header = None, usecols=[1, 2], names=["label", "statement"])

data['label'] = data['label'].apply(lambda x:1 if x in ["false", "pants-fire", "barely-true"] else 0)

x = data['statement']
y = data['label']

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

inputs = tokenizer(list(x), padding=True, truncation=True, return_tensors="pt")

attention_mask = inputs['attention_mask']
inputs_ids = inputs['input_ids']

y_array = y.values
skf = KFold(n_splits=5, shuffle=True, random_state=7)

for fold, (train_idx, val_idx) in enumerate(skf.split(inputs_ids, y_array)):
    
    inputs_ids_train = inputs_ids[train_idx]
    inputs_ids_val = inputs_ids[val_idx]
    y_train = torch.tensor(y_array[train_idx], dtype=torch.long)
    y_val = torch.tensor(y_array[val_idx], dtype=torch.long)

    attention_mask_train = attention_mask[train_idx]
    attention_mask_val = attention_mask[val_idx]

    train_dataset = TensorDataset(inputs_ids_train, attention_mask_train, y_train)
    val_dataset = TensorDataset(inputs_ids_val, attention_mask_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


    encoder = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    

    for param in encoder.parameters():
        param.requires_grad = False

    encoder.classifier = nn.Sequential(
        nn.Linear(768, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2),
        nn.LogSoftmax(dim=1)
    )

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_array), y=y_array)

    criterion = NLLLoss(weight=torch.tensor(class_weights, dtype=torch.float32))

    optimizer = optim.AdamW(encoder.classifier.parameters(), lr=1e-3)

    for epoch in range(5):

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        print(f"\nEpoch {epoch+1}/5")
        encoder.train()
        for batch in train_loader:
            inputs_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = encoder(inputs_ids, attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs_ids.size(0)
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        acc = train_correct / train_total
        print(f"Train Loss: {train_loss/train_total:.4f}, Accuracy: {acc:.4f}")

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        encoder.eval()
        with torch.no_grad():
            for inputs_ids, attention_mask, labels in val_loader:
                outputs = encoder(inputs_ids, attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)

                val_loss   += loss.item() * inputs_ids.size(0)
                preds       = logits.argmax(dim=1)
                val_total  += labels.size(0)
                val_correct+= (preds == labels).sum().item()

        avg_val_loss = val_loss / val_total
        val_acc      = val_correct / val_total
        print(f"Val Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}")
        