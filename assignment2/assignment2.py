from google.colab  import drive
drive.mount("drive")

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.nn import NLLLoss
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train = pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/liar_dataset/train.csv", sep=',', header=None, usecols=[1,2], names=["label", "statement"])
test = pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/liar_dataset/test.csv", sep=',', header = None, usecols=[1, 2], names=["label", "statement"])

train['label'] = train['label'].apply(lambda x:1 if x in ["false", "pants-fire", "barely-true"] else 0)
test['label'] = test['label'].apply(lambda x:1 if x in ["false", "pants-fire", "barely-true"] else 0)

x = train['statement']
y = train['label']
z = test['statement']
w = test['label']

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

inputs = tokenizer(list(x), padding=True, truncation=True, max_length=256, return_tensors="pt")
test_inputs = tokenizer(list(z), padding=True, truncation=True, max_length=256, return_tensors="pt")

attention_mask = inputs['attention_mask']
input_ids = inputs["input_ids"]
y_array = y.values

test_attention_mask = test_inputs['attention_mask']
test_input_ids = test_inputs["input_ids"]
test_w_array = w.values

n= 5

skf = StratifiedKFold(n_splits =n, shuffle=True, random_state=10)

def create_model():
    model = AutoModel.from_pretrained("distilbert-base-uncased")

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(768, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2),
        nn.LogSoftmax(dim=1)
    )
    return model, classifier


epochs = 5
batch_size = 16
lr = 1e-3

fold_accuracies = []
fold_losses = []


for fold, (train_idx, val_idx) in enumerate(skf.split(input_ids, y_array)):
  print(f"\nFold {fold +1}")

  id1_train = input_ids[train_idx].to(device)
  id2_train = attention_mask[train_idx].to(device)
  id3_train = torch.tensor(y_array[train_idx], dtype=torch.long).to(device)

  id1_val = input_ids[val_idx].to(device)
  id2_val = attention_mask[val_idx].to(device)
  id3_val = torch.tensor(y_array[val_idx], dtype=torch.long).to(device)

  train_data = TensorDataset(id1_train, id2_train, id3_train)
  val_data = TensorDataset(id1_val, id2_val, id3_val)
  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

  encoder, classifier = create_model()
  encoder.to(device)
  classifier.to(device)

  optimizer = optim.AdamW(classifier.parameters(), lr=lr)
  class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_array), y=y_array)
  criterion = nn.NLLLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))



  for epoch in range(epochs):
      classifier.train()
      total_loss = 0.0

      for batch in train_loader:
        id1_batch, id2_batch, id3_batch = [x.to(device) for x in batch]

        optimizer.zero_grad()
        outputs = encoder(id1_batch, attention_mask=id2_batch)
        hidden_states = outputs.last_hidden_state[:, 0, :]
        logits = classifier(hidden_states)
        loss = criterion(logits, id3_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

      avg_loss = total_loss / len(train_loader)
      print(f"Epoch {epoch+1} - Avg Train Loss: {avg_loss:.4f}")

  fold_losses.append(avg_loss)


  classifier.eval()
  correct = 0
  total = 0
  with torch.no_grad():
      for batch in val_loader:
          id1_batch, id2_batch, id3_batch = [b.to(device) for b in batch]

          outputs = encoder(id1_batch, attention_mask=id2_batch)
          hidden_states = outputs.last_hidden_state[:, 0, :]

          logits = classifier(hidden_states)
          preds = torch.argmax(logits, dim=1)
          correct += (preds == id3_batch).sum().item()
          total += id3_batch.size(0)

  acc = correct / total
  print(f"\nFold {fold+1} - Validation Accuracy: {acc:.4f}")

  fold_accuracies.append(acc)
  torch.save(encoder.state_dict(), f"encoder_fold{fold}.pt")
  torch.save(classifier.state_dict(), f"classifier_fold{fold}.pt")

mean_acc = sum(fold_accuracies) / len(fold_accuracies)
mean_loss = sum(fold_losses) / len(fold_losses)

print(f"\nMean Validation Accuracy: {mean_acc:.4f}")
print(f"Mean Train Loss: {mean_loss:.4f}")


criterion = nn.NLLLoss()
test_losses = []
test_accuracies = []
test_dataset = TensorDataset(test_input_ids, test_attention_mask, torch.tensor(test_w_array, dtype=torch.long))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


for fold in range(n):
    encoder, classifier = create_model()
    encoder.load_state_dict(torch.load(f"encoder_fold{fold}.pt"))
    classifier.load_state_dict(torch.load(f"classifier_fold{fold}.pt"))
    encoder.to(device)
    classifier.to(device)

    encoder.eval()
    classifier.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            embeddings = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    test_losses.append(avg_loss)
    test_accuracies.append(accuracy)

    print(f"Fold {fold+1} - Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
