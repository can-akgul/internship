import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np



target_classes = [3, 5]

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

all_train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
all_test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_index = []
for i in range(len(all_train_set.targets)):
    if all_train_set.targets[i] in target_classes:
        train_index.append(i)

test_index = []
for i in range(len(all_test_set.targets)):
    if all_test_set.targets[i] in target_classes:
        test_index.append(i)


train_subset = Subset(all_train_set, train_index)
test_subset = Subset(all_test_set, test_index)

def label_dataset(subset):
    for i in range(len(subset)):
        img, label = subset[i]
        original_label = label

        if label == 3:
            new_label = 0
        elif label == 5:
            new_label = 1
        else:
            new_label = -1

        subset.dataset.targets[subset.indices[i]] = new_label

label_dataset(train_subset)
label_dataset(test_subset)

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = correct / total
        print(f"Train Loss: {train_loss/total:.4f}, Accuracy: {acc:.4f}")

train_model(model, train_loader, val_loader, criterion, optimizer, epochs=25)

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

def evaluate_model(model, test_loader):
    model.eval()
    all_predictions = []
    all_labels = []
    all_images = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_images.extend(inputs.cpu())
    
    return np.array(all_predictions), np.array(all_labels), all_images


predictions, true_labels, test_images = evaluate_model(model, val_loader)

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='binary')
recall = recall_score(true_labels, predictions, average='binary')
f1 = f1_score(true_labels, predictions, average='binary')

print(f"\n=== Test Set Evaluation ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Cat (3)', 'Dog (5)'], 
            yticklabels=['Cat (3)', 'Dog (5)'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
plt.savefig('confusion_matrix.png')

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    return tensor * std.view(3, 1, 1) + mean.view(3, 1, 1)

def show_sample_images(images, true_labels, predictions, num_samples=8):
    correct_mask = (true_labels == predictions)
    correct_indices = np.where(correct_mask)[0]
    
    incorrect_mask = (true_labels != predictions)
    incorrect_indices = np.where(incorrect_mask)[0]
    
    class_names = ['Cat', 'Dog']
    
    if len(correct_indices) > 0:
        plt.figure(figsize=(15, 6))
        plt.suptitle('Correctly Classified Images', fontsize=16)
        
        num_correct = min(num_samples//2, len(correct_indices))
        for i in range(num_correct):
            idx = correct_indices[i]
            img = denormalize(images[idx])
            img = torch.clamp(img, 0, 1)
            
            plt.subplot(2, num_samples//2, i+1)
            plt.imshow(img.permute(1, 2, 0))
            plt.title(f'True: {class_names[true_labels[idx]]}\nPred: {class_names[predictions[idx]]}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        plt.savefig('correctly_classified_images.png')
    
    if len(incorrect_indices) > 0:
        plt.figure(figsize=(15, 6))
        plt.suptitle('Incorrectly Classified Images', fontsize=16)
        
        num_incorrect = min(num_samples//2, len(incorrect_indices))
        for i in range(num_incorrect):
            idx = incorrect_indices[i]
            img = denormalize(images[idx])
            img = torch.clamp(img, 0, 1)
            
            plt.subplot(2, num_samples//2, i+1)
            plt.imshow(img.permute(1, 2, 0))
            plt.title(f'Fact: {class_names[true_labels[idx]]}\nPrediction: {class_names[predictions[idx]]}', 
                     color='red')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        plt.savefig('incorrectly_classified_images.png')
    else:
        print("No incorrectly classified images found!")

show_sample_images(test_images, true_labels, predictions)

print(f"\nTotal test samples: {len(true_labels)}")
print(f"Correctly classified: {np.sum(true_labels == predictions)}")
print(f"Incorrectly classified: {np.sum(true_labels != predictions)}")
