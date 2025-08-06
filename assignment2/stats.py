import matplotlib.pyplot as plt
import numpy as np

# --- Train Loss (son epoch'lar), Test Loss ve Accuracy verileri ---
train_loss_last_epoch = [0.6615, 0.6590, 0.6610, 0.6608, 0.6590]
test_loss = [0.6530, 0.6723, 0.6520, 0.6717, 0.7202]

val_acc = [0.6152, 0.5552, 0.5811, 0.5493, 0.5234]
test_acc = [0.6196, 0.5691, 0.6267, 0.5627, 0.5201]

folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']

# ------------------ 1. Train vs Test Loss ------------------
plt.figure(figsize=(9, 5))
plt.plot(folds, train_loss_last_epoch, marker='o', label='Train Loss (Last Epoch)', color='blue')
plt.plot(folds, test_loss, marker='s', label='Test Loss', color='red')

# Ortalama çizgiler
mean_train_loss = np.mean(train_loss_last_epoch)
mean_test_loss = np.mean(test_loss)
plt.axhline(y=mean_train_loss, color='blue', linestyle='--', label=f'Mean Train Loss ({mean_train_loss:.4f})')
plt.axhline(y=mean_test_loss, color='red', linestyle='--', label=f'Mean Test Loss ({mean_test_loss:.4f})')

plt.title('Train Loss vs Test Loss per Fold')
plt.xlabel('Fold')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss.png')

# ------------------ 2. Validation vs Test Accuracy ------------------
plt.figure(figsize=(9, 5))
plt.plot(folds, val_acc, marker='o', label='Validation Accuracy', color='green')
plt.plot(folds, test_acc, marker='s', label='Test Accuracy', color='orange')

# Ortalama çizgiler
mean_val_acc = np.mean(val_acc)
mean_test_acc = np.mean(test_acc)
plt.axhline(y=mean_val_acc, color='green', linestyle='--', label=f'Mean Val Acc ({mean_val_acc:.4f})')
plt.axhline(y=mean_test_acc, color='orange', linestyle='--', label=f'Mean Test Acc ({mean_test_acc:.4f})')

plt.title('Validation Accuracy vs Test Accuracy per Fold')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.ylim(0.5, 0.65)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy.png')
