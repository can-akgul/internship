Selected dog (5) and cat (3) indices from CIFAR-10.

Loaded the dataset and labeled cat=0, dog=1.

Froze pretrained ResNet18 and added a final layer with 2 outputs (cat, dog).

For fine-tuning, didn’t freeze—directly connected the last layer to 2 outputs.

Loss: CrossEntropyLoss; Optimizer: Adam.

In training_evaluation:

optimizer.zero_grad() → forward → loss → loss.backward() → optimizer.step() (weights updated)

Transfer learning: update only last layer’s weights; fine-tuning: update all layers’ weights.

Calculated loss and accuracy for training and validation.

Used save_sample_images to save 10 samples (5 correct, 5 wrong), denormalized back to original.

Printed test scores; plotted confusion matrix and train/validation metrics.
