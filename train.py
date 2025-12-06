import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os

# ========================
# CONFIG
# ========================
DATA_DIR = "data/train"
TEST_DIR = "data/test"
BATCH_SIZE = 16
IMG_SIZE = 224
EPOCHS = 10
VAL_SPLIT = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========================
# TRANSFORMS
# ========================
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ========================
# DATASET LOADING
# ========================
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)

val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

classes = full_dataset.classes
print("Classes:", classes)

# ========================
# MODEL - RESNET50 (HIGH ACCURACY)
# ========================
model = models.resnet50(pretrained=True)

# Freeze feature extractor
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2)  # cracked / non_cracked
)

model = model.to(device)

# ========================
# LOSS & OPTIMIZER
# ========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0008)

# ========================
# TRAINING LOOP
# ========================
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()

    train_accuracy = correct / len(train_dataset)

    # Validation
    model.eval()
    val_correct = 0
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels).item()

    val_accuracy = val_correct / len(val_dataset)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}")

# ========================
# SAVE MODEL
# ========================
torch.save(model.state_dict(), "crack_detector_resnet50.pth")
print("Model saved successfully!")
