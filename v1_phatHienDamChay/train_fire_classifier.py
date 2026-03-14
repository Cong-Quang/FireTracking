import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import cv2
from pathlib import Path
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_DIR = Path("fire_dataset")

FIRE_DIR = DATASET_DIR / "fire_images"
NON_FIRE_DIR = DATASET_DIR / "non_fire_images"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-4


# =========================
# DATASET
# =========================

class FireDataset(Dataset):

    def __init__(self, transform=None):

        self.samples = []

        for p in FIRE_DIR.glob("*.*"):
            self.samples.append((str(p), 1))

        for p in NON_FIRE_DIR.glob("*.*"):
            self.samples.append((str(p), 0))

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, label


# =========================
# TRANSFORMS
# =========================

train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.3),
    transforms.ToTensor(),
])

val_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
])


# =========================
# LOAD DATA
# =========================

dataset = FireDataset(transform=train_tf)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

val_ds.dataset.transform = val_tf

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# =========================
# MODEL
# =========================

model = models.resnet18(weights="IMAGENET1K_V1")

model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(DEVICE)


# =========================
# TRAIN SETUP
# =========================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# =========================
# VALIDATION
# =========================

def evaluate():

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x,y in val_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            out = model(x)

            pred = torch.argmax(out,1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


# =========================
# TRAIN LOOP
# =========================

for epoch in range(EPOCHS):

    model.train()

    loop = tqdm(train_loader)

    for x,y in loop:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        out = model(x)

        loss = criterion(out,y)

        loss.backward()

        optimizer.step()

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

    acc = evaluate()

    print("VAL ACC:", acc)


torch.save(model.state_dict(),"fire_classifier.pt")

print("Model saved: fire_classifier.pt")