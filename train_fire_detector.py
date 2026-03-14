import torch
import cv2
import numpy as np
import time
from datetime import timedelta
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# ========================
# CONFIG
# ========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path("fire_dataset")

FIRE = ROOT / "fire_images"
NON_FIRE = ROOT / "non_fire_images"
LABELS = ROOT / "labels"

EPOCHS = 25
BATCH_SIZE = 2
LR = 1e-4
LOG_INTERVAL = 5

print("Device:", DEVICE)


# ========================
# DATASET
# ========================

class FireDataset(Dataset):

    def __init__(self):

        self.samples = []

        for p in FIRE.glob("*.*"):
            self.samples.append(p)

        for p in NON_FIRE.glob("*.*"):
            self.samples.append(p)

        print("Total images:", len(self.samples))


    def __len__(self):
        return len(self.samples)


    def load_boxes(self, label_path, w, h):

        boxes = []

        if not label_path.exists():
            return boxes

        with open(label_path) as f:

            for line in f:

                c, cx, cy, bw, bh = map(float, line.split())

                x1 = (cx - bw/2) * w
                y1 = (cy - bh/2) * h
                x2 = (cx + bw/2) * w
                y2 = (cy + bh/2) * h

                boxes.append([x1,y1,x2,y2])

        return boxes


    def __getitem__(self, idx):

        img_path = self.samples[idx]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h,w = img.shape[:2]

        label_path = LABELS / (img_path.stem + ".txt")

        boxes = self.load_boxes(label_path,w,h)

        if len(boxes)==0:

            boxes = torch.zeros((0,4),dtype=torch.float32)
            labels = torch.zeros((0,),dtype=torch.int64)

        else:

            boxes = torch.tensor(boxes,dtype=torch.float32)
            labels = torch.ones((len(boxes),),dtype=torch.int64)

        img = F.to_tensor(img)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return img,target


# ========================
# COLLATE
# ========================

def collate(batch):

    imgs = []
    targets = []

    for b in batch:

        imgs.append(b[0])
        targets.append(b[1])

    return imgs,targets


# ========================
# LOAD DATA
# ========================

dataset = FireDataset()

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    collate_fn=collate
)

print("Batches per epoch:", len(loader))


# ========================
# MODEL
# ========================

print("Loading model...")

model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

num_classes = 2

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features,
    num_classes
)

model = model.to(DEVICE)

print("Model ready")


# ========================
# OPTIMIZER
# ========================

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ========================
# TRAIN
# ========================

print("Start training...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    epoch_start = time.time()

    num_batches = len(loader)

    for batch_idx,(imgs,targets) in enumerate(loader, start=1):

        batch_start = time.time()

        imgs = [img.to(DEVICE) for img in imgs]

        targets = [{k:v.to(DEVICE) for k,v in t.items()} for t in targets]

        loss_dict = model(imgs,targets)

        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        batch_loss = loss.item()

        total_loss += batch_loss

        avg_loss = total_loss / batch_idx

        elapsed = time.time() - epoch_start

        batches_left = num_batches - batch_idx

        eta = (elapsed / batch_idx) * batches_left if batch_idx else 0

        # GPU memory
        gpu_info = ""

        if DEVICE == "cuda":

            mem = torch.cuda.memory_allocated() / 1024**3

            gpu_info = f" | GPU {mem:.2f}GB"

        percent = (batch_idx / num_batches) * 100

        if batch_idx % LOG_INTERVAL == 0 or batch_idx == num_batches:

            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Batch {batch_idx}/{num_batches} ({percent:.1f}%) | "
                f"Loss {batch_loss:.4f} | "
                f"Avg {avg_loss:.4f} | "
                f"ETA {timedelta(seconds=int(eta))}"
                f"{gpu_info}"
            )

    epoch_time = time.time() - epoch_start

    print(
        f"\nEpoch {epoch+1} finished | "
        f"Avg Loss {avg_loss:.4f} | "
        f"Time {timedelta(seconds=int(epoch_time))}\n"
    )


# ========================
# SAVE MODEL
# ========================

torch.save(model.state_dict(),"fire_detector.pt")

print("Model saved -> fire_detector.pt")