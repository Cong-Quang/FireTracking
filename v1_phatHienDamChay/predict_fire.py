import torch
import cv2
import torch.nn as nn
from torchvision import models, transforms

DEVICE="cuda"

model=models.resnet18()

model.fc=nn.Linear(model.fc.in_features,2)

model.load_state_dict(torch.load("fire_classifier.pt"))

model=model.to(DEVICE)
model.eval()

tf=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

classes=["non_fire","fire"]

img=cv2.imread("fire_dataset/fire_images/fire.1.png")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

x=tf(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():

    out=model(x)

    pred=out.argmax(1).item()

print("Prediction:",classes[pred])