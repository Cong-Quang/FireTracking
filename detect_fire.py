import torch
import cv2
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

DEVICE="cuda"

model=fasterrcnn_resnet50_fpn()

in_features=model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor=torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features,
    2
)

model.load_state_dict(torch.load("fire_detector.pt"))

model=model.to(DEVICE)
model.eval()


img=cv2.imread("fire_dataset/fire_images/fire.67.png")

rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

tensor=torch.from_numpy(rgb/255.).permute(2,0,1).float().unsqueeze(0).to(DEVICE)

with torch.no_grad():

    pred=model(tensor)[0]


boxes=pred["boxes"].cpu().numpy()
scores=pred["scores"].cpu().numpy()

for box,score in zip(boxes,scores):

    if score<0.1:
        continue

    x1,y1,x2,y2=box.astype(int)

    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.putText(img,"fire",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

cv2.imshow("result",img)
cv2.waitKey(0)