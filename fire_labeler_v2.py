#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import csv
import time
from pathlib import Path
from tqdm import tqdm


# =========================
# CONFIG
# =========================

LOP_YOLO = 0

# Lowered area to catch small fires
DIEN_TICH_TOI_THIEU = 100

# HSV ranges
MAU_LUA_DUOI_LOW  = np.array([0,120,200])
MAU_LUA_DUOI_HIGH = np.array([12,255,255])

MAU_LUA_CAM_LOW   = np.array([15,150,220])
MAU_LUA_CAM_HIGH  = np.array([35,255,255])

MAU_LUA_TREN_LOW  = np.array([160,150,200])
MAU_LUA_TREN_HIGH = np.array([179,255,255])


# =========================
# PATH
# =========================

ROOT = Path(__file__).parent / "fire_dataset"

DIR_FIRE = ROOT / "fire_images"
DIR_NONFIRE = ROOT / "non_fire_images"

DIR_LABEL = ROOT / "labels"
DIR_MASK = ROOT / "masks"
DIR_VIS = ROOT / "visualized"

CSV_FILE = ROOT / "labels.csv"
REPORT_FILE = ROOT / "bao_cao_v2.txt"


# =========================
# UTIL
# =========================

def adaptive_kernels(h,w):

    s=min(h,w)

    k_open=max(3,int(s*0.005))
    if k_open%2==0:
        k_open+=1

    k_close=max(7,int(s*0.02))
    if k_close%2==0:
        k_close+=1

    return (k_open,k_open),(k_close,k_close)


def laplacian_variance(gray):
    return cv2.Laplacian(gray,cv2.CV_64F).var()


# =========================
# FIRE CHECK
# =========================

def kiem_tra_lua(img, contour, mask):
    x,y,w,h=cv2.boundingRect(contour)

    if w<5 or h<5:
        return False

    area=cv2.contourArea(contour)

    if area<DIEN_TICH_TOI_THIEU:
        return False

    roi=img[y:y+h,x:x+w]
    roi_mask=mask[y:y+h,x:x+w]

    pixels=roi[roi_mask>0]

    if len(pixels)==0:
        return False

    b_mean,g_mean,r_mean=np.mean(pixels,axis=0)
    b_var,g_var,r_var=np.var(pixels,axis=0)

    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

    s_vals=hsv[:,:,1][roi_mask>0]
    v_vals=hsv[:,:,2][roi_mask>0]

    s_mean=np.mean(s_vals)
    v_mean=np.mean(v_vals)
    v_max=np.max(v_vals)
    s_var=np.var(s_vals)
    v_var=np.var(v_vals)

    hull=cv2.convexHull(contour)
    hull_area=cv2.contourArea(hull)
    solidity=area/hull_area if hull_area>0 else 0

    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    texture=laplacian_variance(gray)

    # Basic color intensity check
    if r_mean<180:
        return False
    if r_mean<g_mean+25 or r_mean<b_mean+45:
        return False
    if s_mean<100 or v_mean<150:
        return False
    
    # Clothes and helmets are usually very uniform
    # Fire flickers, has smoke, dark spots and bright spots
    if r_var<10 and g_var<10:
        return False
        
    # V2: Fire value variance should be higher than a painted helmet
    if v_var < 50:
        return False

    # V2: Expect at least some pixels in the core to be bright white-hot
    if v_max < 240:
        return False

    if solidity<0.4:
        return False

    # V2: Bump texture check to filter out smooth safety helmets
    if texture<30:
        return False

    # V2: Another layer of texture / blur check
    # Let's verify the maximum edge gradient inside the flame boundary.
    # Fire often has a sharp edge against smoke or background.
    edges = cv2.Canny(gray, 100, 200)
    edges = edges[roi_mask>0]
    # At least some edge pixels inside or on the boundary
    if np.sum(edges) < 255: # meaning at least 1 edge pixel
        return False

    return True


# =========================
# FIRE DETECTION
# =========================

def phat_hien_lua(img):

    h,w=img.shape[:2]

    open_k,close_k=adaptive_kernels(h,w)

    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    h_ch,s_ch,v_ch=cv2.split(hsv)

    clahe=cv2.createCLAHE(3.0,(8,8))
    v_ch=clahe.apply(v_ch)

    hsv=cv2.merge([h_ch,s_ch,v_ch])

    mask1=cv2.inRange(hsv,MAU_LUA_DUOI_LOW,MAU_LUA_DUOI_HIGH)
    mask2=cv2.inRange(hsv,MAU_LUA_CAM_LOW,MAU_LUA_CAM_HIGH)
    mask3=cv2.inRange(hsv,MAU_LUA_TREN_LOW,MAU_LUA_TREN_HIGH)

    mask=mask1|mask2|mask3

    kernel_open=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,open_k)
    kernel_close=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,close_k)

    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel_open)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel_close)

    mask=cv2.dilate(mask,None,iterations=1)

    cnts=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=cnts[0] if len(cnts)==2 else cnts[1]

    final_mask=np.zeros_like(mask)

    good=[]

    for c in contours:

        if cv2.contourArea(c)<DIEN_TICH_TOI_THIEU:
            continue

        c_mask=np.zeros_like(mask)

        cv2.drawContours(c_mask,[c],-1,255,-1)

        if kiem_tra_lua(img,c,c_mask):

            good.append(c)

            cv2.drawContours(final_mask,[c],-1,255,-1)

    return final_mask,good


# =========================
# BOUNDING BOX
# =========================

def tao_box(contours,h,w):

    boxes=[]

    for c in contours:

        x,y,bw,bh=cv2.boundingRect(c)

        cx=x+bw//2
        cy=y+bh//2

        boxes.append({
            "x1":x,
            "y1":y,
            "x2":x+bw,
            "y2":y+bh,
            "cx":cx,
            "cy":cy,
            "w":bw,
            "h":bh,
            "area":bw*bh
        })

    return boxes


def yolo_labels(boxes,h,w):

    labels=[]

    for b in boxes:

        cx=b["cx"]/w
        cy=b["cy"]/h
        bw=b["w"]/w
        bh=b["h"]/h

        labels.append(f"{LOP_YOLO} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return labels


# =========================
# VISUALIZATION
# =========================

def draw_boxes(img,boxes):

    out=img.copy()

    h,w=img.shape[:2]

    for i,b in enumerate(boxes):

        x1,y1,x2,y2=b["x1"],b["y1"],b["x2"],b["y2"]

        cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,0),2)

        ratio=b["area"]/(h*w)*100

        txt=f"fire {ratio:.1f}%"

        cv2.putText(out,txt,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    return out


# =========================
# PROCESS ONE IMAGE
# =========================

def process_image(path,is_fire):

    name=path.name
    stem=path.stem

    img=cv2.imread(str(path))

    if img is None:
        return None

    h,w=img.shape[:2]

    mask,contours=phat_hien_lua(img)

    boxes=tao_box(contours,h,w)

    labels=yolo_labels(boxes,h,w)

    label_path=DIR_LABEL/f"{stem}.txt"

    with open(label_path,"w") as f:

        if labels:
            f.write("\n".join(labels))

    cv2.imwrite(str(DIR_MASK/f"{stem}.png"),mask)

    vis=draw_boxes(img,boxes) if boxes else img

    cv2.imwrite(str(DIR_VIS/f"{stem}.jpg"),vis)

    return {
        "file":name,
        "is_fire":is_fire,
        "detected":len(boxes)>0,
        "boxes":len(boxes),
        "w":w,
        "h":h
    }


# =========================
# DATASET PROCESS
# =========================

def process_dataset():

    DIR_LABEL.mkdir(exist_ok=True)
    DIR_MASK.mkdir(exist_ok=True)
    DIR_VIS.mkdir(exist_ok=True)

    results=[]

    fire=list(DIR_FIRE.glob("*.*"))

    for p in tqdm(fire,desc="fire_images"):
        r=process_image(p,True)
        if r: results.append(r)

    non=list(DIR_NONFIRE.glob("*.*"))

    for p in tqdm(non,desc="non_fire_images"):
        r=process_image(p,False)
        if r: results.append(r)

    return results


# =========================
# REPORT
# =========================

def report(results):

    fire=[r for r in results if r["is_fire"]]
    non=[r for r in results if not r["is_fire"]]

    tp=sum(1 for r in fire if r["detected"])
    fn=sum(1 for r in fire if not r["detected"])
    fp=sum(1 for r in non if r["detected"])
    tn=sum(1 for r in non if not r["detected"])

    precision=tp/(tp+fp) if tp+fp>0 else 0
    recall=tp/(tp+fn) if tp+fn>0 else 0
    f1=2*precision*recall/(precision+recall) if precision+recall>0 else 0

    acc=(tp+tn)/len(results)

    txt=f"""
==================================
RESULT V2
==================================
Total: {len(results)}

TP: {tp}
FP: {fp}
FN: {fn}
TN: {tn}

Accuracy: {acc*100:.2f}%
Precision: {precision*100:.2f}%
Recall: {recall*100:.2f}%
F1: {f1*100:.2f}%
==================================
"""

    print(txt)

    with open(REPORT_FILE,"w") as f:
        f.write(txt)


# =========================
# MAIN
# =========================

def main():

    start=time.time()

    results=process_dataset()

    report(results)

    print("Time:",time.time()-start)


if __name__=="__main__":
    main()
