# plot_one.py
import os, ast
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from DataPostprocessing.utils import get_mask_from_RLE
import cv2
import csv  

# 1) Load the CSV & jpg‑ID list
df = pd.read_csv("Annotations/MIMIC-CXR-JPG.csv")
with open("jpg_filenames_only.txt") as f:
    jpg_ids = {os.path.splitext(line.strip())[0] for line in f if line.strip()}

# 2) Build mask and take first 1000 matches
mask   = df["dicom_id"].isin(jpg_ids)
matched = df.loc[mask].iloc[:300]
matched.to_csv("MIMIC-CXR_matched_first300.csv", index=False)
if matched.empty:
    raise RuntimeError("No matching DICOM IDs found!")

import os, ast, cv2, numpy as np, matplotlib.pyplot as plt
print("check")
# build your dicom→path index once
image_index = {}
for root, _, files in os.walk("../MIMIC_CXR"):
    for fname in files:
        if fname.lower().endswith(".jpg"):
            key = os.path.splitext(fname)[0]
            image_index[key] = os.path.join(root, fname)

print("check0")
# where to save your ready-to-train masks
OUT_HEART = "processed_masks/heart_ctr"
OUT_LUNGS = "processed_masks/lungs_ctr"
os.makedirs(OUT_HEART, exist_ok=True)
os.makedirs(OUT_LUNGS, exist_ok=True)
# path to your text list
list_fp = "processed_masks/dicom_list.csv"
print("check1")
def to_contour(landmarks):
    return landmarks.astype(np.int32).reshape(-1,1,2)

with open(list_fp, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["dicom_id"])
    
    for _, ex in matched.iterrows():
        dicom_id = str(ex["dicom_id"])
        H_df, W_df = ex["Height"], ex["Width"]
        print("check2")
        # parse landmarks
        try:
            lm = ast.literal_eval(ex["Landmarks"])
        except:
            s = (ex["Landmarks"]
                 .replace('[ ', '[')
                 .replace('\n ', ',')
                 .replace('  ', ',')
                 .replace(' ', ','))
            try:
                lm = ast.literal_eval(s)
            except:
                continue
        lm = np.array(lm).reshape(-1,2)
        RL, LL, Ht = lm[:44], lm[44:94], lm[94:]
        print("check3")
        # build full‑res masks (size from DF)
        mask_heart = np.zeros((H_df, W_df), dtype=np.uint8)
        cv2.fillPoly(mask_heart, [to_contour(Ht)], color=1)
    
        mask_lungs = np.zeros((H_df, W_df), dtype=np.uint8)
        cv2.fillPoly(mask_lungs, [to_contour(RL), to_contour(LL)], color=1)
    
        # # save them if you like
        # cv2.imwrite(f"masks/heart/{dicom_id}_heart.png", mask_heart*255)
        # cv2.imwrite(f"masks/lungs/{dicom_id}_lungs.png", mask_lungs*255)
    
        # --- NOW load the jpg and resize masks to match it ---
        img_path = image_index.get(dicom_id)
        if img_path is None:
            continue
    
        # load your pre‐scaled jpg (512×512)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(float)/255.0
        h_img, w_img = img.shape
    
        # resize masks down to match the jpg
        mask_heart_s = cv2.resize(mask_heart,
                                  (w_img, h_img),
                                  interpolation=cv2.INTER_NEAREST)
        mask_lungs_s = cv2.resize(mask_lungs,
                                  (w_img, h_img),
                                  interpolation=cv2.INTER_NEAREST)
        print("check4")
        # # build overlay
        # over = np.zeros((h_img, w_img, 3), dtype=float)
        # over[:,:,0] = img + 0.3*mask_lungs_s - 0.1*mask_heart_s
        # over[:,:,1] = img + 0.3*mask_heart_s - 0.1*mask_lungs_s
        # over[:,:,2] = img
        # over = np.clip(over, 0, 1)
    
        # # draw dots (also scale landmarks to the jpg size!)
        # sx, sy = w_img/W_df, h_img/H_df
        # RL_s = RL * [sx, sy]; LL_s = LL * [sx, sy]; Ht_s = Ht * [sx, sy]
    
        # for x,y in np.vstack([RL_s, LL_s]):
        #     cv2.circle(over, (int(x),int(y)), 5, (1,0,1), -1)
        # for x,y in Ht_s:
        #     cv2.circle(over, (int(x),int(y)), 5, (1,1,0), -1)
    
        # # show
        # plt.figure(figsize=(6,6))
        # plt.imshow(over)
        # plt.title(f"DICOM {dicom_id}")
        # plt.axis('off')
        # plt.show()
    
        # 5) save out as binary PNGs (0/255)
        out_h = os.path.join(OUT_HEART, f"{dicom_id}_heart.png")
        out_l = os.path.join(OUT_LUNGS, f"{dicom_id}_lungs.png")
        ok_h = cv2.imwrite(out_h, mask_heart_s * 255)
        ok_l = cv2.imwrite(out_l, mask_lungs_s * 255)
    
        if ok_h and ok_l:
            # only write the dicom_id
            writer.writerow([dicom_id])
        else:
            print(f"❌ Failed to save masks for {dicom_id}")
        
    
        print(f"Saved resized masks →\n    {out_h}\n    {out_l}")
