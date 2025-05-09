# # plot_one.py
# import os, ast
# import pandas as pd
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from DataPostprocessing.utils import get_mask_from_RLE
# import cv2
# import csv  

# import os, ast, csv
# import pandas as pd
# import numpy as np
# import cv2

# # ========== SETTINGS ==========
# IMG_FOLDER = "filtered_images"   
# ANNOT_CSV  = "Annotations/MIMIC-CXR-JPG.csv"
# OUT_HEART  = "processed_masks_cardiomegaly/heart"
# OUT_LUNGS  = "processed_masks_cardiomegaly/lungs"

# os.makedirs(OUT_HEART, exist_ok=True)
# os.makedirs(OUT_LUNGS, exist_ok=True)

# # ========== Load Annotations ==========
# df = pd.read_csv(ANNOT_CSV).set_index("dicom_id")

# # ========== Utility ==========
# def to_contour(landmarks: np.ndarray):
#     return landmarks.astype(np.int32).reshape(-1, 1, 2)

# # ========== Process Each JPEG ==========
# for fname in os.listdir(IMG_FOLDER):
#     if not fname.lower().endswith(".jpg"):
#         continue

#     dicom_id = os.path.splitext(fname)[0]
#     if dicom_id not in df.index:
#         print(f"⚠️  {dicom_id} not in annotations → skipping")
#         continue

#     ex = df.loc[dicom_id]
#     # parse landmarks string → numpy array (N,2)
#     lm_str = ex["Landmarks"]
#     try:
#         lm = np.array(ast.literal_eval(lm_str)).reshape(-1,2)
#     except:
#         s = (lm_str.replace('[ ', '[')
#                    .replace('\n ', ',')
#                    .replace('  ', ',')
#                    .replace(' ', ','))
#         lm = np.array(ast.literal_eval(s)).reshape(-1,2)

#     # split into right lung / left lung / heart
#     RL, LL, Ht = lm[:44], lm[44:94], lm[94:]

#     # build full‐res masks (Height,Width from CSV)
#     H_df, W_df = int(ex["Height"]), int(ex["Width"])
#     mask_heart = np.zeros((H_df, W_df), dtype=np.uint8)
#     mask_lungs = np.zeros((H_df, W_df), dtype=np.uint8)

#     cv2.fillPoly(mask_heart, [to_contour(Ht)], color=1)
#     cv2.fillPoly(mask_lungs, [to_contour(RL), to_contour(LL)], color=1)

#     # load the actual image to get its shape
#     img_path = os.path.join(IMG_FOLDER, fname)
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         print(f"❌  failed to load {img_path}")
#         continue
#     h_img, w_img = img.shape

#     # resize masks down/up to image size
#     mask_heart_resized = cv2.resize(mask_heart, (w_img, h_img),
#                                      interpolation=cv2.INTER_NEAREST)
#     mask_lungs_resized = cv2.resize(mask_lungs, (w_img, h_img),
#                                      interpolation=cv2.INTER_NEAREST)

#     # save binary PNGs (0 or 255)
#     out_h = os.path.join(OUT_HEART, f"{dicom_id}_heart.png")
#     out_l = os.path.join(OUT_LUNGS, f"{dicom_id}_lungs.png")
#     ok1 = cv2.imwrite(out_h, mask_heart_resized * 255)
#     ok2 = cv2.imwrite(out_l, mask_lungs_resized * 255)
#     if ok1 and ok2:
#         print(f"✅  saved masks for {dicom_id}")
#     else:
#         print(f"❌  failed to save masks for {dicom_id}")
# plot_one.py
# plot_one.py
import os, ast
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from DataPostprocessing.utils import get_mask_from_RLE
import cv2
import csv  

import os, ast, csv
import pandas as pd
import numpy as np
import cv2

# ========== SETTINGS ==========
IMG_FOLDER = "filtered_images"   
ANNOT_CSV  = "Annotations/MIMIC-CXR-JPG.csv"
OUT_HEART  = "processed_masks_cardiomegaly/heart"
OUT_LUNGS  = "processed_masks_cardiomegaly/lungs"

os.makedirs(OUT_HEART, exist_ok=True)
os.makedirs(OUT_LUNGS, exist_ok=True)

# ========== Load Annotations ==========
df = pd.read_csv(ANNOT_CSV).set_index("dicom_id")

# ========== Utility ==========
def to_contour(landmarks: np.ndarray):
    return landmarks.astype(np.int32).reshape(-1, 1, 2)

# ========== Process Each JPEG ==========
for fname in os.listdir(IMG_FOLDER):
    if not fname.lower().endswith(".jpg"):
        continue

    dicom_id = os.path.splitext(fname)[0]
    if dicom_id not in df.index:
        print(f"⚠️  {dicom_id} not in annotations → skipping")
        continue

    ex = df.loc[dicom_id]
    lm_str = ex["Landmarks"]

    try:
        # First attempt direct parse
        lm = np.array(ast.literal_eval(lm_str)).reshape(-1, 2)
    except:
        try:
            # Try to clean and parse
            s = (lm_str.replace('[ ', '[')
                       .replace('\n ', ',')
                       .replace('  ', ',')
                       .replace(' ', ',')
                       .replace(',,', ',0,'))  # Fill missing values with 0
            lm = np.array(ast.literal_eval(s)).reshape(-1, 2)
        except:
            print(f"❌  malformed landmarks for {dicom_id} → skipping")
            continue

    if lm.shape[0] < 100:
        print(f"⚠️  too few landmarks for {dicom_id} → skipping")
        continue

    # split into right lung / left lung / heart
    RL, LL, Ht = lm[:44], lm[44:94], lm[94:]

    # build full‐res masks (Height,Width from CSV)
    H_df, W_df = int(ex["Height"]), int(ex["Width"])
    mask_heart = np.zeros((H_df, W_df), dtype=np.uint8)
    mask_lungs = np.zeros((H_df, W_df), dtype=np.uint8)

    cv2.fillPoly(mask_heart, [to_contour(Ht)], color=1)
    cv2.fillPoly(mask_lungs, [to_contour(RL), to_contour(LL)], color=1)

    # load the actual image to get its shape
    img_path = os.path.join(IMG_FOLDER, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌  failed to load {img_path}")
        continue
    h_img, w_img = img.shape

    # resize masks down/up to image size
    mask_heart_resized = cv2.resize(mask_heart, (w_img, h_img),
                                    interpolation=cv2.INTER_NEAREST)
    mask_lungs_resized = cv2.resize(mask_lungs, (w_img, h_img),
                                    interpolation=cv2.INTER_NEAREST)

    # save binary PNGs (0 or 255)
    out_h = os.path.join(OUT_HEART, f"{dicom_id}_heart.png")
    out_l = os.path.join(OUT_LUNGS, f"{dicom_id}_lungs.png")
    ok1 = cv2.imwrite(out_h, mask_heart_resized * 255)
    ok2 = cv2.imwrite(out_l, mask_lungs_resized * 255)
    if ok1 and ok2:
        print(f"✅  saved masks for {dicom_id}")
    else:
        print(f"❌  failed to save masks for {dicom_id}")
