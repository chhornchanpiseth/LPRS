import argparse
import json
import cv2
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from sklearn.preprocessing import LabelEncoder
from local_utils import reconstruct, ch_predict, wpod_pred
import numpy as np
import requests
from main_LPRS import LPRS, load_ch_model, load_model
import plate_operation
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
print("1")

wpod_url = load_model("model/wpod-net.json")
ch_url = load_ch_model(
    "model/MobileNets_character_recognition.json",
    "model/License_character_recognition_weight.h5",
)
# wpod_url = "http://192.168.0.50:8501/v1/models/wpod-net:predict"
# ch_url = "http://localhost:8504/v1/models/license_character_recog:predict"

labels = LabelEncoder()
labels.classes_ = np.load("model/license_character_classes.npy")
print("Label Classes ",labels.classes_)
# Argument parser for giving input image_path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path of the image")
args = vars(ap.parse_args())
image_path = args["image"]

frame = cv2.imread(image_path)
print("frame",frame)
start_plate = time.time()
try:
    LpImg, lp_type, cor = wpod_pred(wpod_url, frame, 0.55, "motorbike", Dmin=265)
except Exception as err:
    print(f"FPS(plate): {1/(time.time()-start_plate)}")
    print(err)
    exit()
print(f"FPS(plate): {1.0/(time.time()-start_plate)}")
total_LPRS = time.time()
lprs = LPRS(frame, LpImg[0], cor[0], lp_type, frame, debug=True)


# plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
# cv2.imwrite("lpimg.png", plate_image)
cropped_char_imgs = lprs.detect()
print("cropped_char_imgs" , cropped_char_imgs)
res_id = ""
print("total_LPRS: ", 1 / (time.time() - total_LPRS))
total_ch = time.time()
for char_img in cropped_char_imgs:
    start_ch = time.time()
    ch = ch_predict(ch_url, char_img, labels)
    print("ch", ch	)
    print("ch", time.time() - start_ch)
    res_id += ch
    # cv2.imwrite("characters/"+ch+datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')+".png", char_img)

print("RES _ID",res_id)
print("total", time.time() - total_ch)
lprs.res_id = res_id
time_org = time.time()
res_eng, res_kh = lprs.recog_kh_eng()
print("res_eng", res_eng)
print("res_kh", res_kh)
print("org:", time.time() - time_org)

if res_eng != "CAMBODIA":
    res_id = plate_operation.Operation(res_id).operation_plate().lstrip("E")
lprs.res_id = res_id
lprs.draw_box()
frame = lprs.vehicle_image
print(res_id, res_eng, res_kh)
print("org:", time.time() - time_org)
cv2.imwrite("output_images/img.png", frame)
print("org:", time.time() - time_org)
