# -*- coding: utf-8 -*-
import os
from os.path import splitext
import subprocess
from scipy.spatial import distance as dist
from collections import OrderedDict, defaultdict, Counter
import numpy as np

from tensorflow.keras.models import model_from_json
import cv2
import pytesseract
from PIL import Image, ImageFont, ImageDraw
import scipy.ndimage
import plate_operation
from local_utils import wpod_pred, ch_predict
import easyocr
import difflib
import re
import requests

# generate khmer data
from datetime import datetime
import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    filename=f"./logs/example.log",
    filemode="w",
)
logger = logging.getLogger(__name__)


def map_kh_vehicle_types(vehicle_type):
    return {"motorbike": "ម៉ូតូ", "car": "ឡាន", "bus": "ឡានក្រុង"}[vehicle_type]


class CentroidTracker:
    def __init__(self, maxDisappeared=3):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.all_ids = defaultdict(Counter)
        # all organization names results in english
        self.all_engs = defaultdict(Counter)
        self.all_khs = defaultdict(Counter)
        self.all_centroids = defaultdict(list)
        # first image detected
        self.vehicles = defaultdict(list)
        self.timestamps = defaultdict(str)
        # last image detected, (from centroid tracking,
        # can be empty if not seen again)
        self.last_vehicle_frame = defaultdict(list)
        self.all_vehicle_types = defaultdict(str)
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared
        self.direction_in = "left"

    def delete(self, delete_id):
        if delete_id in self.objects:
            del self.objects[delete_id]
            del self.disappeared[delete_id]
            del self.all_ids[delete_id]
            del self.all_engs[delete_id]
            del self.all_khs[delete_id]
            del self.all_centroids[delete_id]
            if delete_id in self.vehicles:
                del self.vehicles[delete_id]
            del self.timestamps[delete_id]
            if delete_id in self.all_vehicle_types:
                del self.all_vehicle_types[delete_id]
            if delete_id in self.last_vehicle_frame:
                del self.last_vehicle_frame[delete_id]

    def register(
        self, centroid, res_id, res_eng, res_kh, vehicle_frame, vehicle_type, timestamp
    ):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.all_ids[self.nextObjectID][res_id] += 1
        self.all_engs[self.nextObjectID][res_eng] += 1
        self.all_khs[self.nextObjectID][res_kh] += 1
        self.all_centroids[self.nextObjectID].append(centroid)
        # if vehicle_frame is not None:
        self.vehicles[self.nextObjectID] = vehicle_frame
        self.timestamps[self.nextObjectID] = timestamp
        # if vehicle_type:
        self.all_vehicle_types[self.nextObjectID] = vehicle_type
        self.nextObjectID += 1

        if self.nextObjectID == 8:
            self.nextObjectID = 0

    def deregister(self, objectID, delete=True, force_send=False):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        # send to Backend here

        counter_id = self.all_ids[objectID]
        counter_eng = self.all_engs[objectID]
        counter_kh = self.all_khs[objectID]
        logger.info(counter_id)
        logger.info(counter_eng)
        logger.info(counter_kh)
        top_id = counter_id.most_common(1)
        top_kh = counter_kh.most_common(1)
        top_eng = counter_eng.most_common(1)
        vehicle_type = self.all_vehicle_types[objectID]
        # timeNow = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        timeNow = self.timestamps[objectID]
        #         if len(counter_id) >= 1 and counter_id.most_common(1)[0][1] >= 2:
        # if len of characters >= 4 and
        # (there are 4 centroids deteced or most frequent id > 2)
        # if len(top_id[0][0]) >= 4 and (
        #     len(self.all_centroids[objectID]) >= 4 or top_id[0][1] >= 2
        # ):
        # if 1:
        logger.info("[DEREGISTER ID]")
        logger.info(top_id[0])
        logger.info(top_eng[0])
        logger.info(top_kh[0])
        logger.info(vehicle_type)
        status = ""
        saved_image = None

        if self.direction_in == "right":
            if self.all_centroids[objectID][0][0] < self.all_centroids[objectID][-1][0]:
                status = "In"
                saved_image = self.last_vehicle_frame[objectID]

            else:
                status = "Out"
                saved_image = self.vehicles[objectID]
            # if vehicle_type == 'motorbike':
            #     status = 'In'
        # car bus truck is only for current whiteshop camera for cars, delete if implement somewhere else
        # or vehicle_type in ('car', 'bus', 'truck'):
        elif self.direction_in == "left":
            if self.all_centroids[objectID][0][0] > self.all_centroids[objectID][-1][0]:
                status = "In"
                saved_image = self.vehicles[objectID]
            else:
                status = "Out"
                saved_image = self.last_vehicle_frame[objectID]
            # if vehicle_type == 'motorbike':
            #     status = 'Out'
        elif self.direction_in == "down":
            if self.all_centroids[objectID][0][1] < self.all_centroids[objectID][-1][1]:
                status = "In"
                saved_image = self.vehicles[objectID]
            else:
                status = "Out"
                saved_image = self.last_vehicle_frame[objectID]
            if vehicle_type == "motorbike":
                status = "Out"
                saved_image = self.last_vehicle_frame[objectID]
        elif self.direction_in == "up":
            if self.all_centroids[objectID][0][1] < self.all_centroids[objectID][-1][1]:
                status = "Out"
                saved_image = self.vehicles[objectID]
            else:
                status = "In"
                saved_image = self.last_vehicle_frame[objectID]
            if vehicle_type == "motorbike":
                status = "In"
                saved_image = self.last_vehicle_frame[objectID]
        # logger.debug(self.vehicles[objectID])
        # logger.debug(self.last_vehicle_frame[objectID])
        # logger.debug(saved_image)
        # self.vehicles[objectID] is supposed to always have images (but sometimes also no)
        saved_image = saved_image if len(saved_image) else self.vehicles[objectID]
        # logger.debug(saved_image)
        # logger.debug(self.vehicles[objectID])
        # saved_image = self.vehicles[objectID]
        if (len(saved_image) and len(self.all_centroids[objectID]) > 5) or force_send:
            api_res = sendData(
                saved_image,
                top_eng[0][0],
                top_kh[0][0],
                top_id[0][0],
                timeNow,
                map_kh_vehicle_types(vehicle_type),
                status,
                self.location,
            )
            print("length of api res:", len(api_res), flush=True)
        else:
            print("Not sent to api")
            # api_res = sendData(
            #     self.vehicles[objectID],
            #     top_eng[0][0],
            #     top_kh[0][0]
            #     top_id[0][0],
            #     timeNow,
            #     map_kh_vehicle_types(vehicle_type),
            #     status,
            #     self.location,
            # )

        logger.info(self.all_centroids[objectID])

        if delete:
            self.delete(objectID)

    def update(
        self,
        rects,
        input_ids,
        input_engs,
        input_khs,
        vehicle_images,
        vehicle_types,
        timestamp,
    ):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        # loop over the bounding box rectangles
        for (i, (cX, cY)) in enumerate(rects):
            inputCentroids[i] = (cX, cY)
        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(
                    inputCentroids[i],
                    input_ids[i],
                    input_engs[i],
                    input_khs[i],
                    vehicle_images[i],
                    vehicle_types[i],
                    timestamp,
                )
        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid

            # logger.info()
            # logger.info("centroids", objectCentroids, inputCentroids)

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]
            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined

            # logger.info(D)
            # logger.info(rows)
            # logger.info(cols)
            # logger.info(inputCentroids)
            # logger.info(self.objects)

            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, and distance > 300, ignore it

                # logger.info(D[row][col])

                if row in usedRows or col in usedCols or D[row][col] > 300:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.last_vehicle_frame[objectID] = vehicle_images[col]
                self.all_centroids[objectID].append(inputCentroids[col])
                # if len(input_ids[col]) >= 6:
                if input_ids[col] != "No Plate":
                    self.all_ids[objectID][input_ids[col]] += 1
                if input_engs[col]:
                    self.all_engs[objectID][input_engs[col]] += 1
                if input_khs[col]:
                    self.all_khs[objectID][input_khs[col]] += 1
                # Deregister and send result when confident enough
                if self.all_ids[objectID][input_ids[col]] == 3:
                    self.deregister(objectID, delete=False, force_send=True)

                self.disappeared[objectID] = 0
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            # logger.info(D)
            # logger.info("D shape", D.shape[0], D.shape[1])
            # logger.info(usedRows, usedCols)
            # logger.info(unusedRows, unusedCols)
            # logger.info()
            """this *if* does not work because we set maximum distance,
             so some rows/cols are not used
            if there is unusedRows, there is an plate not detected in the frame
            if there is unusedCols, there is a new plate
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
            """
            for row in unusedRows:
                # grab the object ID for the corresponding row
                # index and increment the disappeared counter
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                # check to see if the number of consecutive
                # frames the object has been marked "disappeared"
                # for warrants deregistering the object
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we
            #  need to
            # register each new input centroid as a trackable object
            # else:
            for col in unusedCols:
                self.register(
                    inputCentroids[col],
                    input_ids[col],
                    input_engs[col],
                    input_khs[col],
                    vehicle_images[col],
                    vehicle_types[col],
                    timestamp,
                )
        # return the set of trackable objects
        return self.objects


class LPRS:
    def __init__(self, frame, LpImg, cor, lp_type, vehicle_image=None, debug=False):
        self.frame = frame
        self.vehicle_image = vehicle_image
        self.LpImg = LpImg
        self.cor = cor
        self.lp_type = lp_type
        self.res_id = None
        self.plate_image = None
        # plate_image_binary is for visualization purpose only
        self.plate_image_binary = None
        self.plate_height, self.plate_width, _ = self.LpImg.shape
        self.x_min_kh = self.y_min_kh = self.x_min_eng = self.y_min_eng = 3000
        self.y_max_kh = self.x_max_kh = self.y_max_eng = self.x_max_eng = 0
        self.debug = debug

    def remove_shadow(self, img):
        removed_shadow = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result_planes = []
        for plane in cv2.split(removed_shadow):
            # the value will darken
            dilated_img = cv2.dilate(plane, np.ones((9, 9), np.uint8))
            # bg_img = cv2.blur(dilated_img, (8,8))
            bg_img = cv2.medianBlur(dilated_img, 27)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            # norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            # result_norm_planes.append(norm_img)
        return cv2.merge(result_planes)

    def clear_edge(self):
        rows, cols = self.plate_image_binary.shape
        h_projection = np.array([x / rows for x in self.plate_image_binary.sum(axis=0)])
        # threshold = (np.max(h_projection) - np.min(h_projection)) / 4
        threshold = 14
        black_areas = np.where(h_projection < threshold)
        for j in black_areas:
            self.plate_image_binary[:, j] = 0

        v_projection = np.array([x / cols for x in self.plate_image_binary.sum(axis=1)])
        threshold = (np.max(v_projection) - np.min(v_projection)) / 4 - 5
        black_areas = np.where(v_projection < threshold)
        for j in black_areas:
            self.plate_image_binary[j, :] = 0
        self.plate_image_binary = scipy.ndimage.morphology.binary_erosion(
            self.plate_image_binary, structure=np.ones((2, 2))
        ).astype(self.plate_image_binary.dtype)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.plate_image_binary = cv2.morphologyEx(
            self.plate_image_binary, cv2.MORPH_DILATE, kernel3
        )

    def gray_blur_binary(self, frame, ID=False):
        # convert to grayscale and blur the image
        # Applied inversed thresh_binary
        if ID is True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            # If pixel value is greater than 20, it is assigned white(255) otherwise black
            binary = cv2.threshold(
                blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )[1]
            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        else:
            #         frame =  scale_img(frame, 120)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            binary = cv2.threshold(blur, 255, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[
                1
            ]
            thre_mor = None
        return binary, thre_mor

    def sort_contours(self, cnts, reverse=False):
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        try:
            (cnts, boundingBoxes) = zip(
                *sorted(
                    zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse
                )
            )
        except Exception as e:
            logger.info(e)

        return cnts

    def detect(self):
        self.plate_image = cv2.convertScaleAbs(self.LpImg, alpha=(255.0))
        # remove shadow from self.plate_image
        removed_shadow = self.remove_shadow(self.plate_image)
        if self.debug:
            cv2.imwrite("removed_shadow_plate.png", removed_shadow)
        binary, _ = self.gray_blur_binary(removed_shadow, True)
        # make bottom right black because sometimes there is white border covering the plate
        # so there is no contour even though there are characters
        if self.lp_type == 2:
            binary[-30:, :] = np.zeros((30, self.plate_width))
            binary[:7, :] = np.zeros((7, self.plate_width))
            np5 = np.zeros((self.plate_height, 5))
            binary[:, :5] = np5
            binary[:, -5:] = np5
            np50 = np.zeros((50, 50))
            binary[0:50, 0:50] = np50
            binary[0:50:, -50:] = np50

        else:
            binary[-17:, :] = np.zeros((17, self.plate_width))
            binary[:15, :] = np.zeros((15, self.plate_width))
            np10 = np.zeros((self.plate_height, 10))
            binary[:, :10] = np10
            binary[:, -10:] = np10
        self.plate_image_binary = binary

        self.clear_edge()

        cont, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # define standard width and height of character
        digit_w, digit_h = 30, 60
        crop_characters = []

        self.plate_image_binary = cv2.cvtColor(
            self.plate_image_binary, cv2.COLOR_GRAY2RGB
        )
        if self.lp_type == 1:
            for c in self.sort_contours(cont):
                (x, y, w, h) = cv2.boundingRect(c)
                ratio = h / w
                # Only select contour with defined ratio
                # Select contour which has the height larger than 35% of the plate
                # only select contour with width xxx compared to plate
                # 7 might not be a good number here
                if (
                    1.15 <= ratio <= 7
                    and 0.95 >= h / self.plate_height >= 0.39
                    and 0.15 >= w / self.plate_width >= 0.02
                    and (
                        x / self.plate_width >= 0.2
                        or (x / self.plate_width < 0.2 and 0.5 <= h / self.plate_height)
                    )
                ):
                    # Sperate number and gibe prediction
                    curr_num = binary[y : y + h, x : x + w]
                    # Draw bounding box arroung digit number
                    if self.debug:
                        cv2.rectangle(
                            self.plate_image_binary,
                            (x, y),
                            (x + w, y + h),
                            (0, 255, 0),
                            1,
                        )
                    # curr_num = thre_mor[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(
                        curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    crop_characters.append(curr_num)
                elif (
                    0.01 <= x / self.plate_width
                    and (x + w) / self.plate_width <= 0.35
                    and 0.12 <= y / self.plate_height
                    and (y + h) / self.plate_height <= 0.70
                ):
                    #                         cv2.rectangle(plate_image, (x, y), (x + w, y + h), (0, 0, 0), 1)
                    self.y_min_kh = min(self.y_min_kh, y)
                    self.y_max_kh = max(self.y_max_kh, y + h)
                    self.x_min_kh = min(self.x_min_kh, x)
                    self.x_max_kh = max(self.x_max_kh, x + w)
                elif (
                    y / self.plate_height >= 0.65
                    and (x + w) / self.plate_width <= 0.39
                    and self.plate_height - (y + h) > 2
                ):
                    #             cv2.rectangle(plate_image, (x, y), (x + w, y + h), (0, 0, 0), 1)
                    self.y_min_eng = min(self.y_min_eng, y)
                    self.y_max_eng = max(self.y_max_eng, y + h)
                    self.x_min_eng = min(self.x_min_eng, x)
                    self.x_max_eng = max(self.x_max_eng, x + w)
        # short plate
        elif self.lp_type == 2:
            self.y_min_num = []
            self.y_max_num = []
            for c in self.sort_contours(cont):
                (x, y, w, h) = cv2.boundingRect(c)
                ratio = h / w
                #         cv2.rectangle(plate_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 7 might not be a good number here
                if (
                    1 <= ratio <= 7.5
                    and 0.7 >= h / self.plate_height >= 0.30
                    and 0.22 >= w / self.plate_width >= 0.045
                    and y / self.plate_height > 0.25
                ):
                    #                         and 0.95 >= (y+h)/self.plate_height:
                    # Sperate number and gibe prediction
                    curr_num = binary[y : y + h, x : x + w]
                    # Draw bounding box arroung digit number
                    if self.debug:
                        cv2.rectangle(
                            self.plate_image_binary,
                            (x, y),
                            (x + w, y + h),
                            (0, 255, 0),
                            1,
                        )

                    #                         curr_num = thre_mor[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(
                        curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    crop_characters.append(curr_num)
                    self.y_min_num.append(y)
                    self.y_max_num.append(y + h)
        return crop_characters

    # need to assign self.res_id before this
    def recog_kh_eng(self):
        if self.lp_type == 1:
            # max because need to make sure it's >= 0
            # Extend the range by a few pixels
            self.y_min_kh = max(0, self.y_min_kh - 5)
            self.x_min_kh = max(0, self.x_min_kh - 5)
            self.y_min_eng = max(0, self.y_min_eng - 5)
            self.x_min_eng = max(0, self.x_min_eng - 5)
            self.y_max_kh += 5
            self.x_max_kh += 5
            self.y_max_eng += 5
            self.x_max_eng += 5
        # if lp_type == 2 and have results (because short plate depends on the id result)
        elif self.res_id:
            # get average of ymin ymax of num, then use it to get ENG/KH province
            self.y_min_num = sum(self.y_min_num) // len(self.y_min_num)
            self.y_max_num = sum(self.y_max_num) // len(self.y_max_num)
            self.x_min_kh = self.x_min_eng = int(self.plate_width * 0.2)
            self.x_max_kh = self.x_max_eng = int(self.plate_width * 0.8)
            self.y_max_kh = self.y_min_num - 3
            self.y_min_eng = self.y_max_num + 4
            self.y_max_eng = self.plate_height
            self.y_min_kh = 2
        organization_name = organization_name_khmer = ""
        # check if x_max_eng variable changed, it is detected
        if self.x_max_kh > 5:
            # save kh pic
            binary_kh, _ = self.gray_blur_binary(
                self.remove_shadow(
                    self.plate_image[
                        self.y_min_kh : self.y_max_kh, self.x_min_kh : self.x_max_kh
                    ]
                )
            )
            #             random_string = "".join(random.choices(
            #                 string.ascii_uppercase + string.digits, k=6))

            # recognize kh
            #             cropped_kh = Image.open("cropped_kh.png")
            text_kh = pytesseract.image_to_string(
                binary_kh, lang="khm", config="--oem 3 -l khm --psm 7"
            ).strip()
            organization_name_khmer = self.match_org(text_kh, "kh")
            logger.info(text_kh + " => " + organization_name_khmer)

            #             cv2.imwrite("khmer/"+organization_name_khmer +
            #                         "_"+datetime.now().strftime('%Y-%m-%dT%H:%M:%S')+".png", binary_kh)
            #             cv2.imwrite("cropped_kh.png", plate_image[y_min_kh-3:self.y_max_kh+3, self.x_min_kh-3:self.x_max_kh+3])
            # draw kh box on plate img
            if self.debug:
                cv2.rectangle(
                    self.plate_image_binary,
                    (self.x_min_kh, self.y_min_kh),
                    (self.x_max_kh, self.y_max_kh),
                    (255, 100, 100),
                    2,
                )

                # show kh on frame
                font = ImageFont.truetype("./utils/KhmerMN.ttc", 50)
                img_pil = Image.fromarray(self.frame)
                draw = ImageDraw.Draw(img_pil)
                draw.text(
                    (self.plate_width + 10, 5),
                    organization_name_khmer,
                    font=font,
                    fill=(0, 255, 0, 255),
                )
                self.frame = np.array(img_pil)

            """
        # check if x_max_eng variable changed, it is detected
        if self.x_max_eng > 3:
            
            # save eng pic
            binary, _ = self.gray_blur_binary(self.scale_img(
                self.plate_image[self.y_min_eng: self.y_max_eng, self.x_min_eng: self.x_max_eng], 125))
#             cv2.imwrite("cropped_eng.png", binary)
            # cv2.imwrite("cropped_eng.png", plate_image[self.y_min_eng-3: self.y_max_eng+3, self.x_min_eng-3: self.x_max_eng+3])
            cv2.rectangle(self.plate_image, (self.x_min_eng, self.y_min_eng),
                          (self.x_max_eng, self.y_max_eng), (0, 0, 255), 2)

            # recognize eng
#             cropped_eng = Image.open("cropped_eng.png")
            text_eng = pytesseract.image_to_string(
                binary, lang='eng', config='--psm 7 --oem 3').strip().upper()
            """
            organization_name = map_En(organization_name_khmer)
            if self.debug:
                cv2.putText(
                    self.frame,
                    organization_name,
                    (10, self.plate_height + 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (36, 255, 12),
                    2,
                )
        if self.debug:
            self.plate_image_binary = cv2.convertScaleAbs(
                self.plate_image_binary, alpha=(255.0)
            )
            cv2.imwrite("plate.png", self.plate_image_binary)
            self.frame[
                15 : self.plate_height + 15, 0 : self.plate_width
            ] = self.plate_image_binary

        return organization_name, organization_name_khmer

    @staticmethod
    def scale_img(img, scale_percent):
        # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized

    # have frame, cor, res_id before calling this function

    def draw_box(self, thickness=2):
        # if vehicle image exists, then put it in the vehicle image
        # else, don't draw in the frame
        if self.vehicle_image is not None:
            pts = []
            x_coordinates = self.cor[0]
            y_coordinates = self.cor[1]
            # clockwise
            # store the top-left, top-right, bottom-right, bottom-left
            # of the plate license respectively --maybe
            for i in range(4):
                pts.append([int(x_coordinates[i]), int(y_coordinates[i])])

            pts = np.array(pts, np.int32)
            pts = pts.reshape((-1, 1, 2))
            #             logger.info("[DEBUG]:", x_coordinates)
            #             logger.info("[DEBUG]:", y_coordinates)
            # x and y coordinates for displaying ID
            # x_text = int(round(x_coordinates[0]))
            # y_text = int(round(y_coordinates[0]))
            #             cv2.putText(self.frame, self.res_id, (self.plate_width+10, 100),
            #                                         cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)
            cv2.polylines(self.vehicle_image, [pts], True, (0, 255, 0), thickness)

    #             cv2.putText(self.frame, self.res_id, (self.plate_width+10,
    #                                                   100), cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)
    #             cv2.putText(self.vehicle_image, self.res_id, (x_text, y_text-10),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)

    @staticmethod
    def match_org(text, org_type):
        match_value = []
        if org_type == "kh":
            provinces = [
                "ភ្នំពេញ",
                "កំពង់ស្ពឺ",
                "បាត់ដំបង",
                "សៀមរាប",
                "បមានជ័យ",
                "កំពង់ឆ្នាំង",
                "កំពង់ធំ",
                "កំពង់ចាម",
                "កំពត",
                "កណ្តាល",
                "កោះកុង",
                "កែប",
                "ក្រចេះ",
                "មណ្ឌលគីរី",
                "ឧមានជ័យ",
                "ប៉ៃលិន",
                "ព្រះសីហនុ",
                "ព្រះវិហារ",
                "ពោធិ៍សាត់",
                "ព្រៃវែង",
                "រតនគីរី",
                "ស្ទឹងត្រែង",
                "ស្វាយរៀង",
                "តាកែវ",
                "ត្បូងឃ្មុំ",
                "កម្ពុជា",
                "នគរបាល",
            ]
        else:
            regex = re.compile("[^A-Z ]")
            # First parameter is the replacement, second parameter is your input string
            text = regex.sub("", text)
            provinces = [
                "PHNOM PENH",
                "KAMPONG SPEU",
                "BATTAMBANG",
                "SIEM REAP",
                "BANTEAY MEANCHEY",
                "KAMPONG CHHNANG",
                "KAMPONG THOM",
                "KAMPONG CHAM",
                "KAMPOT",
                "KANDAL",
                "KOH KONG",
                "KEP",
                "KRATIE",
                "MONDULKIRI",
                "ODDARMEANCHEY",
                "PAILIN",
                "PREAH SIHANOUK",
                "PREAH VIHEAR",
                "PURSAT",
                "PREY VENG",
                "RATANAKIRI",
                "STUNG TRENG",
                "SVAY RIENG",
                "TAKEO",
                "TBOUNG KHMUM",
                "CAMBODIA",
                "POLICE",
            ]
        for x in provinces:
            sequence = difflib.SequenceMatcher(isjunk=None, a=x, b=text)
            difference = sequence.ratio()
            match_value.append(difference)
        confidence = max(match_value)
        logger.info(f"{org_type} Org matching confidence: {confidence}")
        a = match_value.index(confidence)
        organization_name = provinces[a]
        return organization_name if confidence else ""


def get_class_name(class_id):
    class_names = {
        0: "background",
        1: "aeroplane",
        2: "bicycle",
        3: "bird",
        4: "boat",
        5: "bottle",
        6: "bus",
        7: "car",
        8: "cat",
        9: "chair",
        10: "cow",
        11: "diningtable",
        12: "dog",
        13: "horse",
        14: "motorbike",
        15: "person",
        16: "pottedplant",
        17: "sheep",
        18: "sofa",
        19: "train",
        20: "tvmonitor",
    }
    return class_names.get(class_id)

def japaneseEasyOcr(image) : 
    reader = easyocr.Reader(['ja','en'],recog_network='japanese_g2',)
    stringList = "'-0123456789あいうえおかきくけこさしすせそたちつてとなにぬねのはひふ〜へほまみむめもやゆよらりるれろをわ北土見斎岡福浦島岐部阜群馬広金沢高知新潟市帯大分徳富山宇都宮札幌仙台沼津千葉梨多摩阪横浜前橋練和歌神戸崎埼玉熊谷品川奈良足立京姫路野田鹿児"
    result = reader.readtext(image,paragraph=False,detail = 0,decoder='beamsearch' , beamWidth = 10 ,
                width_ths = 0.01,
                height_ths = 0.1,
                text_threshold=0.1,
                
                           allowlist=stringList)
    combined_string = ''.join(result)

    return combined_string

def list_vehicle_coordinates(
    frame, net, allowed_classes=["motorbike", "car", "bus", "truck"]
):
    frame_height, frame_width, _ = frame.shape
    heightFactor = frame_height / 300.0
    widthFactor = frame_width / 300.0
    frame_resized = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(
        frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False
    )
    # Set to network the input blob
    net.setInput(blob)

    # Prediction of network
    detections = net.forward()
    # Size of frame resize (300x300)
    cols = frame_resized.shape[1]
    rows = frame_resized.shape[0]

    coordinates = []
    classes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Confidence of prediction
        if confidence > 0.5:  # Filter prediction
            class_id = int(detections[0, 0, i, 1])  # Class label
            class_name = get_class_name(class_id)
            if class_name is not None and class_name in allowed_classes:
                obj_xmin = int(detections[0, 0, i, 3] * cols)
                obj_ymin = int(detections[0, 0, i, 4] * rows)
                obj_xmax = int(detections[0, 0, i, 5] * cols)
                obj_ymax = int(detections[0, 0, i, 6] * rows)

                vehicle_add = 0
                motorbike = class_name == "motorbike"
                four_wheels = class_name in ("car", "bus", "truck")
                if motorbike:
                    vehicle_add = 40
                elif four_wheels:
                    vehicle_add = 15

                # Scale object detection to frame
                obj_xmin = max(0, int(widthFactor * obj_xmin) - vehicle_add)
                obj_ymin = max(0, int(heightFactor * obj_ymin) - vehicle_add)
                obj_xmax = int(widthFactor * obj_xmax) + vehicle_add
                obj_ymax = int(heightFactor * obj_ymax) + vehicle_add
                coordinates.append((obj_ymin, obj_ymax, obj_xmin, obj_xmax))
                classes.append(class_name)
    logger.debug(classes)
    return coordinates, classes


def detect_plate(
    frame,
    class_name,
    obj_ymin,
    obj_ymax,
    obj_xmin,
    obj_xmax,
    labels,
    wpod_url,
    ch_url,
    debug=False,
):
    vehicle_image = frame[obj_ymin:obj_ymax, obj_xmin:obj_xmax]
    try:
        LpImg, lp_type, cor = wpod_pred(wpod_url, vehicle_image, 0.85, class_name)
    except Exception as e:
        logger.info(f"detection: {e}")
        raise
    if len(LpImg) >= 1:
        # this means the vehicle is too small
        if LpImg[0].shape[1] > (obj_xmax - obj_xmin) or LpImg[0].shape[0] > (
            obj_ymax - obj_ymin
        ):
            raise
        cor = cor[0]
        lprs = LPRS(
            frame,
            LpImg[0],
            cor,
            lp_type,
            frame[obj_ymin:obj_ymax, obj_xmin:obj_xmax],
            debug=debug,
        )
        # crop_characters = lprs.detect()
        res_id = japaneseEasyOcr(LpImg[0])

        # res_id = ""
        # for ch_img in crop_characters:
        #     ch = ch_predict(ch_url, ch_img, labels)
        #     res_id += ch
        lprs.res_id = res_id

        res_eng, res_kh = lprs.recog_kh_eng()
        if res_eng != "CAMBODIA":
            edited_res_id = (
                plate_operation.Operation(res_id).operation_plate().lstrip("E")
            )
            lprs.res_id = edited_res_id
            print(f"OldID: {res_id}, CorrectedID: {edited_res_id}")
        lprs.draw_box()
        return lprs.res_id, res_eng, res_kh
    else:
        raise


def map_kh_location(location):
    return {"Ou Tasek": "អូតាសេក", "White Shop": "វ៉ាយហ្សប"}[location]


def sendData(
    saved_image,
    organization_name,
    organization_name_khmer,
    plate_number,
    timestamp,
    vehicle_type,
    status,
    location,
):
    #     datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%dT%H:%M:%S')
    auth_token = os.environ.get("BACKEND_API_TOKEN")
    if plate_number == "No Plate":
        organization_name = organization_name_khmer = ""
    hed = {"Authorization": "Bearer " + auth_token}
    data = {
        "plate_number": plate_number,
        "organization_name": organization_name,
        "organization_name_khmer": organization_name_khmer,
        "location": map_kh_location(location),
        "time": timestamp,
        "vehicle_type": vehicle_type,
        "status": status,
    }
    location_folder = location.lower().replace(" ", "")
    img_path = "../realtime_images/" + location_folder + "/" + timestamp + ".png"
    url = os.environ.get("BACKEND_API_COMMITLOGS_URL")
    try:
        response = requests.post(url, json=data, headers=hed)
        res_json = response.json()
        logger.info(res_json)
        # logger.debug(saved_image)
        if len(saved_image) and len(res_json) == 3:
            cv2.imwrite(img_path, saved_image)
            if os.environ.get("SEND_IMAGE") == "true":
                subprocess.Popen(
                    [
                        "sshpass",
                        "-p",
                        os.environ.get("SERVER_PASS"),
                        "scp",
                        img_path,
                        f"{os.environ.get('SERVER_REALTIME_IMAGES_PATH')}/{location_folder}",
                    ]
                )
        return res_json
    except Exception as e:
        logger.debug(f"Saved image: {saved_image}, {type(saved_image)}")
        logger.debug(f"Errors: {e}")
        return {}


def map_En(khmer_org):

    org_dict = {
        "ភ្នំពេញ": "PHNOM PENH",
        "កំពង់ស្ពឺ": "KAMPONG SPEU",
        "បាត់ដំបង": "BATTAMBANG",
        "សៀមរាប": "SIEM REAP",
        "បមានជ័យ": "BANTEAY MEANCHEY",
        "កំពង់ឆ្នាំង": "KAMPONG CHHNANG",
        "កំពង់ធំ": "KAMPONG THOM",
        "កំពង់ចាម": "KAMPONG CHAM",
        "កំពត": "KAMPOT",
        "កណ្តាល": "KANDAL",
        "កោះកុង": "KOH KONG",
        "កែប": "KEP",
        "ក្រចេះ": "KRATIE",
        "មណ្ឌលគីរី": "MONDULKIRI",
        "ឧមានជ័យ": "ODDARMEANCHEY",
        "ប៉ៃលិន": "PAILIN",
        "ព្រះសីហនុ": "PREAH SIHANOUK",
        "ព្រះវិហារ": "PREAH VIHEAR",
        "ពោធិ៍សាត់": "PURSAT",
        "ព្រៃវែង": "PREY VENG",
        "រតនគីរី": "RATANAKIRI",
        "ស្ទឹងត្រែង": "STUNG TRENG",
        "ស្វាយរៀង": "SVAY RIENG",
        "តាកែវ": "TAKEO",
        "ត្បូងឃ្មុំ": "TBOUNG KHMUM",
        "កម្ពុជា": "CAMBODIA",
        "": "",
        "នគរបាល": "POLICE",
    }
    return org_dict[khmer_org]


def load_model(path):
    try:
        path = splitext(path)[0]
        with open("%s.json" % path, "r") as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights("%s.h5" % path)
        logger.info("Loading model successfully...")
        return model
    except Exception as e:
        logger.warning(f"load_model error: {e}")


def load_ch_model(nets_path, weight_path):
    json_file = open(nets_path, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weight_path)
    return model

def loadEasyOCR():
    reader = easyocr.Reader(["en"])
    return reader