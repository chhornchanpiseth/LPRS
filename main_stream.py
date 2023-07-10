from sklearn.preprocessing import LabelEncoder
import numpy as np
import cv2

import argparse
from datetime import datetime
import queue

# import sys  # for getsizeof
import time
import threading
from collections import Counter

import plate_operation
from local_utils import wpod_pred, ch_predict
from main_LPRS import LPRS, CentroidTracker, logger, load_ch_model, load_model, japaneseEasyOcr


parser = argparse.ArgumentParser()
parser.add_argument(
    "--debug",
    default=False,
    help="when debug, it will write a video and draw visualization on images",
)
parser.add_argument("--ip", required=True, help="ip address if the camera")
parser.add_argument(
    "--wpod_url",
    required=False,
    help="tf serving wpod net model url to predict from tf serving",
)
parser.add_argument(
    "--ch_url",
    required=False,
    help="tf serving license character recognition model url to predict",
)
parser.add_argument(
    "--in_direction",
    required=True,
    help="The direction of vehicles going in (left, right, down",
)
parser.add_argument("--location", default="White Shop", help="location of the camera")
parser.add_argument(
    "--limit_top", required=False, help="top coodinates to limit vehicles"
)
parser.add_argument(
    "--limit_bottom", required=False, help="below coodinates to limit vehicles"
)
parser.add_argument(
    "--is_motor_only",
    required=True,
    default=False,
    help="True or False for the script to detect only motorbikes",
)
parser.add_argument(
    "--limit_top_motor", required=True, help="top coodinates to limit motorbikes"
)
parser.add_argument(
    "--limit_bottom_motor", required=True, help="below coodinates to limit motorbikes"
)
parser.add_argument(
    "--motion_right",
    required=False,
    default=1,
    help="max x coordinate to limit motion detection",
)
parser.add_argument(
    "--motion_top",
    required=False,
    default=0,
    help="min y coordinate to limit motion detection",
)
args = parser.parse_args()


def Receive():
    logger.info("start Receive")
    src = args.ip
    cap = cv2.VideoCapture(src)
    logger.info(src)
    #     frame = cv2.resize(frame, (1620, 1080))
    # width = int(width // 1.1)
    logger.info(src)
    while True:
        ret, frame = cap.read()
        if ret:
            #             frame = cv2.resize(frame, (1620, 1080))
            q.put(frame)
        else:
            time.sleep(2)
            cap = cv2.VideoCapture(src)
            logger.warning("Frame Not Found!")


def motion_detection():
    logger.info("Start detecting motion")
    time.sleep(1)
    frame1 = q.get()
    frame2 = q.get()
    # assign minimum motion size in the frame
    frame_height, frame_width, _ = frame1.shape
    motion_y_min = round(float(args.motion_top) * frame_height)
    motion_x_max = round(float(args.motion_right) * frame_width)
    motion_width_size = frame_width * 0.2
    motion_height_size = frame_height * 0.4
    frame_cnt = 0
    logger.debug(f"motion_width_size: {motion_width_size}")
    logger.debug(f"motion_height_size: {motion_height_size}")
    logger.debug(f"motion_y_min: {motion_y_min}")
    logger.debug(f"motion_x_max: {motion_x_max}")
    while True:
        # time_motion = time.time()
        diff = cv2.absdiff(
            frame1[motion_y_min:, 0:motion_x_max], frame2[motion_y_min:, 0:motion_x_max]
        )
        #         diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        Motion = False
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w < motion_width_size or h < motion_height_size:
                continue
            Motion = True
            break
        # due to sometimes car comes in and out without frames in between
        # we need to update 1 frame every 5m+
        if frame_cnt == 300:
            Motion = True
            frame_cnt = 0
        if Motion:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            motion_q.put((frame1, timestamp))
            # logger.info(
            #     f"Motion Detected. w:{w}, h:{h} | Ip cam Queue size: {q.qsize()} | motion_qsize: {motion_q.qsize()}")
        frame_cnt += 1
        frame1 = frame2
        # logger.debug(f"time info {1/(time.time()- time_motion)}")
        frame2 = q.get()


def main_AI():
    logger.info("Start AI")
    #     time.sleep(5)
    logger.info(f"motion_q {motion_q.qsize()}")
    frame, timestamp = motion_q.get()
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    # motion_x_max = round(float(args.motion_right) * frame_width)
    is_motor_only = True if args.is_motor_only == "True" else False
    logger.info(frame.shape)
    heightFactor = frame_height / 300.0
    widthFactor = frame_width / 300.0
    if debug:
        video_path = (
            f"main_stream_videos/{args.location}_"
            + datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            + "Z"
            + ".avi"
        )
        out = cv2.VideoWriter(
            video_path,
            # vide compress 
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            26.0,
            (frame_width, frame_height),
        )

    # labels encoder for character recognition
    
    in_direction = args.in_direction
    ct = CentroidTracker()
    ct.direction_in = in_direction
    ct.location = args.location
    # ch_url = args.ch_url
    wpod_url = args.wpod_url
    if wpod_url is None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logger.debug(f"wpod_url: {wpod_url}")
        wpod_net_path = "model/wpod-net.json"
        wpod_url = load_model(wpod_net_path)
    # if ch_url is None:
    #     logger.debug(f"ch_url: {ch_url}")
    #     # replace with easy ocr 
    #     ch_url = load_ch_model(
    #         "model/MobileNets_character_recognition.json",
    #         "model/License_character_recognition_weight.h5",
    #     )

    allowed_classes = ["motorbike"]
    if not is_motor_only:
        allowed_classes += ["car", "bus"]
    logger.debug(allowed_classes)
    while True:
        # logger.info(f"motion_qsize: {motion_q.qsize()}")
        #         timestamp_prev = timestamp
        frame, timestamp = motion_q.get()
        if frame is None:
            continue
        #         if (datetime.strptime(timestamp[:-1], '%Y-%m-%dT%H:%M:%S.%f') - datetime.strptime(timestamp_prev[:-1], '%Y-%m-%dT%H:%M:%S.%f')).total_seconds() > 5:
        #             for object_id in range(6):
        #                 ct.delete(object_id)
        start_vehicle = time.time()
        # MobileNet requires fixed dimensions for input image(s)
        # so we have to ensure that it is resized to 300x300 pixels.
        # set a scale factor to image because network the objects has differents size.
        # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
        # after executing this command our "blob" now has the shape:
        # (1, 3, 300, 300)
        frame_resized = cv2.resize(frame[:, :frame_width], (300, 300))
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

        logger.info(f"[TIME] vehicle: FPS {1.0/(time.time() - start_vehicle)}")
        LpImg = []
        # for object tracking
        rects = []
        ids = []
        engs = []
        khs = []
        vehicle_types = []
        vehicle_images = []
        # For get the class and location of object detected,
        # There is a fix index for class, location and confidence
        # value in @detections array .
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]  # Confidence of prediction
            if confidence > 0.5:  # Filter prediction
                class_id = int(detections[0, 0, i, 1])  # Class label
                class_name = class_names[class_id]
                if class_id in class_names and class_name in allowed_classes:
                    # Object location
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
                    vehicle_image = frame[obj_ymin:obj_ymax, obj_xmin:obj_xmax]
                    start_plate = time.time()
                    try:
                        LpImg, lp_type, cor = wpod_pred(
                            wpod_url, vehicle_image, 0.85, class_name
                        )
                    except Exception as e:
                        LpImg = []
                        logger.info(f"detection: {e}")
                    logger.info(
                        f"[TIME] (plate): FPS {1.0/(time.time() - start_plate)}"
                    )
                    if len(LpImg) >= 1:
                        # this means the vehicle is too small
                        if LpImg[0].shape[1] > (obj_xmax - obj_xmin) or LpImg[0].shape[
                            0
                        ] > (obj_ymax - obj_ymin):
                            continue
                        cor = cor[0]
                        lprs = LPRS(
                            frame,
                            LpImg[0],
                            cor,
                            lp_type,
                            frame[obj_ymin:obj_ymax, obj_xmin:obj_xmax],
                            debug=debug,
                        )
                       
                       
                        res_id = japaneseEasyOcr(LpImg[0])
                        #  crop_characters = lprs.detect()
                        # res_id = ""
                        # for ch_img in crop_characters:
                        #     start_ch = time.time()
                        #     ch = ch_predict(ch_url, ch_img, labels)
                        #     logger.debug(f"[TIME] ch: FPS {1.0/(time.time()-start_ch)}")
                        #     res_id += ch
                        logger.info(
                            f"[TIME] (plate+ch): FPS {1.0/(time.time() - start_plate)}"
                        )
                            
                        lprs.res_id = res_id

                        # res_eng, res_kh = lprs.recog_kh_eng() 
                        # sample output res_eng KAMPONG CHAM res_kh កំពង់ចាម
                        # because CAMBODIA is customized plate, so cannot use plate_operation
                        # if res_eng != "CAMBODIA":
                        #     edited_res_id = (
                        #         plate_operation.Operation(res_id)
                        #         .operation_plate()
                        #         .lstrip("E")
                        #     )
                        #     lprs.res_id = edited_res_id
                        #     logger.info(
                        #         f"[RESULT] OldID: {res_id}, CorrectedID: {edited_res_id}"
                        #     )
                        #     res_id = edited_res_id
                        lprs.draw_box()
                        logger.info(
                            f"{timestamp} [TIME] (detect done): FPS {1.0/(time.time() - start_plate)}"
                        )
                        # x_plate = cor[0]
                        # y_plate = cor[1]
                        # cx = round(sum(x_plate) / 4) + obj_xmin
                        # cy = round(sum(y_plate) / 4) + obj_ymin
                        # centroid = (cx, cy)
                        # put text, rects... in the frame for showing in the video
                        if debug:
                            frame = lprs.frame
                            frame[
                                obj_ymin:obj_ymax, obj_xmin:obj_xmax
                            ] = lprs.vehicle_image
                            cv2.putText(
                                frame,
                                f"{class_name}: {str(confidence*100)}%",
                                (obj_xmin, obj_ymin - 5),
                                0,
                                0.75,
                                (255, 255, 255),
                                2,
                            )
                            cv2.rectangle(
                                frame,
                                (obj_xmin, obj_ymin),
                                (obj_xmax, obj_ymax),
                                (255, 255, 255),
                                2,
                            )
                    else:
                        res_eng = res_kh = ""
                        res_id = "No Plate"
                    # centroid of the car in the frame
                    rects.append(
                        (
                            int((obj_xmin + obj_xmax) // 2),
                            int((obj_ymin + obj_ymax) // 2),
                        )
                    )
                    vehicle_types.append(class_name)
                    vehicle_images.append(vehicle_image)
                    engs.append(res_id)
                    khs.append(res_id)
                    ids.append(res_id)

                    logger.info(Counter(ct.all_engs[ct.nextObjectID - 1]))
                    logger.info(Counter(ct.all_khs[ct.nextObjectID - 1]))

        objects = ct.update(
            rects, ids, engs, khs, vehicle_images, vehicle_types, timestamp
        )
        if debug:
            # loop over the tracked objects
            for (objectID, centroid) in objects.items():
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {} disappeared {}".format(objectID, ct.disappeared[objectID])
                cv2.putText(
                    frame,
                    text,
                    (centroid[0] - 10, centroid[1] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            out.write(frame)

        logger.info(
            f"motionq: {motion_q.qsize()} {Counter(ct.all_ids[ct.nextObjectID-1])}"
        )


if __name__ == "__main__":
    # DEBUG HERE
    debug = False if args.debug == "False" else True
    logger.info(debug)

    net = cv2.dnn.readNetFromCaffe(
        "model/MobileNetSSD_deploy.prototxt", "model/MobileNetSSD_deploy.caffemodel"
    )
    # Labels of Network.
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
    q = queue.Queue()
    motion_q = queue.Queue()
    t1 = threading.Thread(target=Receive)
    t2 = threading.Thread(target=main_AI)
    t3 = threading.Thread(target=motion_detection)
    t1.start()
    t2.start()
    t3.start()
