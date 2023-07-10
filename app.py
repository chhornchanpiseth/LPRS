import cv2
import os
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from flask import Flask, request, make_response
from main_LPRS import list_vehicle_coordinates, detect_plate, sendData, logger
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def image_classifier():
    params = request.get_json()
    arr = np.asarray(
        bytearray(
            requests.get(
                params.get("image_url"),
                auth=HTTPBasicAuth(
                    os.environ["CameraUsername"], os.environ["CameraPassword"]
                ),
            ).content
        ),
        dtype=np.uint8,
    )
    logger.debug(f"params: {params}")
    frame = cv2.imdecode(arr, -1)
    if frame is None:
        return make_response(
            {"message": "Error getting image from IP Camera", "status": "error"}, 200
        )
    coordinates, classes = list_vehicle_coordinates(frame, net)
    if not coordinates:
        return make_response({"message": "No vehicles found", "status": "error"}, 200)
        # return abort(404, "No vehicles found")
    plates = []
    for i, c in enumerate(coordinates):
        try:
            class_name = classes[i]
            obj_ymin, obj_ymax, obj_xmin, obj_xmax = c
            res_id, res_eng, res_kh = detect_plate(
                frame, class_name, *c, labels, wpod_url, ch_url
            )
            logger.info(f"{res_id} {res_eng} {res_kh}")
        except Exception as err:
            logger.error(err.__traceback__)
            logger.error(type(err))
            logger.error(err)

        res_backend = sendData(
            frame[obj_ymin:obj_ymax, obj_xmin:obj_xmax],
            res_eng,
            res_kh,
            res_id,
            params["datetime"],
            class_name,
            params["status"],
            params["location"],
        )
        if res_backend is not None:
            plates.append(
                {
                    "plate_number": res_id,
                    "organization_name": res_eng,
                    "organization_name_khmer": res_kh,
                }
            )
    return make_response({"message": plates, "status": "success"}, 201)


if __name__ == "__main__":
    net = cv2.dnn.readNetFromCaffe(
        "model/MobileNetSSD_deploy.prototxt", "model/MobileNetSSD_deploy.caffemodel"
    )
    ch_url = os.environ["CH_URL"]
    wpod_url = os.environ["WPOD_URL"]
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
    labels = LabelEncoder()
    labels.classes_ = np.load("model/license_character_classes.npy")

    allowed_classes = ["motorbike", "car", "bus"]

    app.run(host="0.0.0.0", port=5000)
