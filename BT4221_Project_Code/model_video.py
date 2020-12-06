# To use this script, please enter either of the codes below the code below:
# python model_video.py

import argparse
import time

import cv2
import imutils
import numpy as np
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


def detect_and_predict_mask(frame, face_detection_model, mask_model, gender_net, age_model):
    # age buckets for prediction
    AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
    # gender buckets for prediction
    GENDER_BUCKETS = ['Male', 'Female']

    # constructing a blob from the image
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    face_detection_model.setInput(blob)
    detections = face_detection_model.forward()

    # initialise a list of faces
    faces = []
    locs = []
    faceBlobs = []
    results = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the probability
        confidence = detections[0, 0, i, 2]
        # filtering out weak detections
        if confidence > args["confidence"]:
            # computing bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # data preprocessing
            face = frame[startY:endY, startX:endX]
            faceBlob = cv2.dnn.blobFromImage(face,
                                             1.0,
                                             (227, 227),
                                             (78.4263377603, 87.7689143744, 114.895847746),
                                             swapRB=False)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            faceBlobs.append(faceBlob)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    # ensuring a prediction is made only when face is detected
    if len(faces) > 0:
        for i in range(len(faces)):
            # predicting face mask
            (mask, withoutMask) = mask_model.predict(faces[i])[0]

            # predicting age
            age_model.setInput(faceBlobs[i])
            preds = age_model.forward()
            x = preds[0].argmax()
            age = AGE_BUCKETS[x]

            # predicting gender
            gender_net.setInput(faceBlobs[i])
            preds = gender_net.forward()
            y = preds[0].argmax()
            gender = GENDER_BUCKETS[y]

            d = {
                "loc": locs[i],
                "gender_age": (gender, age),
                "mask_pred": (mask, withoutMask)
            }
            results.append(d)
    return results


# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="model/face_mask_detector.model", help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# loading opencv face detection model
model_path = "model/opencv_face_detector_uint8.pb"
weights_path = "model/opencv_face_detector.pbtxt"
face_detection_model = cv2.dnn.readNet(model_path, weights_path)

# load pretrained age model
model_path = "model/age_model/age_model.caffemodel"
weights_path = "model/age_model/age_weight.prototxt"
age_model = cv2.dnn.readNet(model_path, weights_path)

# loading pretrained gender model
model_path = "model/gender_model/gender_model.caffemodel"
weights_path = "model/gender_model/gender_weight.prototxt"
gender_detection_model = cv2.dnn.readNet(model_path, weights_path)

# loading pretrained face mask detection model
mask_model = load_model(args["model"])

# initialize video stream from webcam
print("Starting Video Stream...............................")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # resizing the frame
    frame = vs.read()
    frame = imutils.resize(frame, width=1500)

    # model prediction
    results = detect_and_predict_mask(frame, face_detection_model, mask_model, gender_detection_model, age_model)

    for r in results:
        # labelling Mask or No Mask
        (startX, startY, endX, endY) = r["loc"]
        (mask, withoutMask) = r["mask_pred"]
        gender = r["gender_age"][0]
        age = r["gender_age"][1]
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        # computing probability
        text = "{}, {}, {}: {:.2f}%".format(gender, age, label, max(mask, withoutMask) * 100)
        cv2.putText(frame, text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # frame output
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # ending the video stream with "Q"
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
