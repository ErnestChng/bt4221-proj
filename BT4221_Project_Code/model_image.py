# To use this script, please enter either of the codes below the code below:
# python model_image.py --image test_images/prof.png
# python model_image.py --image test_images/prof2.jpg
# python model_image.py --image test_images/prof3.jpg

import argparse

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-m", "--model", type=str, default="model/face_mask_detector.model", help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# age buckets for prediction
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

# gender buckets for prediction
GENDER_BUCKETS = ['Male', 'Female']

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

# reading image and constructing a blob
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                             (104.0, 177.0, 123.0))
face_detection_model.setInput(blob)
detections = face_detection_model.forward()

# iterating over different faces
for i in range(0, detections.shape[2]):
    # extracting the probability
    detection_confidence = detections[0, 0, i, 2]
    # filtering out weak detections
    if detection_confidence > args["confidence"]:
        # computing bounding box
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        # data preprocessing for age and gender detection
        face = image[startY:endY, startX:endX]
        faceBlob = cv2.dnn.blobFromImage(face,
                                         1.0,
                                         (227, 227),
                                         (78.4263377603, 87.7689143744, 114.895847746),
                                         swapRB=False)

        # making age predictions
        age_model.setInput(faceBlob)
        preds = age_model.forward()
        i = preds[0].argmax()
        age = AGE_BUCKETS[i]
        print(f'Age: {age[1:-1]} years')

        # making gender predictions
        gender_detection_model.setInput(faceBlob)
        preds = gender_detection_model.forward()
        i = preds[0].argmax()
        gender = GENDER_BUCKETS[i]
        print(f'Gender: {gender}')

        # data preprocessing for mask detection
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # making face mask detections
        (mask, withoutMask) = mask_model.predict(face)[0]
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        # computing probability
        label = "{}, {}, {}: {:.2f}%".format(gender, age, label, max(mask, withoutMask) * 100)

        cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# image output
cv2.imshow("Output", image)
cv2.waitKey(0)
