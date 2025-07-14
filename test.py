import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf
import time

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="Model/model_unquant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape'][1:3]  # (224, 224)
input_height, input_width = input_shape

# Load labels
with open("Model/labels.txt", "r") as f:
    labels = f.read().splitlines()

# Init camera and detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

# FPS init
pTime = time.time()

while True:
    success, img = cap.read()
    if not success:
        print("❌ Camera frame not captured.")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Bounding box safety
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(x + w + offset, img.shape[1])
        y2 = min(y + h + offset, img.shape[0])
        imgCrop = img[y1:y2, x1:x2]

        # Aspect ratio handling
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Preprocess for model
        imgInput = cv2.resize(imgWhite, (input_width, input_height))
        imgInput = np.expand_dims(imgInput, axis=0).astype(np.float32) / 255.0

        # Model Inference
        interpreter.set_tensor(input_details[0]['index'], imgInput)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        index = int(np.argmax(predictions))
        confidence = predictions[index]

        # Label & Confidence
        label = labels[index] if index < len(labels) else "Unknown"
        displayText = f"{label} ({confidence*100:.1f}%)"

        # Drawing
        cv2.rectangle(imgOutput, (x - offset, y - offset - 60),
                      (x - offset + 180, y - offset - 10), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, displayText, (x - offset + 5, y - offset - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 3)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(imgOutput, f"FPS: {int(fps)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Sign Detection", imgOutput)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
