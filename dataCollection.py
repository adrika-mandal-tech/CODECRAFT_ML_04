import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

folder = "Data/Z" #Total Dataset is 26 Alphabets X 300 pictures = 7800 files
counter = 0

if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    exit()
while True:
    success, img = cap.read()
    if not success or img is None:
        print("⚠️ Warning: Frame not captured. Skipping...")
        continue
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure crop box stays within image bounds
        x1 = max(x - offset, 0)
        y1 = max(y - offset, 0)
        x2 = min(x + w + offset, img.shape[1])
        y2 = min(y + h + offset, img.shape[0])

        imgCrop = img[y1:y2, x1:x2]

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:             #Height is greater than Width
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:                           #Width is greater than Height
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Hand Tracking", img)

    key = cv2.waitKey(1)
    if key == ord("s"): #Saves the image
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)

cap.release()
cv2.destroyAllWindows()
