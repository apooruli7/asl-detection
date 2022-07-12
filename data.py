import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

off = 20

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((300,300,3), np.uint8) * 255
        imgCrop = img[y - off:y + h + off, x - off:x + w + off]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        if aspectRatio > 1:
            k = 300/h
            wCalculated = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCalculated, 300))
            imgResizeShape = imgResize.shape
            imgWhite[0:imgResizeShape[0], 0:imgResizeShape[1]] = imgResize


        cv2.imshow("Crop Image", imgCrop)
        cv2.imshow("White Image", imgWhite)


    cv2.imshow("Image", img)
    cv2.waitKey(1)
