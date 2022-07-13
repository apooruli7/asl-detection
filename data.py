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
            widthGap = math.ceil((300 - wCalculated) / 2) # making a gap to center the image in the white overlay
            imgWhite[:, widthGap: wCalculated + widthGap] = imgResize

        else:
            k = 300 / w
            hCalculated = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (300, hCalculated))
            imgResizeShape = imgResize.shape
            heightGap = math.ceil((300 - hCalculated) / 2)
            imgWhite[heightGap: hCalculated + heightGap, :] = imgResize


        cv2.imshow("Crop Image", imgCrop)
        cv2.imshow("White Image", imgWhite)


    cv2.imshow("Image", img)
    cv2.waitKey(1)
