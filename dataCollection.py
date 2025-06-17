import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/best_sign_language_model30.h5", "Model/labels2.txt")

offset=20
imgSize = 224

folder = "Data/test/H"
counter = 0
if not os.path.exists(folder):
    os.makedirs(folder)


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgHeight, imgWidth, _ = img.shape
        
        # Calculate safe boundaries with offset
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(imgWidth, x + w + offset)
        y2 = min(imgHeight, y + h + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
    
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape


        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap =  math.ceil((imgSize -wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction, index)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap =  math.ceil((imgSize -hCal)/2)
            imgWhite[hGap:hCal+hGap,:] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction, index)



        cv2.imshow ('Image', imgCrop)
        cv2.imshow ('imageWhite', imgWhite)

    cv2.imshow ('Image', img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter +=1 
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite) 
        print(counter)
    elif key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

