import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import os
import json

folderPath = r"D:\Smart-Mirror\Menu"
tshirtPath = r"D:\Smart-Mirror\assets"

myList = os.listdir(folderPath)
tList = os.listdir(tshirtPath)

overlayTop = []
overlayBottom = []

for i in range(len(myList)):
    image = cv.imread(f'{folderPath}/{myList[i]}')
    overlayTop.append(image)

    timage = cv.imread(f'{tshirtPath}/{tList[i]}', cv.IMREAD_UNCHANGED)
    overlayBottom.append(timage)

header = overlayTop[0]
garment = overlayBottom[0]

if garment is None:
    raise ValueError("Error: Could not load the T-shirt image. Check the file path.")

wcam, hcam = 1280, 720
video = cv.VideoCapture(0)
video.set(3, wcam)
video.set(4, hcam)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pT = 0
fingList = [8, 12, 16, 20]

mpPose = mp.solutions.pose
pose = mpPose.Pose()
tshirtColor = (0, 0, 225)

while True:
    ret, img = video.read()
    img = cv.flip(img, 1)
    img[0:100, 0:1280] = header
    h, w, c = img.shape

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    resultsH = hands.process(imgRGB)
    resultsP = pose.process(imgRGB)

    if resultsP.pose_landmarks:
        landmarksP = resultsP.pose_landmarks.landmark

        rightS, leftS = landmarksP[12], landmarksP[11]
        rightH, leftH = landmarksP[24], landmarksP[23]

        rsx, rsy = int(rightS.x * w), int(rightS.y * h)
        lsx, lsy = int(leftS.x * w), int(leftS.y * h)
        rhx, rhy = int(rightH.x * w), int(rightH.y * h)
        lhx, lhy = int(leftH.x * w), int(leftH.y * h)

        torsoWidth = int((((lsx - rsx) ** 2) + ((lsy - rsy) ** 2)) ** 0.5) * 2
        torsoHeight = int((((lsx - rhx) ** 2) + ((lsy - rhy) ** 2)) ** 0.5) * 2

        scale_w, scale_h = 1.4, 1.7
        tshirt_width = int(torsoWidth * scale_w)
        tshirt_height = int(torsoHeight * scale_h)

        center_x = (lsx + rsx) // 2
        top_y = max(lsy, rsy) - 20
        x1 = center_x - torsoWidth // 2
        y1 = top_y - int(torsoHeight * 0.075)

        # Resize and overlay
        if torsoWidth > 0 and torsoHeight > 0:
            tshirt_resized = cv.resize(garment, (torsoWidth, torsoHeight))

            # Boundary checks
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x1 + torsoWidth)
            y2 = min(h, y1 + torsoHeight)

            roi = img[y1:y2, x1:x2]
            tshirt_cropped = tshirt_resized[0:y2 - y1, 0:x2 - x1]

            # Alpha blending
            if tshirt_cropped.shape[2] == 4:
                alpha_t = tshirt_cropped[:, :, 3] / 255.0
                alpha_b = 1.0 - alpha_t

                for c in range(3):
                    roi[:, :, c] = alpha_t * tshirt_cropped[:, :, c] + alpha_b * roi[:, :, c]

                img[y1:y2, x1:x2] = roi

    if resultsH.multi_hand_landmarks:
        for handIdx, hl in enumerate(resultsH.multi_hand_landmarks):
            lmdic = {}
            fingers = []

            for (i, lm) in enumerate(hl.landmark):
                lmdic[i] = [lm.x * w, lm.y * h]

            handLabel = resultsH.multi_handedness[handIdx].classification[0].label
            cx = int(lmdic[4][0])
            cy = int(lmdic[4][1])

            if handLabel == 'Right':
                if lmdic[4][0] < lmdic[3][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if lmdic[4][0] > lmdic[3][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            for i in fingList:
                fx = int(lmdic[i][0])
                fy = int(lmdic[i][1])
                if fy < lmdic[i - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            if fingers[1] == 1 and fingers[2] == 1:
                # selection mode
                x1, y1 = int(lmdic[8][0]), int(lmdic[8][1])
                x2, y2 = int(lmdic[12][0]), int(lmdic[12][1])
                cv.circle(img, (x1, y1), 15, (0, 0, 225), 3)
                cv.circle(img, (x2, y2), 15, (0, 0, 225), 3)

                if y1 < 150:
                    if 140 < x1 < 425:
                        header = overlayTop[0]
                        garment = overlayBottom[0]
                    elif 425 < x1 < 710:
                        header = overlayTop[1]
                        garment = overlayBottom[1]
                    elif 710 < x1 < 995:
                        header = overlayTop[2]
                        garment = overlayBottom[2]
                    elif 995 < x1 < 1280:
                        header = overlayTop[3]
                        garment = overlayBottom[3]
                    else:
                        continue

            if fingers[1] == 1 and fingers[2] == 0:
                # drawing mode
                pass

    cT = time.time()
    fps = 1 / (cT - pT)
    pT = cT

    cv.putText(img, str(int(fps)), (15, 40), cv.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 0), 2)
    cv.imshow('Video', img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()
