import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import os

folderPath = "D:\Smart-Mirror\Menu"
myList = os.listdir(folderPath)
overlay = []
for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    overlay.append(image)
header = overlay[0]

wcam, hcam = 1280, 720
video = cv.VideoCapture(0)
video.set(3, wcam)
video.set(4, hcam)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pT = 0
fingList = [8,12,16,20]

mpPose = mp.solutions.pose
pose = mpPose.Pose()
tshirtColor = (0,0,225)

while True:
    ret, img = video.read()
    img = cv.flip(img, 1)
    img[0: 100, 0:1280] = header
    h, w, c = img.shape
    
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    resultsH = hands.process(imgRGB)
    resultsP = pose.process(imgRGB)

    if resultsP.pose_landmarks:
        
        landmarksP = resultsP.pose_landmarks.landmark

        rightS, leftS = landmarksP[12], landmarksP[11]
        rightH, leftH = landmarksP[24], landmarksP[23]

        rsx, rsy = int(rightS.x*w), int(rightS.y*h)
        lsx, lsy = int(leftS.x*w), int(leftS.y*h)
        rhx, rhy = int(rightH.x*w), int(rightH.y*h)
        lhx, lhy = int(leftH.x*w), int(leftH.y*h)

        # torsoWidth = (((lsx - rsx)**2)+((lsy-rsy)**2))**0.5
        # torsoHeight = (((lsx - rhx)**2)+((lsy-rhy)**2))**0.5

        # resizedGarment = cv.resize(garment, (torsoWidth, torsoHeight))

        pts = np.array([[rsx, rsy], [lsx, lsy], [lhx, lhy], [rhx, rhy]], np.int32)
        cv.fillPoly(img, [pts], tshirtColor)

        # cv.rectangle(img, (lsx,lsy), (rhx,rhy), tshirtColor, -1)
        mpDraw.draw_landmarks(img,  resultsP.pose_landmarks, mpPose.POSE_CONNECTIONS)


    if resultsH.multi_hand_landmarks:
        for handIdx, hl in enumerate(resultsH.multi_hand_landmarks):
            lmdic = {}
            fingers = []
            for (i, lm) in enumerate(hl.landmark):
                lmdic[i] = [lm.x*w, lm.y*h]

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
                if fy < lmdic[i-2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            if fingers[1] == 1 and fingers[2] == 1:
                #selection mode
                x1, y1 = int(lmdic[8][0]), int(lmdic[8][1])
                x2, y2 = int(lmdic[12][0]), int(lmdic[12][1])
                cv.circle(img, (x1, y1), 15, (0,0,225), 3)
                cv.circle(img, (x2, y2), 15, (0,0,225), 3)
                if y1 < 150:
                    if 140<x1<425:
                        header = overlay[0]
                        tshirtColor = (0,0,225)
                    elif 425<x1<710:
                        header = overlay[1]
                        tshirtColor = (225,0,225)
                    elif 710<x1<995:
                        header = overlay[2]
                        tshirtColor = (0,225,225)
                    elif 995<x1<1280:
                        header = overlay[3]
                        tshirtColor = (225,225,225)
                    else:
                        continue
            
            if fingers[1] == 1 and fingers[2] == 0:
                #drawing mode
                pass
                
            # mpDraw.draw_landmarks(img, hl, mpHands.HAND_CONNECTIONS)
    
    cT = time.time()
    fps = 1/(cT-pT)
    pT = cT

    cv.putText(img, str(int(fps)), (15,40), cv.FONT_HERSHEY_COMPLEX, 1.5, (255,255,0), 2)

    cv.imshow('Video', img)
    if cv.waitKey(1) & 0XFF == ord('q'):
        break
video.release()
cv.destroyAllWindows()