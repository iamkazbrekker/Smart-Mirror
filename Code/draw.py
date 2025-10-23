import cv2 as cv
import mediapipe as mp
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

while True:
    ret, img = video.read()
    img = cv.flip(img, 1)
    img[0: 100, 0:1280] = header
    h, w, c = img.shape
    
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handIdx, hl in enumerate(results.multi_hand_landmarks):
            lmdic = {}
            fingers = []
            for (i, lm) in enumerate(hl.landmark):
                lmdic[i] = [lm.x*w, lm.y*h]

            handLabel = results.multi_handedness[handIdx].classification[0].label
            cx = int(lmdic[4][0])
            cy = int(lmdic[4][1])

            if handLabel == 'Right':
                if lmdic[4][0] < lmdic[3][0]:
                    cv.circle(img, (cx, cy), 15, (0,0,225), 5)
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if lmdic[4][0] > lmdic[3][0]:
                    cv.circle(img, (cx, cy), 15, (0,0,225), 5)
                    fingers.append(1)
                else:
                    fingers.append(0)

            for i in fingList:
                fx = int(lmdic[i][0])
                fy = int(lmdic[i][1])
                if fy < lmdic[i-2][1]:
                    cv.circle(img, (fx, fy), 15, (0,0,225), 5)
                    fingers.append(1)
                else:
                    fingers.append(0)

            if fingers[1] == 1 and fingers[2] == 1:
                #selection mode
                x1, y1 = lmdic[8][0], lmdic[8][1]
                x2, y2 = lmdic[12][0], lmdic[12][1]
                if y1 < 150:
                    if 140<x1<425:
                        header = overlay[0]
                    elif 425<x1<710:
                        header = overlay[1]
                    elif 710<x1<995:
                        header = overlay[2]
                    elif 995<x1<1280:
                        header = overlay[3]
                    else:
                        continue
            
            if fingers[1] == 1 and fingers[2] == 0:
                #drawing mode
                pass
                
            mpDraw.draw_landmarks(img, hl, mpHands.HAND_CONNECTIONS)
    
    cT = time.time()
    fps = 1/(cT-pT)
    pT = cT

    cv.putText(img, str(int(fps)), (15,40), cv.FONT_HERSHEY_COMPLEX, 1.5, (255,255,0), 2)

    cv.imshow('Video', img)
    if cv.waitKey(1) & 0XFF == ord('q'):
        break
video.release()
cv.destroyAllWindows()