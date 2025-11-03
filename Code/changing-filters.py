import cv2 as cv
import mediapipe as mp
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets')))
from filters import apply_rgb, apply_gray, apply_hsv, apply_sepia, apply_invert, apply_warm, apply_cool, apply_edge # type: ignore

mpHands = mp.solutions.hands  # type: ignore
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils # type: ignore

video = cv.VideoCapture(0)
video.set(3, 1280)
video.set(4, 720)

pT = 0
last_change = 0

filters = [
    ('BGR',apply_rgb), 
    ('GRAY', apply_gray), 
    ('HSV', apply_hsv),
    ('SEPIA', apply_sepia),
    ('INVERT', apply_invert),
    ('WARM', apply_warm),
    ('COOL', apply_cool),
    ('EDGE', apply_edge)
    ]
filterIdx = 0

while True:
    ret, img = video.read()
    img = cv.flip(img, 1)
    h, w, c = img.shape
    fingers = {}

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handIdx, hl in enumerate(results.multi_hand_landmarks):
            lmdic = {}

            for (i, lm) in enumerate(hl.landmark):
                lmdic[i] = [int(lm.x * w), int(lm.y * h)]

            handLabel = results.multi_handedness[handIdx].classification[0].label
            fingers[handLabel] = [lmdic[8][0], lmdic[8][1]]

            if 'Right' in fingers and 'Left' in fingers:
                fx, fy = fingers['Right']
                sx, sy = fingers['Left']

                cv.rectangle(img, (fx, fy), (sx, sy), (0,0,0), 3)
                dist = (((sx-fx)**2)+((sy-fy)**2))**0.5

                x1, x2 = sorted([sx,fx])
                y1, y2 = sorted([sy, fy])
                roi = img[y1:y2, x1:x2]
                
                if roi.size>0:
                    if dist < 75 and (time.time() - last_change > 1):
                        filterIdx = (filterIdx + 1) % len(filters)
                        last_change = time.time()
                    filterMode, filterFunction = filters[filterIdx]
                    filtered = (filterFunction(roi))
                    if len(filtered.shape) == 2:  # grayscale (H, W)
                        filtered = cv.cvtColor(filtered, cv.COLOR_GRAY2BGR)
                    img[y1:y2, x1:x2] = filtered

                cv.putText(img, f'DIST : {int(dist)}', (15,75), cv.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 2)
                cv.putText(img, f'MODE: {filterMode}', (15,110), cv.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 2)

            mpDraw.draw_landmarks(img, hl, mpHands.HAND_CONNECTIONS)
    cT = time.time()
    fps = 1/(cT-pT)
    pT = cT

    cv.putText(img, f'FPS: {int(fps)}', (15,40), cv.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 2)

    cv.imshow('Video', img)
    if cv.waitKey(1) & 0XFF == ord('q'):
        break
video.release()
cv.destroyAllWindows()
