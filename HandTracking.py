import cv2 as cv
import mediapipe as mp
import time

video = cv.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Run webcam
while True:
    success, img = video.read()
    img_RBG = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Hand results
    results = hands.process(img_RBG)

    # location of individual hands
    if results.multi_hand_landmarks:
        for h in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, h)

    cv.imshow("Image", img)
    cv.waitKey(1)
