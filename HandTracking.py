import cv2 as cv
import mediapipe as mp
import time

video = cv.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

previous_time = 0
current_time = 0

# Run webcam
while True:
    success, img = video.read()
    img_RBG = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    image_height, image_width, _ = img.shape

    # Hand results
    results = hands.process(img_RBG)

    # Location of individual hands
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for id, landmark in enumerate(hand.landmark):
                landmark_h, landmark_w = int(landmark.y * image_height), int(landmark.x * image_width)
                if id == 8 or id == 7 or id == 6 or id == 5: # 8 is index finger
                    cv.circle(img, (landmark_w, landmark_h), 5, (255, 0, 255), cv.FILLED)
            # Draw hand onto image (img, points, connections)
            # mp_draw.draw_landmarks(img, hand)


    '''
    drawing landmarks of finger has fps about 20, whole hand has fps about 10
    '''


    # fps
    current_time = time.time()
    fps = 1/(current_time-previous_time)
    previous_time = current_time
    cv.putText(img, str(int(fps)), (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

    # Show everything
    cv.imshow("Image", img)
    key = cv.waitKey(1)
    # If q entered whole process will stop
    if key == ord('q'):
        break
