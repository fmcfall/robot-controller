import cv2 as cv
import numpy as np
import mediapipe as mp
import time
from sensapex import UMP

# webcam
video = cv.VideoCapture(0)
# robot
#ump = UMP.get_ump()
#dev_ids = ump.list_devices()
#manipulator = ump.get_device(1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1) # less hands = higher fps
mp_draw = mp.solutions.drawing_utils

previous_time = 0
current_time = 0

hand_numbers = [8] # index finger range from 5-8

# finger position
previous_pos = [0.5, 0.5, 0]
current_pos = [0.5, 0.5, 0]

# robot position (calibrated 0)
robot_pos = np.array([9999.8818359375, 9999.986328125, 9999.9501953125])
robot_pos_new = np.array([9999.8818359375, 9999.986328125, 9999.9501953125])

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
                # according to https://google.github.io/mediapipe/solutions/hands.html, z uses similar scale to x
                landmark_d = int(landmark.z * image_width)

                if id in hand_numbers:
                    id_norm_pos = [landmark.x, landmark.y, landmark.z]
                    id_pos = (landmark_w, landmark_h, landmark_d)

                    # draw circle on finger
                    cv.circle(img, id_pos[:2], 5, (255, 0, 255), cv.FILLED)
                    #print(id, id_norm_pos)

                    # calcualate normalised position difference
                    current_pos = id_norm_pos
                    pos_diff = np.subtract(current_pos, previous_pos)
                    previous_pos = current_pos
                    #print(pos_diff)

                    # move robot
                    np.multiply(robot_pos[:3], (np.subtract(1, pos_diff)), robot_pos_new)
                    #manipulator.goto_pos(robot_new_pos, speed=1500)
                    print(robot_pos)
                    robot_pos = robot_pos_new
            

            # Draw entire hand onto image (img, points, connections)
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
