import cv2 as cv
import numpy as np
import mediapipe as mp
import roboticstoolbox as rtb
import numpy as np
import spatialmath as sm
import spatialgeometry as sg
from roboticstoolbox.backends.swift import Swift 

# webcam
video = cv.VideoCapture(0)

# hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1) # less hands = higher fps
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hand_numbers = [8]

# finger position
previous_pos = [0.5, 0.5, 0]
current_pos = [0.5, 0.5, 0]

# Hand to robot scale factor
scale_factor = 0.1

# init swift environment
env = Swift()
env.launch(realtime=True)

# load robot
robot = rtb.models.URDF.UR5()
robot.q = robot.q1

# add robot to swift
env.add(robot)

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

                    # normalised position diff
                    current_pos = id_norm_pos
                    pos_diff = np.subtract(current_pos, previous_pos)
                    previous_pos = current_pos
                    pos_diff = pos_diff * scale_factor
                    x, y, z = pos_diff

                    # goal position, offset by args in m
                    # fkine finds end effector position (forward kinematics)
                    Tep =  robot.fkine(robot.q) * sm.SE3.Tx(x) * sm.SE3.Ty(y) * sm.SE3.Tz(z)
                    print(robot.q)
                    # axes
                    axes = sg.Axes(length=0.05, base=Tep)
                    env.add(axes)
                
                    arrived = False      # arrived flag
                
                    dt = 0.01
                    while not arrived:
                    
                        # gain is speed to goal, thresh is when arrived=True
                        v, arrived = rtb.p_servo(robot.fkine(robot.q), Tep, gain=50, threshold=0.01)
            
                        # Jacobian of robot in end effector frame
                        J = robot.jacobe(robot.q)
            
                        # desired joint velocity of robot, qd is joint velocity in rad
                        robot.qd = np.linalg.pinv(J) @ v
            
                        # v is vector representing spatial error
                        # step the env
                        env.step(dt)

    # Show everything
    cv.imshow("Image", img)
    key = cv.waitKey(1)
    # If q entered whole process will stop
    if key == ord('q'):
        break

env.hold()