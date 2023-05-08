import cv2 as cv
import numpy as np
import mediapipe as mp
import time
from filters.kf2D import KalmanFilter
import matplotlib.pyplot as plt
import serial.tools.list_ports
import pandas as pd
from filterpy.stats import plot_covariance

ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()
calib = False
portsList = []

for onePort in ports:
    portsList.append(str(onePort))

val = 13 # port number

for x in range(0,len(portsList)):
    if portsList[x].startswith("COM" + str(val)):
        portVar = "COM" + str(val)

serialInst.baudrate = 9600
serialInst.port = portVar
serialInst.open()

# position data
data1 = []
data2 = []

landmark_w = 0
landmark_h = 0
vxm = 0
vym = 0

# velocity data
data3 = []
data4 = []

# webcam
video = cv.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1) # less hands = higher fps
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

previous_time = 0
current_time = 0

hand_numbers = [8] # index finger range from 5-8

# finger position
previous_pos = [0.5, 0.5, 0]
current_pos = [0.5, 0.5, 0]

#Create KalmanFilter object KF
#KalmanFilter(dt, u_x, u_y, x_std_meas, y_std_meas, x_dot_std_meas, y_dot_std_meas)
KF = KalmanFilter(0.1, 1, 1, 0.1, 0.1, 0.1, 0.1)

r_cam = np.zeros([2, 2])
r_cam[0][0] = 0.01
r_cam[1][1] = 0.01
r_imu = np.zeros([2, 2])
r_imu[0][0] = 0.01
r_imu[1][1] = 0.01

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

                    # calcualate normalised position difference
                    current_pos = id_norm_pos
                    pos_diff = np.subtract(current_pos, previous_pos)
                    previous_pos = current_pos
                    a, b, c = current_pos       

    if serialInst.in_waiting:
        packet = serialInst.readline()
        reading_string = packet.decode('utf').rstrip('\r\n')

        if calib == True:
            readings = reading_string.split(" ")
            accel = readings[:3]

            if len(accel) == 3:
                vxm = float(accel[0])
                vym = float(accel[1])
                vzm = float(accel[2])
                velocity = np.array([vxm, vym])

        if "CALIBRATION FINISHED" in reading_string:
            calib = True
            print("IMU CALIBRATION FINISHED")

    if landmark_h != 0 and vxm != 0:

        # KF predict
        x, y, vx, vy = KF.predict()

        # KF update
        (x1, y1) = KF.update([0,1], [[id_pos[0]], [id_pos[1]]], r_cam)
        (vx1, vy1) = KF.update([2,3], [[vxm], [vym]], r_imu)
        #plot_covariance((int(x1), int(y1)), KF.P[0:2, 0:2], std=6, facecolor='g', alpha=0.8)  
        cv.circle(img, (int(x1), int(y1)), 5, (255, 255, 0), 1)
        print(KF.x)
        # data
        data1.append([landmark_w, landmark_h])
        data2.append([int(x1),int(y1)])

    # fps
    current_time = time.time()
    fps = 1/(current_time-previous_time)
    previous_time = current_time
    cv.putText(img, str(int(fps)), (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

    cv.rectangle(img, (150, 150), (350, 350), (255, 0, 255), 2)

    # Show everything
    cv.imshow("Image", img)
    key = cv.waitKey(1)
    # If q entered whole process will stop
    if key == ord('q'):
        serialInst.close()
        break

data1 = np.array(data1)
data2 = np.array(data2)

plt.plot(data1[:,0], data1[:,1], label="without kalman")
plt.plot(data2[:,0], data2[:,1], label="with kalman")

plt.xlim([0, 640])
plt.ylim([480, 0])

plt.legend()
plt.show()

# data1 = pd.DataFrame(data1)
# data2 = pd.DataFrame(data2)
# data1.to_csv('test2d_without_kalman.csv')
# data2.to_csv('test2d_with_kalman.csv')