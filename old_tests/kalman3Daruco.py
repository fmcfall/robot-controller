from DLT import *
from readCamParams import *
import serial.tools.list_ports
import numpy as np
import cv2 as cv
import mediapipe as mp
import time
from filters.kf3D import KalmanFilter
import matplotlib.pyplot as plt
import pandas as pd
from aruco.arucoDetect import *

# Kalman
KF = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1, 0.01, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01)

# data init
data1 = []
data2 = []

# IMU 
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

# stereo cam
video = cv.VideoCapture(0)

# media pipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1) # less hands = higher fps
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hand_numbers = [8] # index finger range from 5-8

# Projection matrices
P0 = get_projection_matrix(0)
P1 = get_projection_matrix(1)
intr0, dist0 = read_camera_parameters(0)
intr1, dist1 = read_camera_parameters(1)
aruco_type = "DICT_5X5_250"
arucoDict = cv.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
arucoParams = cv.aruco.DetectorParameters_create()

# time
previous_time = 0
current_time = 0

# finger position
previous_pos = [0.5, 0.5, 0]
current_pos = [0.5, 0.5, 0]

# key points
kpts_camL = []
kpts_camR = []
kpts_3d = []

# meas noise
r_cam = np.zeros([3, 3])
r_cam[0][0] = 0.01
r_cam[1][1] = 0.01
r_cam[2][2] = 0.01
r_imu = np.zeros([3, 3])
r_imu[0][0] = 0.01
r_imu[1][1] = 0.01
r_imu[2][2] = 0.01

# Run webcam
while True:
    success, img = video.read()
    if not success: break

    imgL = img[:, 0:img.shape[1]//2]
    imgR = img[:, img.shape[1]//2:img.shape[1]]

    image_height, image_width, _ = img.shape

    Cx0, Cy0, ret0 = get_centre_points(imgL, ARUCO_DICT[aruco_type], intr0, dist0)
    Cx1, Cy1, ret1 = get_centre_points(imgR, ARUCO_DICT[aruco_type], intr1, dist1)

    points0 = [[Cx0, Cy0]]
    points1 = [[Cx1, Cy1]]

    cv.circle(imgL, (int(Cx0), int(Cy0)), 5, (255, 0, 0), cv.FILLED)
    cv.circle(imgR, (int(Cx1), int(Cy1)), 5, (255, 0, 0), cv.FILLED)

    if ret0 and ret1:
        frame_p3ds = []
        for uv1, uv2 in zip(points0, points1):
            if uv1[0] == -1 or uv2[0] == -1:
                _p3d = [-1, -1, -1]
            else:
                _p3d = DLT(P0, P1, uv1, uv2) #calculate 3d position of keypoint
            frame_p3ds.append(_p3d)
    
        frame_p3ds = np.array(frame_p3ds[0]).reshape((len(hand_numbers), 3))
        kpts_3d.append(frame_p3ds)
        x3d, y3d, z3d = frame_p3ds[0]
        coord_str = "(%.2f %.2f %.2f)" % (x3d, y3d, z3d)
        cv.putText(imgL, coord_str, (30, 30), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

    if serialInst.in_waiting:
        packet = serialInst.readline()
        reading_string = packet.decode('utf').rstrip('\r\n')

        if calib == True:
            readings = reading_string.split(" ")
            accel = readings[:6]

            if len(accel) == 6:
                vxm = float(accel[0])
                vym = float(accel[1])
                vzm = float(accel[2])
                axm = float(accel[3])
                aym = float(accel[4])
                azm = float(accel[5])
                velocity = np.array([vxm, vym, vzm])
                acceleration = np.array([axm, aym, azm])

        if "CALIBRATION FINISHED" in reading_string:
            calib = True
            print("IMU CALIBRATION FINISHED")

    if ret0 and ret1 and vxm != 0:

        # KF predict
        x, y, z, vx, vy, vz, ax, ay, az = KF.predict()

        # KF update
        (x1, y1, z1) = KF.update([0,1,2], [[float(x3d)], [float(y3d)], [float(z3d)]], r_cam)
        (vx1, vy1, vz1) = KF.update([3,4,5], [[vxm], [vym], [vzm]], r_cam)
        (ax1, ay1, az1) = KF.update([6,7,8], [[axm], [aym], [azm]], r_cam)
        #print(float(x3d), float(y3d), vx, vy)
        #cv.circle(img, (float(x1), float(y1)), 5, (255, 255, 0), 1)
        # data
        data1.append([x3d, y3d, z3d])
        data2.append([float(x1), float(y1), float(z1)])
        print("estimated: ", [float(x1), float(y1), float(z1)])
        print("without filter: ", [x3d, y3d, z3d])

    # cv.imshow('left', imgL)
    # cv.imshow('right', imgR)
    cv.imshow('Cam', img)

    # fps
    current_time = time.time()
    fps = 1/(current_time-previous_time)
    previous_time = current_time
    #cv.putText(img, str(int(fps)), (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

    key = cv.waitKey(1)
    # If q entered whole process will stop
    if key == ord('q'):
        break

predictions = np.array(data2)
data = np.array(data1)

''' PLOT '''

x, y, z = data[:, 0], data[:, 1], data[:, 2]
px, py, pz = predictions[:,0], predictions[:,1], predictions[:,2]

fig = plt.figure()

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot3D(x, y, z, 'r-')
ax.set_title("Without Kalman Filter")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_title("With Kalman Filter")
ax.plot3D(px, py, pz, 'g-')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

cv.destroyAllWindows()

data1 = pd.DataFrame(data)
data2 = pd.DataFrame(predictions)
data1.to_csv('test3d_without_kalman.csv')
data2.to_csv('test3d_with_kalman_1.csv')