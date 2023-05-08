#!/usr/bin/env python
import cv2 as cv
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import serial.tools.list_ports
from filters.ukf import UKF
import numpy as np
from filterpy.stats import plot_covariance
from DLT import *
from readCamParams import *
from filters.ekf import ExtendedKalmanFilter

timestep = 0.1

def f(x, u):
    return np.array([x[0] + timestep * x[3] + 0.5 * (timestep**2) * x[6],
                     x[1] + timestep * x[4] + 0.5 * (timestep**2) * x[7],
                     x[2] + timestep * x[5] + 0.5 * (timestep**2) * x[8],
                     x[3] + timestep * x[6],
                     x[4] + timestep * x[7],
                     x[5] + timestep * x[8],
                     x[6],
                     x[7],
                     x[8]])

def jacobian_f(x, u):
    return np.array([[1, 0, 0, x[0]*timestep, 0, 0, 0.5*x[3]*(timestep**2), 0, 0],
                     [0, 1, 0, 0, x[1]*timestep, 0, 0, 0.5*x[4]*(timestep**2), 0],
                     [0, 0, 1, 0, 0, x[2]*timestep, 0, 0, 0.5*x[5]*(timestep**2)],
                     [0, 0, 0, 1, 0, 0, x[3]*timestep, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, x[4]*timestep, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, x[5]*timestep],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1]])

def jacobian_h(x, states):
    return np.eye(9)[states, :]

def h(x, states):
    ret = np.zeros([9, 9])
    ret[0][0] = x[0]
    ret[1][1] = x[1]
    ret[2][2] = x[2]
    ret[3][3] = x[3]
    ret[4][4] = x[4]
    ret[5][5] = x[5]
    ret[6][6] = x[6]
    ret[7][7] = x[7]
    ret[8][8] = x[8]

    return ret[states, states]



def main():

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

    # CAM
    video = cv.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1) # less hands = higher fps
    mp_draw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hand_numbers = [8] # index finger range from 5-8

    # traiangulation 
    # Projection matrices
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)
    kpts_camL = []
    kpts_camR = []
    kpts_3d = []

    landmark_w = 0
    landmark_h = 0
    vx = 0
    vy = 0

    data1 = []
    data2 = []

    np.set_printoptions(precision=3)

    # Process Noise
    q = np.eye(9)
    q[0][0] = 0.1
    q[1][1] = 0.1
    q[2][2] = 0.1
    q[3][3] = 0.05
    q[4][4] = 0.05
    q[5][5] = 0.05
    q[6][6] = 0.1
    q[7][7] = 0.1
    q[8][8] = 0.1

    # create measurement noise covariance matrices
    r_cam = np.zeros([3, 3])
    r_cam[0][0] = 0.01
    r_cam[1][1] = 0.01
    r_cam[2][2] = 0.01
    r_imu = np.zeros([3, 3])
    r_imu[0][0] = 0.01
    r_imu[1][1] = 0.01
    r_imu[2][2] = 0.01

    # initials
    x_in = np.zeros(9)
    p_in = 0.0001*np.eye(9)
    r = 0.1*np.ones(len(x_in))

    state_estimator = ExtendedKalmanFilter(x_in, None, p_in, q, r, f, h, jacobian_f, jacobian_h)

    while True:
        success, img = video.read()
        if not success: break

        imgL = img[:, 0:img.shape[1]//2]
        imgR = img[:, img.shape[1]//2:img.shape[1]]

        image_height, image_width, _ = img.shape

        # Hand results
        resultsL = hands.process(imgL)
        resultsR = hands.process(imgR)

        if resultsL.multi_hand_landmarks:
            for hand in resultsL.multi_hand_landmarks:
                frameL_keypoints = []
                for id, landmark in enumerate(hand.landmark):
                    if id in hand_numbers:
                        pxl_x = landmark.x * imgL.shape[1]
                        pxl_y = landmark.y * imgL.shape[0]
                        pxl_x = int(round(pxl_x))
                        pxl_y = int(round(pxl_y))
                        kpts = [pxl_x, pxl_y]
                        cv.circle(img, kpts, 5, (255, 0, 255), cv.FILLED)
                        frameL_keypoints.append(kpts)
        else:
            frameL_keypoints = [[-1, -1]]*len(hand_numbers)
        kpts_camL.append(frameL_keypoints)

        if resultsR.multi_hand_landmarks:
            for hand in resultsR.multi_hand_landmarks:
                frameR_keypoints = []
                for id, landmark in enumerate(hand.landmark):
                    if id in hand_numbers:
                        pxl_x = landmark.x * imgR.shape[1]
                        pxl_y = landmark.y * imgR.shape[0]
                        pxl_x = int(round(pxl_x))
                        pxl_y = int(round(pxl_y))
                        kpts = [pxl_x, pxl_y]
                        kpts_right = [pxl_x+imgR.shape[1], pxl_y]
                        cv.circle(img, kpts_right, 5, (255, 0, 255), cv.FILLED)
                        frameR_keypoints.append(kpts)
        else:
            frameR_keypoints = [[-1, -1]]*len(hand_numbers)
        kpts_camR.append(frameR_keypoints)

        frame_p3ds = []
        for uv1, uv2 in zip(frameL_keypoints, frameR_keypoints):
            if uv1[0] == -1 or uv2[0] == -1:
                _p3d = [-1, -1, -1]
            else:
                _p3d = DLT(P0, P1, uv1, uv2) #calculate 3d position of keypoint
            frame_p3ds.append(_p3d)

        frame_p3ds = np.array(frame_p3ds[0]).reshape((len(hand_numbers), 3))
        kpts_3d.append(frame_p3ds)
        x3d, y3d, z3d = frame_p3ds[0]

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

        if resultsL.multi_hand_landmarks and resultsR.multi_hand_landmarks and vxm != 0 and axm != 0:

            cam_data = np.array([x3d, y3d, z3d])
            imu_v_data = velocity
            imu_a_data = acceleration

            state_estimator.predict()
            (x1, y1, z1) = state_estimator.update([0,1,2], cam_data, r_cam)
            (vx1, vy1, vz1) = state_estimator.update([3,4,5], imu_v_data, r_imu)
            (ax1, ay1, az1) = state_estimator.update([6,7,8], imu_a_data, r_imu)

            print("Estimated state: ", state_estimator.get_state())

            data1.append([x3d, y3d, z3d])
            data2.append([float(x1), float(y1), float(z1)])

            #cv.circle(img, (int(x1), int(y1)), 5, (255, 255, 0), 1)
            #plot_covariance((state_estimator.x[0], state_estimator.x[1]), state_estimator.p[0:2, 0:2], std=6, facecolor='k', alpha=0.3)
            #plot_covariance((state_estimator.x[0], state_estimator.x[1]), state_estimator.p[0:2, 0:2], std=6, facecolor='g', alpha=0.8)           
            
        # Show everything
        cv.imshow("Image", img)
        key = cv.waitKey(1)
        # If q entered whole process will stop
        if key == ord('q'):
            serialInst.close()
            break

    ''' PLOT '''

    data1 = np.array(data1)
    data2 = np.array(data2)

    x, y, z = data1[:, 0], data1[:, 1], data1[:, 2]
    px, py, pz = data2[:, 0], data2[:, 1], data2[:, 2]

    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot3D(x, y, z, 'r-')
    ax.set_title("Without Kalman Filter")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title("With Unscented Kalman Filter")
    ax.plot3D(px, py, pz, 'g-')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

if __name__ == "__main__":
    
    main()