#!/usr/bin/env python
import cv2 as cv
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import serial.tools.list_ports
from ukf import UKF
import numpy as np
from filterpy.stats import plot_covariance
from DLT import *
from readCamParams import *
from arucoDetect import *
import pandas as pd

def iterate_x(x_in, timestep, inputs):
    '''this function is based on the x_dot and can be nonlinear as needed'''
    ret = np.zeros(len(x_in))
    ret[0] = x_in[0] + timestep * x_in[3] + 0.5 * (timestep**2) * x_in[6] #x
    ret[1] = x_in[1] + timestep * x_in[4] + 0.5 * (timestep**2) * x_in[7] #y
    ret[2] = x_in[2] + timestep * x_in[5] + 0.5 * (timestep**2) * x_in[8] #z
    ret[3] = x_in[3] + timestep * x_in[6] #vx
    ret[4] = x_in[4] + timestep * x_in[7] #vy
    ret[5] = x_in[5] + timestep * x_in[8] #vz
    ret[6] = x_in[6] #ax
    ret[7] = x_in[7] #ay
    ret[8] = x_in[8] #az
    return ret


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
    video = cv.VideoCapture(1)
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

    # traiangulation 
    kpts_camL = []
    kpts_camR = []
    kpts_3d = []

    landmark_w = 0
    landmark_h = 0
    vx = 0
    vy = 0

    data1 = []
    data2 = []
    data3 = []

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

    # pass all the parameters into the UKF!
    # number of state variables, process noise, initial state, initial covariance, alpha, kappa, beta, iterate function
    state_estimator = UKF(9, q, np.zeros(9), 0.0001*np.eye(9), 0.1, 0.0, 2.0, iterate_x)
    state_estimator1 = UKF(9, q, np.zeros(9), 0.0001*np.eye(9), 0.2, 0.0, 2.0, iterate_x)
    state_estimator2 = UKF(9, q, np.zeros(9), 0.0001*np.eye(9), 0.5, 0.0, 2.0, iterate_x)

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

                if "CALIBRATION FINISHED" in reading_string:
                    calib = True
                    print("IMU CALIBRATION FINISHED")

        if ret0 and ret1 and vxm != 0 and axm != 0:

            cam_data = np.array([x3d, y3d, z3d])
            imu_v_data = velocity
            imu_a_data = acceleration

            d_time = 0.1
            state_estimator.predict(d_time)
            state_estimator1.predict(d_time)
            state_estimator2.predict(d_time)

            #plot_covariance((state_estimator.x[0], state_estimator.x[1]), state_estimator.P[0:2, 0:2], std=6, facecolor='k', alpha=0.3)

            state_estimator.update([0,1,2], cam_data, r_cam)
            state_estimator.update([3,4,5], imu_v_data, r_imu)
            state_estimator.update([6,7,8], imu_a_data, r_imu)
            state_estimator1.update([0,1,2], cam_data, r_cam)
            state_estimator1.update([3,4,5], imu_v_data, r_imu)
            state_estimator1.update([6,7,8], imu_a_data, r_imu)
            state_estimator2.update([0,1,2], cam_data, r_cam)
            state_estimator2.update([3,4,5], imu_v_data, r_imu)
            state_estimator2.update([6,7,8], imu_a_data, r_imu)

            x1, y1, z1, vx1, vy1, vz1, ax1, ay1, az1 = state_estimator.get_state()
            x1_1, y1_1, z1_1, vx1, vy1, vz1, ax1, ay1, az1 = state_estimator1.get_state()
            x2_1, y2_1, z2_1, vx2, vy2, vz2, ax2, ay2, az2 = state_estimator2.get_state()

            data3.append([float(x2_1), float(y2_1), float(z2_1)])
            data2.append([float(x1), float(y1), float(z1)])
            data1.append([float(x1_1), float(y1_1), float(z1_1)])

            #plot_covariance((state_estimator.x[0], state_estimator.x[1]), state_estimator.P[0:2, 0:2], std=6, facecolor='g', alpha=0.8)           
            
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
    data3 = np.array(data3)

    x, y, z = data1[20:, 0], data1[20:, 1], data1[20:, 2]
    px, py, pz = data2[20:, 0], data2[20:, 1], data2[20:, 2]
    dx, dy, dz = data3[20:, 0], data3[20:, 1], data3[20:, 2]

    # plt.plot(data2[20:,0], data2[20:,1], label="q=0.1", color='g')
    # plt.plot(data1[20:,0], data1[20:,1], label="q=0.01", color='b')
    # plt.plot(data3[20:,0], data3[20:,1], label="q=0.001", color='r')

    fig = plt.figure()

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.plot3D(x, y, z, 'r-')
    ax.set_title("a = 0.1")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax = plt.gca()
    ax.set_zlim(ax.get_zlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.plot3D(dx, dy, dz, 'b-')
    ax.set_title("a = 0.2")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax = plt.gca()
    ax.set_zlim(ax.get_zlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.set_title("a = 0.5")
    ax.plot3D(px, py, pz, 'g-')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax = plt.gca()
    ax.set_zlim(ax.get_zlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])

    plt.show()

    # data1 = pd.DataFrame(data1)
    # data2 = pd.DataFrame(data2)
    # data2.to_csv('ufk_3d_1.csv')
    # data1.to_csv('no_ufk_3d.csv')

if __name__ == "__main__":
    
    main()