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
from kf3D import KalmanFilter
from ekf import ExtendedKalmanFilter

timestep = 0.1

# ekf f func
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

# ekf f jacobian
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

# ekf h jacobian
def jacobian_h(x, states):
    return np.eye(9)[states, :]

# ekf h func
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

# ukf sigma point iterate func
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

# run all
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

    # Projection matrices
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)
    intr0, dist0 = read_camera_parameters(0)
    intr1, dist1 = read_camera_parameters(1)    
    aruco_type = "DICT_5X5_250"

    # traiangulation points
    kpts_camL = []
    kpts_camR = []
    kpts_3d = []

    # final data
    ekf_data = []
    ukf_data = []
    kf_data = []

    # print options
    np.set_printoptions(precision=3)

    # Process Noise
    q = np.eye(9)
    q[0][0] = 0.1
    q[1][1] = 0.1
    q[2][2] = 0.1
    q[3][3] = 0.1
    q[4][4] = 0.1
    q[5][5] = 0.1
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

    # detection count
    nul_count = 0
    count = 0

    # initials
    x_in = np.zeros(9)
    p_in = 0.0001*np.eye(9)
    r = 0.1*np.ones(len(x_in))

    # filters init
    state_estimator = UKF(9, q, np.zeros(9), 0.0001*np.eye(9), 0.1, 0.0, 2.0, iterate_x)
    KF = KalmanFilter(0.1, 1, 1, 1, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001)
    EKF = ExtendedKalmanFilter(x_in, None, p_in, q, r, f, h, jacobian_f, jacobian_h)

    # loop()
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
        
            frame_p3ds = np.array(frame_p3ds[0]).reshape((1, 3))
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
            state_estimator.update([0,1,2], cam_data, r_cam)
            state_estimator.update([3,4,5], imu_v_data, r_imu)
            state_estimator.update([6,7,8], imu_a_data, r_imu)
            x1, y1, z1, vx1, vy1, vz1, ax1, ay1, az1 = state_estimator.get_state()

            EKF.predict()
            (ex1, ey1, ez1) = EKF.update([0,1,2], cam_data, r_cam)
            (evx1, evy1, evz1) = EKF.update([3,4,5], imu_v_data, r_imu)
            (eax1, eay1, eaz1) = EKF.update([6,7,8], imu_a_data, r_imu)

            kx2, ky2, kz2, kvx, kvy, kvz, kax, kay, kaz = KF.predict()
            (kx1, ky1, kz1) = KF.update([0,1,2], [[float(x3d)], [float(y3d)], [float(z3d)]], r_cam)
            (kvx1, kvy1, kvz1) = KF.update([3,4,5], [[vxm], [vym], [vzm]], r_cam)
            (kax1, kay1, kaz1) = KF.update([6,7,8], [[axm], [aym], [azm]], r_cam)


            ekf_data.append([ex1, ey1, ez1])
            ukf_data.append([x1, y1, z1])
            kf_data.append([float(kx1), float(ky1), float(kz1)])

        if ret1 and ret0:
            count += 1
        else:
            nul_count += 1

        # Show everything
        cv.imshow("Image", img)
        key = cv.waitKey(1)

        # If q entered whole process will stop
        if key == ord('q'):
            serialInst.close()
            break
        # if r entered count restarts
        if key==ord('r'):
            count = 0
            nul_count = 0
            print(count, nul_count)

    ''' PLOT '''

    kf_data = np.array(kf_data)
    ekf_data = np.array(ekf_data)
    ukf_data = np.array(ukf_data)

    # plt.plot(ukf_data[:,0], ukf_data[:,1], label="UKF result", color='g')
    # plt.plot(ekf_data[:,0], ekf_data[:,1], label="EKF result", color='b')
    # plt.plot(kf_data[:,0], kf_data[:,1], label="KF result", color='r')

    kx, ky, kz = kf_data[10:, 0], kf_data[10:, 1], kf_data[10:, 2]
    ex, ey, ez = ekf_data[10:, 0], ekf_data[10:, 1], ekf_data[10:, 2]
    ux, uy, uz = ukf_data[10:, 0], ukf_data[10:, 1], ukf_data[10:, 2]

    fig = plt.figure()

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.scatter(kx[0],ky[0],kz[0])
    ax.plot3D(kx, ky, kz, 'r-')
    ax.set_title("Kalman Filter")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax = plt.gca()
    ax.set_zlim(ax.get_zlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.scatter(ex[0],ey[0],ez[0])
    ax.set_title("Extended Kalman Filter")
    ax.plot3D(ex, ey, ez, 'g-')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax = plt.gca()
    ax.set_zlim(ax.get_zlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.scatter(ux[0],uy[0],uz[0])
    ax.set_title("Unscented Kalman Filter")
    ax.plot3D(ux, uy, uz, 'b-')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax = plt.gca()
    ax.set_zlim(ax.get_zlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])
    print(count, nul_count+count)
    plt.show()

    kf_data = pd.DataFrame(kf_data)
    ekf_data = pd.DataFrame(ekf_data)
    ukf_data = pd.DataFrame(ukf_data)
    kf_data.to_csv("kfcircle.csv")
    ekf_data.to_csv("ekfcircle.csv")
    ukf_data.to_csv("ukfcircle.csv")

if __name__ == "__main__":
    
    main()