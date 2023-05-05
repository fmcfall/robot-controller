import numpy as np

class KalmanFilter(object):
    def __init__(self, dt, u_x, u_y, u_z, x_std_meas, y_std_meas, z_std_meas, x_dot_std_meas, y_dot_std_meas, z_dot_std_meas, x_ddot_std_meas, y_ddot_std_meas, z_ddot_std_meas):
        
        self.dt = dt

        # input variables
        self.u = np.matrix([[0], [0], [0]])

        # Intial State
        self.x = np.matrix([[0], [0], [0], [0], [0], [0], [0], [0], [0]])

        # State Transition Matrix
        self.A = np.matrix([[1, 0, 0, self.dt, 0, 0, (self.dt**2)/2, 0, 0],
                            [0, 1, 0, 0, self.dt, 0, 0, (self.dt**2)/2, 0],
                            [0, 0, 1, 0, 0, self.dt, 0, 0, (self.dt**2)/2],
                            [0, 0, 0, 1, 0, 0, self.dt, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, self.dt, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, self.dt],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1]])

        # Matrix B
        self.B = np.matrix([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])

        # Initial Process Noise Covariance
        self.Q = np.matrix([[x_std_meas**2, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, y_std_meas**2, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, z_std_meas**2, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, x_dot_std_meas**2, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, y_dot_std_meas**2, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, z_dot_std_meas**2, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, x_ddot_std_meas**2, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, y_ddot_std_meas**2, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, z_ddot_std_meas**2]])

        # Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])    

    def predict(self):
        
        # upate time state
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        # error covariance
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        return self.x

    def update(self, states, z, R):

        # measurement matrix
        H = np.zeros([3,9])
        H[0][states[0]] = 1
        H[1][states[1]] = 1
        H[2][states[2]] = 1

        S = np.dot(H, np.dot(self.P, H.T)) + R

        # Kalman Gain
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S)) 

        self.x = self.x + np.dot(K, (z - np.dot(H, self.x)))  

        I = np.eye(H.shape[1])

        # error covariance matrix
        self.P = (I - (K * H)) * self.P  
        
        return self.x[states]