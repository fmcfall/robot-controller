import numpy as np

def check_none(arr, broad_to):
    if arr is None:
        arr = np.full_like(broad_to, np.nan)

    return arr

class ExtendedKalmanFilter:

    def __init__(self, x, U, P, Q, R, f, h, jacobian_F, jacobian_H):
        self.f = f
        self.jacobian_F = jacobian_F
        self.x = x
        self.P = P
        self.h = h
        self.jacobian_H = jacobian_H
        self.Q = Q
        self.R = R

        self.state_size = self.x.shape[0]
        self.I = np.identity(self.state_size)
        self.kalman_gains = []

        self.U = check_none(U, self.x)

    def predict(self):

        # jacobian of f with respect to x evaluated at x
        A = self.jacobian_F(self.x, self.U)

        # project state ahead
        x_prior = self.f(self.x, self.U)

        # project error covariance ahead
        P_prior = A @ ((self.P @ A.T) + self.Q)

        self.P = P_prior

        return x_prior

    def update(self, states, z, R):

        # jacobian of h with respect to x evaluated at x
        H = self.jacobian_H(self.x, states)

        # innovation (pre-fit residual) covariance
        S = H @ (self.P @ H.T) + R

        # optimal kalman gain
        K = np.linalg.solve(S.T, H @ self.P.T).T
        self.kalman_gains.append(K)

        # update estimate via z
        x_post = self.x + K @ np.atleast_1d((z - self.h(self.x, states)))

        # update error covariance
        P_post = (self.I - K @ H) @ self.P

        self.P = P_post

        return x_post[states]

    def get_state(self, index=-1):

        if index >= 0:
            return self.x[index]
        else:
            return self.x