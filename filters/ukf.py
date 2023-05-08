import numpy as np
import scipy.linalg
from copy import deepcopy
from threading import Lock

class UKF:
    def __init__(self, num_states, process_noise, initial_state, initial_covar, alpha, kappa, beta, iterate_function):

        self.n = int(num_states)
        self.n_sig = 1 + num_states * 2
        self.Q = process_noise
        self.x = initial_state
        self.P = initial_covar
        self.beta = beta
        self.alpha = alpha
        self.kappa = kappa
        self.iterate = iterate_function

        self.lambd = pow(self.alpha, 2) * (self.n + self.kappa) - self.n

        self.c_weights = np.zeros(self.n_sig)
        self.m_weights = np.zeros(self.n_sig)

        self.c_weights[0] = (self.lambd / (self.n + self.lambd)) + (1 - pow(self.alpha, 2) + self.beta)
        self.m_weights[0] = (self.lambd / (self.n + self.lambd))

        for i in range(1, self.n_sig):
            self.c_weights[i] = 1 / (2*(self.n + self.lambd))
            self.m_weights[i] = 1 / (2*(self.n + self.lambd))

        self.sigmas = self.__get_sigmas()

        self.lock = Lock()

    def __get_sigmas(self):

        sig = np.zeros((self.n_sig, self.n))

        sqr = scipy.linalg.sqrtm((self.n + self.lambd)*self.P)

        sig[0] = self.x
        for i in range(self.n):
            sig[i+1] = self.x + sqr[i]
            sig[i+1+self.n] = self.x - sqr[i]

        return sig.T

    def predict(self, timestep, inputs=[]):

        self.lock.acquire()

        sig_out = np.array([self.iterate(x, timestep, inputs) for x in self.sigmas.T]).T

        x_out = np.zeros(self.n)

        # for each variable in X
        for i in range(self.n):
            # the mean of that variable is the sum of
            # the weighted values of that variable for each iterated sigma point
            x_out[i] = sum((self.m_weights[j] * sig_out[i][j] for j in range(self.n_sig)))

        P_out = np.zeros((self.n, self.n))
        # for each sigma point
        for i in range(self.n_sig):
            # take the distance from the mean
            # make it a covariance by multiplying by the transpose
            # weight it using the calculated weighting factor
            # and sum
            diff = sig_out.T[i] - x_out
            diff = np.atleast_2d(diff)
            P_out += self.c_weights[i] * np.dot(diff.T, diff)

        # add process noise
        P_out += timestep * self.Q

        self.sigmas = sig_out
        self.x = x_out
        self.P = P_out

        self.lock.release()

    def update(self, states, data, r_matrix):

        self.lock.acquire()

        num_states = len(states)

        # create y, sigmas of just the states that are being updated
        sigmas_split = np.split(self.sigmas, self.n)
        y = np.concatenate([sigmas_split[i] for i in states])

        # create y_mean, the mean of just the states that are being updated
        x_split = np.split(self.x, self.n)
        y_mean = np.concatenate([x_split[i] for i in states])

        # differences in y from y mean
        y_diff = deepcopy(y)
        x_diff = deepcopy(self.sigmas)
        for i in range(self.n_sig):
            for j in range(num_states):
                y_diff[j][i] -= y_mean[j]
            for j in range(self.n):
                x_diff[j][i] -= self.x[j]

        # covariance of measurement
        p_yy = np.zeros((num_states, num_states))
        for i, val in enumerate(np.array_split(y_diff, self.n_sig, 1)):
            p_yy += self.c_weights[i] * val.dot(val.T)

        # add measurement noise
        p_yy += r_matrix

        # covariance of measurement with states
        p_xy = np.zeros((self.n, num_states))
        for i, val in enumerate(zip(np.array_split(y_diff, self.n_sig, 1), np.array_split(x_diff, self.n_sig, 1))):
            p_xy += self.c_weights[i] * val[1].dot(val[0].T)

        K_gain = np.dot(p_xy, np.linalg.inv(p_yy))

        y_actual = data

        self.x += np.dot(K_gain, (y_actual - y_mean))
        self.P -= np.dot(K_gain, np.dot(p_yy, K_gain.T))
        self.sigmas = self.__get_sigmas()

        self.lock.release()

    def get_state(self, index=-1):

        if index >= 0:
            return self.x[index]
        else:
            return self.x

    def get_covar(self):

        return self.P