import numpy as np
from scipy.linalg import cholesky
import pyorbs
from datetime import timedelta
from pyorbs.pyorbs_det import process_meas_record

class UKF():
    def __init__(self, P, t_start, x0, Q, R, alpha = 1e-3, beta = 2.0, kappa = 0.0):
        self.cov_matrix = P
        self.t_start = t_start
        self.state = x0
        self.Q = Q
        self.R = R
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.dim_x = len(x0)
        self.par_lambda = alpha**2 * (self.dim_x + kappa) - self.dim_x

        self.set_weights()

        self.Y = np.zeros((2 * self.dim_x + 1, self.dim_x))

    def set_weights(self):
        n_sigma = 2 * self.dim_x + 1
        self.w_mean = np.full(n_sigma, 1.0 / (2.0 * (self.dim_x + self.par_lambda)))
        self.w_cov = self.w_mean.copy()

        self.w_mean[0] = self.par_lambda / (self.dim_x + self.par_lambda)
        self.w_cov[0] = self.w_mean[0] + (1 - self.alpha**2 + self.beta)

    def generate_sigma_points(self):
        n = self.dim_x
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = self.state

        try:
            L = cholesky((n + self.par_lambda) * self.cov_matrix)
        except np.linalg.LinAlgError:
            L = cholesky((n + self.par_lambda) * self.cov_matrix + np.eye(n) * 1e-8)

        for i in range(n):
            sigma_points[i+1] = self.state + L[i]
            sigma_points[i+1+n] = self.state - L[i]

        return sigma_points
    
    def prediction(self, k, dt):
        n = self.dim_x
        # 1. Создание сигма-точек:
        sigma_points = self.generate_sigma_points()

        # 2. Получаем преобразованные сигма-точки:
        for i in range (2*n + 1):
            orb = pyorbs.pyorbs.orbit(x = sigma_points[i, :6], time = self.t_start + k * timedelta(seconds = dt))
            orb.move(t = self.t_start + (k+1)*timedelta(seconds = dt))
            self.Y[i, :6] = orb.state_v

        # 3. Предсказываем среднее и ковариационную матрицу:
        y_mean = self.w_mean @ sigma_points
        P_y = (self.w_cov[:, None, None] * (
               (self.Y-y_mean)[:, :, None] @ (self.Y-y_mean)[:, None, :])
               ).sum(axis = 0) + self.Q

        return y_mean, P_y
    
    def correction(self, z, y_mean, P_y):
        # 4. Создаем экземпляр каждой преобразованной
        #    сигма-точки по измерению: 

        Z = np.array([self.h(point) for point in self.Y])

        # 5. Вычисляем предсказанное среднее и предсказанную
        #    ковариационную матрицу:
        z_mean = self.w_mean @ Z
        P_z = (self.w_cov[:, None, None] * (
               (Z-z_mean)[:, :, None] @ (Z-z_mean)[:, None, :])
               ).sum(axis = 0) + self.R
        
        # 6. Вычисляем перекрестную ковариацию:
        P_yz = (self.w_cov[:, None, None] * (
               (self.Y-y_mean)[:, :, None] @ (Z-z_mean)[:, None, :])
               ).sum(axis = 0)
        
        # 7. Вычисляем матрицу усиления:
        try:
            K = P_yz @ np.linalg.inv(P_z)
        except np.linalg.LinAlgError:
            K = P_yz @ np.linalg.pinv(P_z)

        # 8. Вычисляем невязки по измерениям:
        res_z = z - z_mean

        # 9. Корректируем вектор состояния и ковариационную матрицу:
        self.state = y_mean + K @ res_z
        self.cov_matrix = P_y - K @ P_z @ K.T

        return self.state, self.cov_matrix
    
    def step(self, z, k, dt):
        y_mean, P_y = self.prediction(k, dt)
        return self.correction(z, y_mean, P_y)