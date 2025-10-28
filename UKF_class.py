import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from datetime import datetime
import multiprocessing as mp
import functools as ft
import math

import pyorbs
import _config

def process_single_orbit(m, orbit):
    """ Вспомогательная функция для протягивания одной траектории,
    возвращает невязку в данный момоент времени.
    """
    meas = pyorbs.pyorbs_det.process_meas_record(orbit, m, 6, None) #setup_parameters ???
    res = meas['res']

    return res


class Trajectories:
    """ Класс сигма-точек, протягивающий орбиту по измерениям
        для каждой точки.
    """
    
    # Количество сигма точек
    N: int

    # Набор сигма точек
    sigma_points: np.ndarray | None = None

    # Набор измерений
    measurements: pd.DataFrame | None = None

    # Время начала фильтрации
    t_start: datetime | None = None

    # Время, до которого тянутся траектории
    t_k: datetime | None = None

    def __init__(
            self,
            N: int,
            sigma_points: np.ndarray | None = None,
            measurements: pd.DataFrame | None = None,
            t_start: datetime | None = None,
            t_k: datetime | None = None
        ) -> None:
        """
        Args:
            N: Количество сигма точек.
            sigma_points: набор сигма точек.
            measurements: набор измерений.
            t_start: начальный момент времени.
            t_k: время, до которого протягиваем орбиту.
            
        """
        self.N = N
        self.sigma_points = sigma_points
        self.measure = measurements
        self.t_start = t_start
        self.t_cur = t_k


    def to_pyorbs_orbits(self) -> list[pyorbs.pyorbs.orbit]:
        """ Функция, возвращающая список орбит в формате pyorbs
        """
        orb_list = []
        for i in range (self.N):
            orb = pyorbs.pyorbs.orbit(
                x = self.sigma_points[i, :6], time = self.t_start
            )
            orb_list.append(orb)

        return orb_list

    def set_residuals(self) -> np.ndarray:
        """ Выдает невязки по измерениям для всех сигма точек
        в момент времени t_cur, взятый из таблицы измерений.
        """
        orbits = self.to_pyorbs_orbits()
        fixed_meas = ft.partial(process_single_orbit, self.measure)

        with mp.Pool(processes = mp.cpu_count()) as pool:
            dz = pool.map(fixed_meas, orbits)

        return np.array(dz)
    
    def find_Z_from_dz(self, w, dz):
        W = np.tile(w, (self.N, 1))
        A = np.eye(self.N) - W
        Z = np.linalg.solve(A, dz)

        Z = Z % (2 * math.pi)
        if (Z[:,1] > math.pi).any():
            Z[:,1] -= 2 * math.pi
        #print(f'Z = {Z}')
        return Z


class UKF:
    """ Класс, реализующий сигматочечный фильтр Калмана
    """
    # Ковариационнная матрица вектора состояния 
    cov_matrix: np.ndarray

    # Время, с которого начинается фильтрация
    t_start: datetime

    # Вектор состояния
    state: np.ndarray

    # Ковариационная матрица процесса
    Q: np.ndarray | None = None

    # Ковариационная матрица измерений
    R: np.ndarray | None = None

    # Параметр отвечающий за разброс от вектора 
    # состояния. Нужен для определения параметра лямбда
    alpha: float = None

    # Второй параметр для определения лямбда
    kappa: float = None

    # Параметр для инициализации веса ковариации
    beta: float | None = None

    # Размерность вектора динамической системы
    dim_x: int = None

    # Параметр для разложения Холесского
    par_lambda: float | None = None

    # Матрица преобразованных сигма точек
    Y: np.ndarray = None

    # Невязка измерения
    res_z: float = None

    def __init__(
            self, 
            P: np.ndarray,
            t_start: datetime,
            x: np.ndarray,
            Q: np.ndarray | None,
            R: np.ndarray | None,
            alpha: float = _config.DEFAULT_ALPHA,
            beta: float =  _config.DEFAULT_BETA,
            kappa: float = _config.DEFAULT_KAPPA
        ) -> None:
            """Конструктор сигматочечного фильтра Калмана
            
            Args:
                P: Ковариационная матрица вектора состояния.
                t_start: Время начала предсказания.
                x: Вектор состояния.
                Q: Ковариационная матрица процесса.
                R: Ковариационная матрица наблюдения.
                alpha: Параметр для опредения параметра /lambda для вычисления корня в разложении Холесского.
                beta: Параметр для инициализации веса w0 ковариации.
                kappa: Параметр для опредения параметра /lambda для вычисления корня в разложении Холесского.

            """
            self.cov_matrix = P
            self.t_start = t_start
            self.state = x
            self.Q = Q
            self.R = R
            self.alpha = alpha
            self.beta = beta
            self.kappa = kappa

            self.dim_x = len(x)
            self.par_lambda = alpha**2 * (self.dim_x + kappa) - self.dim_x

            self.set_weights()

            self.Y = np.zeros((2 * self.dim_x + 1, self.dim_x))


    def set_weights(self):
        n_sigma = 2 * self.dim_x + 1
        self.w_mean = np.full(n_sigma, 1.0 / (2.0 * (self.dim_x + self.par_lambda)))
        self.w_cov = self.w_mean.copy()

        self.w_mean[0] = self.par_lambda / (self.dim_x + self.par_lambda)
        self.w_cov[0] = self.w_mean[0] + (1 - self.alpha**2 + self.beta)

    def generate_sigma_points(self) -> np.ndarray:
        n = self.dim_x
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = self.state

        try:
            L = cholesky((n + self.par_lambda) * self.cov_matrix)
        except np.linalg.LinAlgError:
            L = cholesky((n + self.par_lambda) * self.cov_matrix + np.eye(n) * 1e-5)

        for i in range(n):
            sigma_points[i+1] = self.state + L[:, i]
            sigma_points[i+1+n] = self.state - L[:, i]

        return sigma_points
    
    def prediction(self, t_k: pd.Timestamp):
        n = self.dim_x

        # 1. Создание сигма-точек:
        sigma_points = self.generate_sigma_points()

        # 2. Протягиваем сигма-точки по эволюции с помощью пакета pyorbs:
        for i in range (2 * n + 1):
            orb = pyorbs.pyorbs.orbit(x = sigma_points[i, :6], time = self.t_start)
            orb.setup_parameters()
            orb.move(t_k)
            self.Y[i, :6] = orb.state_v

        # 3. Предсказываем среднее и ковариационную матрицу:
        y_mean = self.w_mean @ sigma_points
        P_y = (self.w_cov[:, None, None] * (
               (self.Y-y_mean)[:, :, None] @ (self.Y-y_mean)[:, None, :])
               ).sum(axis = 0) + self.Q

        return y_mean, P_y
    
    def correction(self, z: pd.DataFrame, y_mean: np.ndarray,
                    P_y: np.ndarray, t_k: pd.Timestamp):
        """
        Args:
            z: таблица с измерениями из класса ContextOD.
            y_mean: предсказанный вектор состояния в момент времени t_k.
            P_y: предсказанная ковариационная матрица вектора 
            состояния в момент времени t_k.
            t_k: время, на которое предсказываем и корректируем данные.
        """

        # 4. Создаем класс траекторий всех сигма точек и тянем их по
        #    моментам времени, когда проводились измерения.
        #    Возвращаем невязки по этим измерениям с помощью
        #    пакета pyorbs:

        sigma_points = self.generate_sigma_points()

        traj = Trajectories(N = 2 * self.dim_x + 1, sigma_points = sigma_points,
                            measurements = z, t_start = self.t_start, t_k = t_k)
        
        dz = traj.set_residuals()
        dz = dz[:, :, 0]
        Z = traj.find_Z_from_dz(self.w_mean, dz)

        # 5. Вычисляем предсказанное среднее и предсказанную
        #    ковариационную матрицу:
        z_mean = self.w_mean @ Z
        P_z = (self.w_cov[:, None, None] * (
               dz[:, :, None] @ dz[:, None, :])
               ).sum(axis = 0) + self.R

        # 6. Вычисляем перекрестную ковариацию:
        P_yz = (self.w_cov[:, None, None] * (
               (self.Y-y_mean)[:, :, None] @ (dz)[:, None, :])
               ).sum(axis = 0)
        
        # 7. Вычисляем матрицу усиления:
        try:
            K = P_yz @ np.linalg.inv(P_z)
        except np.linalg.LinAlgError:
            K = P_yz @ np.linalg.pinv(P_z)

        # 8. Вычисляем невязку по измерениям:
        self.res_z = z['val'] - z_mean
        #print(f'z_val = {z['val']}, z_mean = {z_mean}')
        #print(f'res_z = {self.res_z}')

        # 9. Корректируем вектор состояния и ковариационную матрицу:
        self.state = y_mean + K @ self.res_z.T
        self.cov_matrix = P_y - K @ P_z @ K.T

        return self.state, self.cov_matrix

    def smoothing(self): # функция сглаживания
        pass

    def step(self, z, t_k):
        """ Шаг фильтрации"""
        y_mean, P_y = self.prediction(t_k)
        return self.correction(z, y_mean, P_y, t_k)