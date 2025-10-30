import numpy as np
import pandas as pd
from typing import List
from scipy.linalg import cholesky
from datetime import datetime
import multiprocessing as mp
import functools as ft

import pyorbs
import _config

def process_single_orbit(m: pd.DataFrame, 
                         orbit: pyorbs.pyorbs.orbit) -> np.ndarray[np.float64]:
    """ Вспомогательная функция для протягивания одной траектории,
    возвращает невязку в данный момент времени.
    """
    meas = pyorbs.pyorbs_det.process_meas_record(orbit, m, 6, None) # setup parameters
    res = meas['res']

    return res


class Trajectories:
    """ Класс сигма-точек, протягивающий орбиту по измерениям
        для каждой точки.
    """
    
    # Количество сигма точек
    N: int

    # Набор сигма точек
    sigma_points: np.ndarray[np.float64] | None = None

    # Набор измерений
    measurements: pd.DataFrame | None = None

    # Время начала фильтрации
    t_start: datetime | None = None

    # Время, до которого тянутся траектории
    t_k: datetime | None = None

    def __init__(
            self,
            N: int,
            sigma_points: np.ndarray[np.float64] | None = None,
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

    def set_residuals(self) -> np.ndarray[np.float64]:
        """ Выдает невязки по измерениям для всех сигма точек
        в момент времени t_cur, взятый из таблицы измерений.
        """
        orbits = self.to_pyorbs_orbits()
        fixed_meas = ft.partial(process_single_orbit, self.measure)

        with mp.Pool(processes = mp.cpu_count()) as pool:
            dz = pool.map(fixed_meas, orbits)

        return np.array(dz)


class UKF:
    """ Класс, реализующий сигматочечный фильтр Калмана.
        Фильтрация вектора состояния и его ковариационной 
        матрицы по массиву измерений, взятых из ContextOD.
    """
    # Ковариационнная матрица вектора состояния 
    cov_matrix: np.ndarray[np.float64]

    # Время, с которого начинается фильтрация
    t_start: datetime | None

    # Вектор состояния
    state_v: np.ndarray[np.float64]

    # Ковариационная матрица процесса
    Q: np.ndarray[np.float64] | None = None

    # Ковариационная матрица измерений
    R: np.ndarray[np.float64] | None = None

    # Параметр отвечающий за разброс от вектора 
    # состояния. Нужен для определения параметра лямбда
    alpha: float

    # Второй параметр для определения лямбда
    kappa: float

    # Параметр для инициализации веса ковариации
    beta: float | None = None

    # Размерность вектора состояния динамической системы
    dim_x: int

    # Параметр для разложения Холесского
    par_lambda: float | None = None

    # Матрица преобразованных сигма точек
    Y: np.ndarray[np.float64]

    # Массив сигма точек
    sigma_points: np.ndarray[np.float64]

    def __init__(
            self, 
            P: np.ndarray[np.float64],
            t_start: datetime,
            x: np.ndarray[np.float64],
            R: np.ndarray = _config.R_DEFAULT,
            alpha: float = _config.DEFAULT_ALPHA,
            beta: float =  _config.DEFAULT_BETA,
            kappa: float = _config.DEFAULT_KAPPA,
            dim_x: int = _config.DEFAULT_DIMENSION
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
            self.state_v = x
            self.Q = np.zeros((dim_x, dim_x))
            self.alpha = alpha
            self.beta = beta
            self.kappa = kappa

            self.dim_x = dim_x
            self.par_lambda = alpha**2 * (self.dim_x + kappa) - self.dim_x

            self.set_weights()

            self.Y = np.zeros((2 * self.dim_x + 1, self.dim_x))


    def set_weights(self):
        n_sigma = 2 * self.dim_x + 1
        self.w_mean = np.full(n_sigma, 1.0 / (2.0 * (self.dim_x + self.par_lambda)))
        self.w_cov = self.w_mean.copy()

        self.w_mean[0] = self.par_lambda / (self.dim_x + self.par_lambda)
        self.w_cov[0] = self.w_mean[0] + (1 - self.alpha**2 + self.beta)

    def generate_sigma_points(self) -> np.ndarray[np.float64]:
        """
            Создает массив сигма точек вектора состояния,
            образующий окрестность.  
        """
        n = self.dim_x
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = self.state_v

        try:
            L = cholesky((n + self.par_lambda) * self.cov_matrix)
        except np.linalg.LinAlgError:
            L = cholesky((n + self.par_lambda) * self.cov_matrix + np.eye(n) * 1e-5)

        for i in range(n):
            sigma_points[i+1] = self.state_v + L[:, i]
            sigma_points[i+1+n] = self.state_v - L[:, i]

        return sigma_points
    
    def prediction(self, t_k: pd.Timestamp) \
          -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
        n = self.dim_x

        # 1. Создание сигма-точек:
        self.sigma_points = self.generate_sigma_points()

        # 2. Протягиваем сигма-точки по эволюции с помощью пакета pyorbs:
        for i in range (2 * n + 1):
            orb = pyorbs.pyorbs.orbit(x = self.sigma_points[i, :6], time = self.t_start)
            orb.setup_parameters()
            orb.move(t_k)
            self.Y[i, :6] = orb.state_v

        # 3. Предсказываем среднее и ковариационную матрицу:
        y_mean = self.w_mean @ self.Y
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
        traj = Trajectories(N = 2 * self.dim_x + 1, 
                            sigma_points = self.sigma_points,
                            measurements = z,
                            t_start = self.t_start, t_k = t_k)
        
        dz = traj.set_residuals()[:, :, 0]
        Z = z['val'] - dz

        # 5. Вычисляем предсказанное среднее и предсказанную
        #    ковариационную матрицу:
        z_mean = self.w_mean @ Z
        P_z = (self.w_cov[:, None, None] * (
               (Z - z_mean)[:, :, None] @ (Z - z_mean)[:, None, :])
               ).sum(axis = 0) + self.R

        # 6. Вычисляем перекрестную ковариацию:
        P_yz = (self.w_cov[:, None, None] * (
               (self.Y-y_mean)[:, :, None] @ (Z - z_mean)[:, None, :])
               ).sum(axis = 0)
        
        # 7. Вычисляем матрицу усиления:
        try:
            K = P_yz @ np.linalg.inv(P_z)
        except np.linalg.LinAlgError:
            K = P_yz @ np.linalg.pinv(P_z)

        # 8. Вычисляем невязку по измерениям:
        res_z = z['val'] - z_mean
        #print(z['val'], z_mean)
        # 9. Корректируем вектор состояния и ковариационную матрицу:
        self.state_v = y_mean + K @ res_z.T
        self.cov_matrix = P_y - K @ P_z @ K.T

        return self.state_v, self.cov_matrix

    def step(self, z, t_k):
        """ Шаг фильтрации"""
        y_mean, P_y = self.prediction(t_k)
        return self.correction(z, y_mean, P_y, t_k)


def smoothing(ukf: UKF, forward_states: List[np.float64], 
              forward_covs: List[np.float64], meas: pd.DataFrame) \
                    -> tuple[List[np.float64], List[np.float64]]:
    """Процесс сглаживания оценок вектора состояния 
    и ковариационной матрицы.
    """
    pred_states = []
    pred_covs = []

    for m in meas.iterrows():
        t = m['time']
        y_mean, P_y = ukf.prediction(t)
        pred_states.append(y_mean)
        pred_covs.append(P_y)

    smoothed_states = forward_states.copy()
    smoothed_covs = forward_covs.copy()

    for k in range (len(forward_states)-2, -1, -1):
        C_k = forward_covs[k] @ np.linalg.inv(pred_covs[k+1])

        smoothed_states += C_k @ (smoothed_states[k+1] - pred_states[k+1])
        smoothed_covs += C_k @ (smoothed_covs[k+1] - pred_covs[k+1])
    
    return smoothed_states, smoothed_covs