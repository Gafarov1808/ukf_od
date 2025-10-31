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
                         orbit: pyorbs.pyorbs.orbit) ->  pd.Series | np.ndarray | None:
    """ Вспомогательная функция для протягивания одной траектории,
    возвращает невязку в данный момент времени.
    """
    orbit.setup_parameters()
    meas: pd.Series | pd.DataFrame | None = pyorbs.pyorbs_det.process_meas_record(orbit, m, 6, None)
    
    if meas is not None:
        res: pd.Series | np.ndarray = meas['res']
        return res

    return


class Trajectories:
    """ Класс сигма-точек, протягивающий орбиту по измерениям
        для каждой точки.
    """
    
    # Количество сигма точек
    amount_points: int

    # Набор сигма точек
    sigma_points: np.ndarray

    # Набор измерений
    measurements: pd.DataFrame 

    # Время начала фильтрации
    t_start: pyorbs.pyorbs.ephem_time

    # Время, до которого тянутся траектории
    t_k: datetime | None = None

    def __init__(
            self,
            amount_points: int,
            sigma_points: np.ndarray,
            measurements: pd.DataFrame,
            t_start: pyorbs.pyorbs.ephem_time,
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
        self.N = amount_points
        self.sigma_points = sigma_points
        self.measure = measurements
        self.t_start = t_start
        self.t_cur = t_k


    def to_pyorbs_orbits(self) -> list[pyorbs.pyorbs.orbit]:
        """ Функция, возвращающая список орбит в формате pyorbs
        """
        orb_list: List[pyorbs.pyorbs.orbit] = []

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


class UKF:
    """ Класс, реализующий сигматочечный фильтр Калмана.
        Фильтрация вектора состояния и его ковариационной 
        матрицы по массиву измерений, взятых из ContextOD.
    """
    # Ковариационнная матрица вектора состояния 
    cov_matrix: np.ndarray

    # Время, с которого начинается фильтрация
    t_start: pyorbs.pyorbs.ephem_time

    # Вектор состояния
    state_v: np.ndarray

    # Ковариационная матрица процесса
    cov_process_matrix: np.ndarray

    # Ковариационная матрица измерений
    cov_matrix_measure: np.ndarray | None = None

    # Параметр отвечающий за разброс от вектора 
    # состояния. Нужен для определения параметра лямбда
    alpha: float

    # Второй параметр для определения лямбда
    kappa: float

    # Параметр для инициализации веса ковариации
    beta: float

    # Размерность вектора состояния динамической системы
    dim_x: int

    # Параметр для разложения Холесского
    par_lambda: float

    # Матрица преобразованных сигма точек
    transform_points: np.ndarray

    # Массив сигма точек
    sigma_points: np.ndarray

    def __init__(
            self, 
            P: np.ndarray,
            t_start: pyorbs.pyorbs.ephem_time,
            x: np.ndarray,
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
            self.cov_process_matrix = np.zeros((dim_x, dim_x))
            self.cov_matrix_measure = R
            self.alpha = alpha
            self.beta = beta
            self.kappa = kappa

            self.dim_x = dim_x
            self.par_lambda = alpha**2 * (self.dim_x + kappa) - self.dim_x

            self.set_weights()

            self.transform_points = np.zeros((2 * self.dim_x + 1, self.dim_x))


    def set_weights(self):
        n_sigma = 2 * self.dim_x + 1
        self.w_mean = np.full(n_sigma, 1.0 / (2.0 * (float(self.dim_x) + self.par_lambda)))
        self.w_cov = self.w_mean.copy()

        self.w_mean[0] = self.par_lambda / (self.dim_x + self.par_lambda)
        self.w_cov[0] = self.w_mean[0] + (1 - self.alpha**2 + self.beta)

    def generate_sigma_points(self) -> np.ndarray:
        """
            Создает массив сигма точек вектора состояния,
            образующий окрестность.  
        """
        n = self.dim_x
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = self.state_v

        try:
            SR_cov_matrix = cholesky((n + self.par_lambda) * self.cov_matrix)
        except np.linalg.LinAlgError:
            SR_cov_matrix = cholesky((n + self.par_lambda) * self.cov_matrix + np.eye(n) * 1e-5)

        for i in range(n):
            sigma_points[i+1] = self.state_v + SR_cov_matrix[:, i]
            sigma_points[i+1+n] = self.state_v - SR_cov_matrix[:, i]

        return sigma_points
    
    def prediction(self, t_k: pd.Timestamp) \
          -> tuple[np.ndarray, np.ndarray]:
        n = self.dim_x

        # 1. Создание сигма-точек:
        self.sigma_points = self.generate_sigma_points()

        # 2. Протягиваем сигма-точки по эволюции с помощью пакета pyorbs:
        for i in range (2 * n + 1):
            orb = pyorbs.pyorbs.orbit(x = self.sigma_points[i, :6], time = self.t_start)
            orb.setup_parameters()
            orb.move(t_k)
            self.transform_points[i, :6] = orb.state_v

        # 3. Предсказываем среднее и ковариационную матрицу:
        y_mean = self.w_mean @ self.transform_points
        P_y = (self.w_cov[:, None, None] * (
               (self.transform_points-y_mean)[:, :, None] @ (self.transform_points-y_mean)[:, None, :])
               ).sum(axis = 0) + self.cov_process_matrix

        return y_mean, P_y
    
    def correction(self, z: pd.DataFrame, y_mean: np.ndarray,
                    P_y: np.ndarray, t_k: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
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
        traj = Trajectories(amount_points = 2 * self.dim_x + 1, 
                            sigma_points = self.sigma_points,
                            measurements = z,
                            t_start = self.t_start, t_k = t_k)
        
        dz = traj.set_residuals()[:, :, 0]
        Z = z['val'] - dz

        # 5. Вычисляем предсказанное среднее и предсказанную
        #    ковариационную матрицу:
        z_mean = self.w_mean @ Z
        P_z: np.ndarray = (self.w_cov[:, None, None] * (
               (Z - z_mean)[:, :, None] @ (Z - z_mean)[:, None, :])
               ).sum(axis = 0) + self.cov_matrix_measure

        # 6. Вычисляем перекрестную ковариацию:
        P_yz: np.ndarray = (self.w_cov[:, None, None] * (
               (self.transform_points-y_mean)[:, :, None] @ (Z - z_mean)[:, None, :])
               ).sum(axis = 0)
        
        # 7. Вычисляем матрицу усиления:
        try:
            Kalman_gain = P_yz @ np.linalg.inv(P_z)
        except np.linalg.LinAlgError:
            Kalman_gain = P_yz @ np.linalg.pinv(P_z)

        # 8. Вычисляем невязку по измерениям:
        res_z = z['val'] - z_mean

        # 9. Корректируем вектор состояния и ковариационную матрицу:
        self.state_v = y_mean + Kalman_gain @ res_z.T
        self.cov_matrix = P_y - Kalman_gain @ P_z @ Kalman_gain.T
        self.t_start = t_k

        return self.state_v, self.cov_matrix

    def step(self, z: pd.DataFrame, t_k: pd.Timestamp):
        """ Шаг фильтрации"""
        y_mean, P_y = self.prediction(t_k)
        return self.correction(z, y_mean, P_y, t_k)


def smoothing(ukf: UKF, forward_states: List[np.float64], 
              forward_covs: List[np.float64], meas: pd.DataFrame) \
                    -> tuple[List[np.float64], List[np.float64]]:
    """Процесс сглаживания оценок вектора состояния 
    и ковариационной матрицы.
    """
    
    print(f'Процесс сглаживания начался...')
    pred_states = []
    pred_covs = []
    for _, m in meas.iterrows():
        t = m.loc['time']
        ukf.t_start = pyorbs.pyorbs.ephem_time(data = t.to_pydatetime())
        y_mean, P_y = ukf.prediction(t)
        pred_states.append(y_mean)
        pred_covs.append(P_y)

    smoothed_states = forward_states.copy()
    smoothed_covs = forward_covs.copy()

    for k in range (len(forward_states)-2, -1, -1):
        G_k = forward_covs[k] @ np.linalg.inv(pred_covs[k+1]) # forward_cows?

        smoothed_states = forward_states[k] + G_k @ (smoothed_states[k+1] - pred_states[k+1])
        smoothed_covs = forward_covs[k] + G_k @ (smoothed_covs[k+1] - pred_covs[k+1]) @ G_k.T
    
    return smoothed_states, smoothed_covs