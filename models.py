import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from scipy.linalg import cholesky
from math import *

import pyorbs
import _config

class Trajectories:
    """ Класс сигма-точек, протягивающий орбиту по измерениям
        для каждой точки.
    """
    
    #: Количество сигма точек
    amount_points: int

    #: Набор сигма точек
    sigma_points: np.ndarray

    #: Набор измерений
    measure: pd.DataFrame

    #: Время начала фильтрации
    t_start: pyorbs.pyorbs.ephem_time

    #: Время, до которого протягиваем орбиту
    t_k: pyorbs.pyorbs.ephem_time

    #: Список орбит сигма точек
    orb_list: list[pyorbs.pyorbs.orbit]
    
    def __init__(
            self,
            amount_points: int,
            sigma_points: np.ndarray,
            t_start: pyorbs.pyorbs.ephem_time,
            t_k: pyorbs.pyorbs.ephem_time,
            measure: pd.DataFrame | None = None
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
        self.measure = measure
        self.t_start = t_start
        self.t_k = t_k
        self.orb_list = []


    def get_transform_sigma_points(self):
        transform_points = np.zeros((self.N, (self.N-1) // 2))
        
        for i in range (self.N):
            orb = pyorbs.pyorbs.orbit(x = self.sigma_points[i], time = self.t_start)
            orb.setup_parameters()
            orb.move(self.t_k)
            transform_points[i, :6] = orb.state_v
            self.orb_list.append(orb)
        return transform_points

    def set_residuals(self):
        """ Выдает невязки по измерениям для всех сигма точек
        в момент времени t_cur, взятый из таблицы измерений.
        """
        dz = []

        for orbit in self.orb_list:
            orbit.setup_parameters()
            m = pyorbs.pyorbs_det.process_meas_record(orbit, self.measure, 6, None)
            dz.append(m['res'] / 3600. * _config._DEG2RAD)

        return np.array(dz)


class UKF:
    """ Класс, реализующий сигматочечный фильтр Калмана.
        Фильтрация вектора состояния и его ковариационной 
        матрицы по массиву измерений, взятых из ContextOD.
    """
    #: Ковариационнная матрица вектора состояния 
    cov_matrix: np.ndarray

    #: Время, с которого начинается фильтрация
    t_start: pyorbs.pyorbs.ephem_time

    #: Вектор состояния
    state_v: np.ndarray

    #: Ковариационная матрица процесса
    cov_process_matrix: np.ndarray

    #: Ковариационная матрица измерений
    cov_matrix_measure: np.ndarray | None = None

    #: Параметр отвечающий за разброс от вектора 
    # состояния. Нужен для определения параметра лямбда
    alpha: float

    #: Второй параметр для определения лямбда
    kappa: float

    #: Параметр для инициализации веса ковариации
    beta: float

    #: Размерность вектора состояния динамической системы
    dim_x: int

    #: Параметр для разложения Холесского
    par_lambda: float

    #: Матрица преобразованных сигма точек
    transform_points: np.ndarray

    #: Массив сигма точек
    sigma_points: np.ndarray

    #: Список предсказанных векторов состояния
    pred_states: List = []

    #: Список предсказанных ковариационных матриц 
    pred_covs: List = []

    #: Список скорректированных векторов состояния
    forward_states: List = []

    #: Список скорректированных ковариационннных матриц
    forward_covs: List = []

    #: Cписок моментов времени, в которые фильтруем данные
    times: List = []

    #: Список всех сигма точек
    all_sigma_points: List = []

    #: Список всех сигма точек, протянутых в моменты времени из списка times
    all_transform_points: List = []

    def __init__(
            self, 
            P: np.ndarray,
            t_start: pyorbs.pyorbs.ephem_time,
            x: np.ndarray,
            R: np.ndarray = _config.R_DEFAULT,
            alpha: float = _config.DEFAULT_ALPHA,
            beta: float =  _config.DEFAULT_BETA,
            kappa: float = _config.DEFAULT_KAPPA,
            dim_x: int = _config.DEFAULT_DIMENSION,
            pred_states: List = [],
            pred_covs: List = [],
            forward_states: List = [],
            forward_covs: List = [],
            times: List = [],
            sigma_points: List = [],
            all_transform_points: List = []
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
            self.pred_states = pred_states
            self.pred_covs = pred_covs
            self.forward_states = forward_states
            self.forward_covs = forward_covs
            self.times = times
            self.sigma_points = sigma_points
            self.all_transform_points = all_transform_points


    def set_weights(self) -> None:
        n_sigma = 2 * self.dim_x + 1
        self.w_mean = np.full(n_sigma, 1.0 / (2.0 * (float(self.dim_x) + self.par_lambda)))
        self.w_cov = self.w_mean.copy()

        self.w_mean[0] = self.par_lambda / (self.dim_x + self.par_lambda)
        self.w_cov[0] = self.w_mean[0] + (1 - self.alpha**2 + self.beta)

    def generate_sigma_points(self) -> np.ndarray:
        """ Создает массив сигма точек вектора состояния,
            образующий окрестность."""
        sigma_points = np.zeros((2 * self.dim_x + 1, self.dim_x))
        sigma_points[0] = self.state_v

        SR_cov = cholesky((self.dim_x + self.par_lambda) * self.cov_matrix)
        for i in range(self.dim_x):
            sigma_points[i+1] = self.state_v + SR_cov[:, i]
            sigma_points[i+1+self.dim_x] = self.state_v - SR_cov[:, i]

        return sigma_points
   
    def prediction(self, t_k: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        # 1. Создание сигма-точек:
        self.sigma_points = self.generate_sigma_points()
        self.all_sigma_points.append(self.sigma_points)

        # 2. Протягиваем сигма-точки по эволюции с помощью пакета pyorbs:
        traj = Trajectories(amount_points = 2 * self.dim_x + 1,
                            sigma_points = self.sigma_points,
                            t_start = self.t_start,  t_k = t_k)
        
        self.transform_points = traj.get_transform_sigma_points()
        self.all_transform_points.append(self.transform_points)

        # 3. Предсказываем среднее и ковариационную матрицу.
        pred_state = self.w_mean @ self.transform_points

        pred_cov = np.zeros((self.dim_x, self.dim_x))
        for i in range (2 * self.dim_x + 1):
            diff = (self.transform_points[i] - pred_state).reshape(-1,1)
            pred_cov += self.w_cov[i] * diff @ diff.T
        pred_cov += self.cov_process_matrix

        self.write_pred_data(pred_state, pred_cov)
        self.times.append(t_k)

        return pred_state, pred_cov, traj

    def correction(self, z: pd.DataFrame, pred_state: np.ndarray,
                    pred_cov: np.ndarray, t_k: pd.Timestamp, traj) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            z: таблица с измерениями из класса ContextOD.
            y_mean: предсказанный вектор состояния в момент времени t_k.
            P_y: предсказанная ковариационная матрица вектора 
            состояния в момент времени t_k.
            t_k: время, на которое предсказываем и корректируем данные.
        """
        a_priori_meas = z['val']
        # 4. Возвращаем невязки по измерениям с помощью
        #    пакета pyorbs:
        traj.measure = z
        res_meas = traj.set_residuals()[:, :, 0]
        calc_measure = a_priori_meas - res_meas 
        
        # 5. Вычисляем предсказанное среднее и предсказанную
        #    ковариационную матрицу:
        pred_meas = self.w_mean @ calc_measure
        Pz = (self.w_cov[:, None, None] * (
               (calc_measure - pred_meas)[:, :, None] @
               (calc_measure - pred_meas)[:, None, :])
               ).sum(axis = 0) + self.cov_matrix_measure

        # 6. Вычисляем перекрестную ковариацию:
        Pyz = (self.w_cov[:, None, None] * (
                (self.sigma_points - pred_state)[:, :, None] @
                (calc_measure - pred_meas)[:, None, :]
                )).sum(axis = 0)
        #print(self.sigma_points - pred_state)
        # 7. Вычисляем матрицу усиления:
        try:
            Kalman_gain = Pyz @ np.linalg.inv(Pz)
        except np.linalg.LinAlgError:
            Kalman_gain = Pyz @ np.linalg.pinv(Pz)

        # 8. Вычисляем невязку по измерениям:
        res_z = a_priori_meas - pred_meas

        # 9. Корректируем вектор состояния и ковариационную матрицу
        # и регуляризируем ее, если необходимо:
        self.state_v = pred_state + Kalman_gain @ res_z.T
        self.cov_matrix = pred_cov - Kalman_gain @ Pz @ Kalman_gain.T

        if (np.linalg.eigvals(self.cov_matrix) > 0).all() == False:
            print('Регуляризация скорректированной ковариационной матрицы')
            self.cov_matrix += np.eye(6) * 1e-7

        self.forward_states.append(self.state_v)
        self.forward_covs.append(self.cov_matrix)
        self.t_start = t_k
        return self.state_v, self.cov_matrix

    def step(self, z: pd.DataFrame, t_k: pd.Timestamp):
        """ Шаг фильтрации."""
        pred_state, pred_cov, traj = self.prediction(t_k)
        return self.correction(z, pred_state, pred_cov, t_k, traj)

    def write_pred_data(self, y_mean, P_y):
        """ Записывает предсказанные данные"""
        self.pred_states.append(y_mean)
        self.pred_covs.append(P_y)

    def rts_smoother(self) -> tuple[List[np.float64], List[np.float64]]:
        """Процесс сглаживания оценок вектора состояния 
        и ковариационной матрицы (Unscented Rauch-Tung-Striebel smoother
        for the additive dynamic system).
        """
        
        print(f'Cглаживание...')
        smoothed_states = self.forward_states.copy()
        smoothed_covs = self.forward_covs.copy()
        
        for k in range (len(self.forward_states)-2, -1, -1):
            cross_cov = np.zeros((self.dim_x, self.dim_x))
            for i in range (2 * self.dim_x + 1):
                dif = (self.all_transform_points[k][i] - self.forward_states[k]).reshape(-1,1)
                dif_pred = (self.all_transform_points[k+1][i] - self.pred_states[k+1]).reshape(-1, 1)
                cross_cov += self.w_cov[i] * (dif @ dif_pred.T)

            try:
                G = cross_cov @ np.linalg.inv(self.pred_covs[k+1])
            except:
                G = cross_cov @ np.linalg.pinv(self.pred_covs[k+1])

            smoothed_states[k] = self.forward_states[k] + G @ (smoothed_states[k+1] - self.pred_states[k+1])
            smoothed_covs[k] = self.forward_covs[k] + G @ (smoothed_covs[k+1] - self.pred_covs[k+1]) @ G.T

        return smoothed_states, smoothed_covs

    def draw_plots(self, smoothed_states, smoothed_covs):

        sigma_x = [sqrt(arr[0,0]) for arr in smoothed_covs]
        sigma_y = [sqrt(arr[1,1]) for arr in smoothed_covs]
        sigma_z = [sqrt(arr[2,2]) for arr in smoothed_covs]

        sigma_vx = [sqrt(arr[3,3]) for arr in smoothed_covs]
        sigma_vy = [sqrt(arr[4,4]) for arr in smoothed_covs]
        sigma_vz = [sqrt(arr[5,5]) for arr in smoothed_covs]
        
        plt.figure('Положение', figsize=(19,10))

        plt.subplot(3, 1, 1)
        plt.plot(self.times, sigma_x, '+')
        plt.grid(True)
        plt.title('СКО х')
        plt.xlabel('Время')
        plt.ylabel('$\sigma_x$')

        plt.subplot(3, 1, 2)
        plt.plot(self.times, sigma_y, '+')
        plt.grid(True)
        plt.title('СКО y')
        plt.xlabel('Время')
        plt.ylabel('$\sigma_y$')

        plt.subplot(3, 1, 3)
        plt.plot(self.times, sigma_z, '+')
        plt.grid(True)
        plt.title('СКО z')
        plt.xlabel('Время')
        plt.ylabel('$\sigma_z$')

        plt.tight_layout()
        plt.show()


        plt.figure('Скорость', figsize=(19,10))

        plt.subplot(3, 1, 1)
        plt.plot(self.times, sigma_vx, '+')
        plt.grid(True)
        plt.title('СКО $V_x$')
        plt.xlabel('Время')
        plt.ylabel('$\sigma_{V_x}$')

        plt.subplot(3, 1, 2)
        plt.plot(self.times, sigma_vy, '+')
        plt.grid(True)
        plt.title('СКО $V_y$')
        plt.xlabel('Время')
        plt.ylabel('$\sigma_{V_y}$')

        plt.subplot(3, 1, 3)
        plt.plot(self.times, sigma_vz, '+')
        plt.grid(True)
        plt.title('СКО $V_z$')
        plt.xlabel('Время')
        plt.ylabel('$\sigma_{V_z}$')

        plt.tight_layout()
        plt.show()


class SquareRootUKF:
    """ Класс, реализующий сигматочечный фильтр Калмана.
        Фильтрация вектора состояния и его ковариационной 
        матрицы по массиву измерений, взятых из ContextOD.
    """
    #: Ковариационнная матрица вектора состояния 
    cov_matrix: np.ndarray

    #: Квадратный корень ковариационной матрицы
    SR_cov: np.ndarray

    #: Время, с которого начинается фильтрация
    t_start: pyorbs.pyorbs.ephem_time

    #: Вектор состояния
    state_v: np.ndarray

    #: Ковариационная матрица процесса
    cov_process_matrix: np.ndarray

    #: Ковариационная матрица измерений
    cov_matrix_measure: np.ndarray | None = None

    #: Параметр отвечающий за разброс от вектора 
    # состояния. Нужен для определения параметра лямбда
    alpha: float

    #: Второй параметр для определения лямбда
    kappa: float

    #: Параметр для инициализации веса ковариации
    beta: float

    #: Размерность вектора состояния динамической системы
    dim_x: int

    #: Параметр для разложения Холесского
    par_lambda: float

    #: Матрица преобразованных сигма точек
    transform_points: np.ndarray

    #: Массив сигма точек
    sigma_points: np.ndarray

    #: Список предсказанных векторов состояния
    pred_states: List = []

    #: Список предсказанных ковариационных матриц 
    pred_covs: List = []

    #: Список скорректированных векторов состояния
    forward_states: List = []

    #: Список скорректированных ковариационннных матриц
    forward_covs: List = []

    #: Cписок моментов времени, в которые фильтруем данные
    times: List = []

    #: Список всех сигма точек
    all_sigma_points: List = []

    #: Список всех сигма точек, протянутых в моменты времени из списка times
    all_transform_points: List = []

    def __init__(
            self, 
            P: np.ndarray,
            t_start: pyorbs.pyorbs.ephem_time,
            x: np.ndarray,
            R: np.ndarray = _config.R_DEFAULT,
            alpha: float = _config.DEFAULT_ALPHA,
            beta: float =  _config.DEFAULT_BETA,
            kappa: float = _config.DEFAULT_KAPPA,
            dim_x: int = _config.DEFAULT_DIMENSION,
            pred_states: List = [],
            pred_covs: List = [],
            forward_states: List = [],
            forward_covs: List = [],
            times: List = [],
            sigma_points: List = [],
            all_transform_points: List = []
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
            self.cov_process_matrix = _config.DEFAULT_COV_PROCESS
            self.cov_matrix_measure = R
            self.alpha = alpha
            self.beta = beta
            self.kappa = kappa

            self.dim_x = dim_x
            self.par_lambda = alpha**2 * (self.dim_x + kappa) - self.dim_x
            self.SR_cov = cholesky(P)

            self.set_weights()

            self.transform_points = np.zeros((2 * self.dim_x + 1, self.dim_x))
            self.pred_states = pred_states
            self.pred_covs = pred_covs
            self.forward_states = forward_states
            self.forward_covs = forward_covs
            self.times = times
            self.sigma_points = sigma_points
            self.all_transform_points = all_transform_points


    def set_weights(self) -> None:
        n_sigma = 2 * self.dim_x + 1
        self.w_mean = np.full(n_sigma, 1.0 / (2.0 * (float(self.dim_x) + self.par_lambda)))
        self.w_cov = self.w_mean.copy()

        self.w_mean[0] = self.par_lambda / (self.dim_x + self.par_lambda)
        self.w_cov[0] = self.w_mean[0] + (1 - self.alpha**2 + self.beta)

    def cholupdate(self, R, u, v, epsilon=1e-12, max_value=1e8):
        n = R.shape[0]
        
        if np.any(np.isnan(R)) or np.any(np.isinf(R)):
            print("WARNING: R contains NaN/Inf, using identity")
            R = np.eye(n) * epsilon
        
        u = np.asarray(u, dtype=float).flatten()
        if np.any(np.isnan(u)) or np.any(np.isinf(u)):
            print("WARNING: u contains NaN/Inf, using zeros")
            u = np.zeros(n)
        
        R_norm = np.linalg.norm(R)
        if R_norm > max_value:
            R = R * (max_value / R_norm)
        
        u_norm = np.linalg.norm(u)
        if u_norm > max_value:
            u = u * (max_value / u_norm)
        
        if np.abs(v) < epsilon or u_norm < epsilon:
            return R
        
        try:
            if v > 0:
                return self._cholupdate_positive(R, u, v, epsilon, max_value)
            else:
                return self._cholupdate_negative(R, u, -v, epsilon, max_value)
        except Exception as e:
            print(f"cholupdate failed: {e}, using SVD fallback")
            return self._cholupdate_svd_fallback(R, u, v, epsilon)

    def _cholupdate_positive(self, R, u, v, epsilon, max_value):
        n = R.shape[0]
        R_new = R.copy().astype(float)
        x = u.copy().astype(float)
        
        alpha = np.sqrt(v)
        if alpha > max_value:
            alpha = max_value
        elif alpha < epsilon:
            alpha = epsilon
            
        x = alpha * x
        
        for k in range(n):
            if np.abs(x[k]) < epsilon:
                continue
                
            r_old = R_new[k, k]
            
            if np.abs(r_old) > max_value:
                r_old = max_value * np.sign(r_old)
            if np.abs(x[k]) > max_value:
                x[k] = max_value * np.sign(x[k])
            
            r_old_sq = r_old * r_old
            x_k_sq = x[k] * x[k]
            
            if r_old_sq > max_value - x_k_sq:
                scale = max_value / (r_old_sq + x_k_sq)
                r_old_sq *= scale
                x_k_sq *= scale
                
            r_new_sq = r_old_sq + x_k_sq
            
            if r_new_sq <= 0:
                r_new = epsilon
            else:
                r_new = np.sqrt(r_new_sq)
                if r_new > max_value:
                    r_new = max_value
            
            if np.abs(r_old) > epsilon:
                c = r_new / r_old
                s = x[k] / r_old
            else:
                c = 1.0
                s = 0.0
            
            R_new[k, k] = r_new
            
            if k < n - 1:
                for j in range(k + 1, n):
                    old_val = R_new[k, j]
                    x_j = x[j]
                    
                    if np.abs(old_val) > max_value:
                        old_val = max_value * np.sign(old_val)
                    if np.abs(x_j) > max_value:
                        x_j = max_value * np.sign(x_j)
                    if np.abs(s) > max_value:
                        s = max_value * np.sign(s)
                    
                    new_r_val = (old_val + s * x_j) / c
                    
                    if np.abs(new_r_val) > max_value:
                        new_r_val = max_value * np.sign(new_r_val)
                        
                    R_new[k, j] = new_r_val
                    
                    new_x_val = c * x_j - s * old_val
                    if np.abs(new_x_val) > max_value:
                        new_x_val = max_value * np.sign(new_x_val)
                        
                    x[j] = new_x_val
        
        return R_new

    def _cholupdate_negative(self, R, u, v, epsilon, max_value):
        n = R.shape[0]
        R_new = R.copy().astype(float)
        x = u.copy().astype(float)
        
        alpha = np.sqrt(v)
        if alpha > max_value:
            alpha = max_value
        elif alpha < epsilon:
            alpha = epsilon
            
        x = alpha * x
        
        for k in range(n):
            if np.abs(x[k]) < epsilon:
                continue
                
            r_old = R_new[k, k]
            
            if np.abs(r_old) > max_value:
                r_old = max_value * np.sign(r_old)
            if np.abs(x[k]) > max_value:
                x[k] = max_value * np.sign(x[k])
            
            r_old_sq = r_old * r_old
            x_k_sq = x[k] * x[k]
            
            if r_old_sq - x_k_sq < epsilon:
                if r_old_sq >= epsilon:
                    r_new = epsilon
                else:
                    r_new = epsilon
            else:
                r_new = np.sqrt(r_old_sq - x_k_sq)
                if r_new > max_value:
                    r_new = max_value
            
            if np.abs(r_old) > epsilon:
                c = r_new / r_old
                s = x[k] / r_old
            else:
                c = 1.0
                s = 0.0
            
            R_new[k, k] = r_new
            
            if k < n - 1:
                for j in range(k + 1, n):
                    old_val = R_new[k, j]
                    x_j = x[j]
                    
                    if np.abs(old_val) > max_value:
                        old_val = max_value * np.sign(old_val)
                    if np.abs(x_j) > max_value:
                        x_j = max_value * np.sign(x_j)
                    if np.abs(s) > max_value:
                        s = max_value * np.sign(s)
                    
                    if np.abs(c) > epsilon:
                        new_r_val = (old_val - s * x_j) / c
                    else:
                        new_r_val = old_val
                    
                    if np.abs(new_r_val) > max_value:
                        new_r_val = max_value * np.sign(new_r_val)
                        
                    R_new[k, j] = new_r_val
                    
                    if np.abs(c) > epsilon:
                        new_x_val = c * x_j - s * old_val
                    else:
                        new_x_val = x_j
                        
                    if np.abs(new_x_val) > max_value:
                        new_x_val = max_value * np.sign(new_x_val)
                        
                    x[j] = new_x_val
        
        return R_new

    def _cholupdate_svd_fallback(self, R, u, v, epsilon):

        n = R.shape[0]
        
        try:
            P = R @ R.T
            
            update = v * np.outer(u, u)
            P_new = P + update
            
            P_new = (P_new + P_new.T) / 2
            
            U, s, Vt = np.linalg.svd(P_new, full_matrices=False)
            
            s_clean = np.maximum(s, epsilon)
            
            original_trace = np.trace(P_new)
            if original_trace <= 0:
                original_trace = np.trace(P) if np.trace(P) > 0 else n
                
            recovered_trace = np.sum(s_clean)
            
            if recovered_trace > 0:
                scale_factor = original_trace / recovered_trace
                s_scaled = s_clean * scale_factor
            else:
                s_scaled = s_clean
            
            P_recovered = U @ np.diag(s_scaled) @ Vt
            
            sqrt_s = np.sqrt(s_scaled)
            R_svd = U @ np.diag(sqrt_s)
            
            try:
                R_triangular = np.linalg.cholesky(P_recovered)
                return R_triangular
            except:
                return R_svd
                
        except Exception as e:
            print(f"SVD fallback also failed: {e}, using identity")
            return np.eye(n) * np.sqrt(epsilon)

    def cholupdate_simple(self, R, u, v, epsilon=1e-12):
        try:
            if np.abs(v) < 1e-15 or np.linalg.norm(u) < 1e-15:
                return R
                
            P_current = R @ R.T
            update = v * np.outer(u, u)
            P_new = P_current + update
            
            P_new = (P_new + P_new.T) / 2
            
            n = P_new.shape[0]
            eigenvals = np.linalg.eigvalsh(P_new)
            min_eig = np.min(eigenvals)
            
            if min_eig < epsilon:
                P_new += (epsilon - min_eig) * np.eye(n)
            
            return np.linalg.cholesky(P_new)
            
        except Exception as e:
            print(f"cholupdate_simple failed: {e}")
            n = R.shape[0] if hasattr(R, 'shape') else len(u)
            return np.eye(n) * np.sqrt(epsilon)
             
    def generate_sigma_points(self) -> np.ndarray:
        """ Создает массив сигма точек вектора состояния,
            образующий окрестность."""
        sigma_points = np.zeros((2 * self.dim_x + 1, self.dim_x))
        sigma_points[0] = self.state_v

        eta = self.alpha * np.sqrt(self.dim_x)
        for i in range(self.dim_x):
            sigma_points[i+1] = self.state_v + eta * self.SR_cov[:, i]
            sigma_points[i+1+self.dim_x] = self.state_v - eta * self.SR_cov[:, i]
        return sigma_points
   
    def prediction(self, t_k: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        # 1. Создание сигма-точек:
        self.sigma_points = self.generate_sigma_points()
        self.all_sigma_points.append(self.sigma_points)

        # 2. Протягиваем сигма-точки по эволюции с помощью пакета pyorbs:
        traj = Trajectories(amount_points = 2 * self.dim_x + 1,
                            sigma_points = self.sigma_points,
                            t_start = self.t_start,  t_k = t_k)

        self.transform_points = traj.get_transform_sigma_points()
        self.all_transform_points.append(self.transform_points)

        # 3. Предсказываем среднее и ковариационную матрицу.
        pred_state = self.w_mean @ self.transform_points

        QR_matrix = np.zeros((self.dim_x, 2 * self.dim_x))
        for i in range (2 * self.dim_x):
            QR_matrix[:, i] = sqrt(self.w_cov[1]) * (self.transform_points[i+1] - pred_state)
        QR_matrix = np.hstack([QR_matrix, scipy.linalg.sqrtm(self.cov_process_matrix)])
        _, R = np.linalg.qr(QR_matrix)
        R = R[:self.dim_x, :self.dim_x]

        SR_predict = self.cholupdate(R, self.transform_points[0] - pred_state, self.w_cov[0])
        pred_cov = SR_predict @ SR_predict.T
        
        self.write_pred_data(pred_state, pred_cov)
        self.times.append(t_k)

        return pred_state, pred_cov, SR_predict, traj

    def correction(self, z: pd.DataFrame, pred_state: np.ndarray, pred_cov: np.ndarray,
                   SR_predict: np.ndarray, t_k: pd.Timestamp, traj) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            z: таблица с измерениями из класса ContextOD.
            y_mean: предсказанный вектор состояния в момент времени t_k.
            P_y: предсказанная ковариационная матрица вектора 
            состояния в момент времени t_k.
            t_k: время, на которое предсказываем и корректируем данные.
        """
        a_priori_meas = z['val']
        # 4. Возвращаем невязки по измерениям с помощью
        #    пакета pyorbs:
        traj.measure = z
        res_meas = traj.set_residuals()[:, :, 0]
        calc_measure = a_priori_meas - res_meas

        # 5. Вычисляем предсказанное среднее и предсказанную
        #    ковариационную матрицу:
        pred_meas = self.w_mean @ calc_measure

        QR_matrix = np.zeros((2, 2 * self.dim_x))
        for i in range (2 * self.dim_x):
            QR_matrix[:, i] = sqrt(self.w_cov[1]) * (calc_measure[i+1] - pred_meas)
        
        min_variance = 1e-5
        for j in range(2):
            col_variance = np.var(QR_matrix[j, :])
            if col_variance < min_variance:
                #print(f"Column {j} has low variance {col_variance:.2e}, adding regularization")
                QR_matrix[j, :] += np.random.normal(0, np.sqrt(min_variance), 2 * self.dim_x)

        QR_matrix = np.hstack([QR_matrix, scipy.linalg.sqrtm(self.cov_matrix_measure)])
        _, R = np.linalg.qr(QR_matrix)
        R = R[:2, :2].T
        S = self.cholupdate(R, calc_measure[0]-pred_meas, self.w_cov[0])

        # 6. Вычисляем перекрестную ковариацию:
        Pyz = (self.w_cov[:, None, None] * (
                (self.sigma_points - pred_state)[:, :, None] @
                (calc_measure - pred_meas)[:, None, :]
                )).sum(axis = 0)
        print(f'S = {S}')
        # 7. Вычисляем матрицу усиления:
        try:
            temp = scipy.linalg.solve_triangular(S, Pyz.T, lower=False)
            Kalman_gain = scipy.linalg.solve_triangular(S.T, temp, lower=True).T
        except np.linalg.LinAlgError:
            Kalman_gain = Pyz @ np.linalg.pinv(S @ S.T)

        # 8. Вычисляем невязку по измерениям:
        res_z = a_priori_meas - pred_meas

        # 9. Корректируем вектор состояния и ковариационную матрицу
        self.state_v = pred_state + Kalman_gain @ res_z.T
        U = Kalman_gain @ S

        SR_temp = SR_predict.copy()
        for i in range(U.shape[1]):  
            u_col = U[:, i]
            SR_temp = self.cholupdate(SR_temp, u_col, -1.0)

        self.SR_cov = SR_temp
        self.cov_matrix = self.SR_cov @ self.SR_cov.T

        self.forward_states.append(self.state_v)
        self.forward_covs.append(self.cov_matrix)
        self.t_start = t_k

        return self.state_v, self.cov_matrix

    def step(self, z: pd.DataFrame, t_k: pd.Timestamp):
        """ Шаг фильтрации."""
        pred_state, pred_cov, SR_predict, traj = self.prediction(t_k)
        return self.correction(z, pred_state, pred_cov, SR_predict, t_k, traj)

    def write_pred_data(self, y_mean, P_y):
        """ Записывает предсказанные данные"""
        self.pred_states.append(y_mean)
        self.pred_covs.append(P_y)

    def rts_smoother(self) -> tuple[List[np.float64], List[np.float64]]:
        """Процесс сглаживания оценок вектора состояния 
        и ковариационной матрицы (Unscented Rauch-Tung-Striebel smoother
        for the additive dynamic system).
        """
        
        print(f'Cглаживание...')
        smoothed_states = self.forward_states.copy()
        smoothed_covs = self.forward_covs.copy()
        
        for k in range (len(self.forward_states)-2, -1, -1):
            cross_cov = np.zeros((self.dim_x, self.dim_x))
            for i in range (2 * self.dim_x + 1):
                dif = (self.all_transform_points[k][i] - self.forward_states[k]).reshape(-1,1)
                dif_pred = (self.all_transform_points[k+1][i] - self.pred_states[k+1]).reshape(-1, 1)
                cross_cov += self.w_cov[i] * (dif @ dif_pred.T)

            try:
                G = cross_cov @ np.linalg.inv(self.pred_covs[k+1])
            except:
                G = cross_cov @ np.linalg.pinv(self.pred_covs[k+1])

            smoothed_states[k] = self.forward_states[k] + G @ (smoothed_states[k+1] - self.pred_states[k+1])
            smoothed_covs[k] = self.forward_covs[k] + G @ (smoothed_covs[k+1] - self.pred_covs[k+1]) @ G.T

        return smoothed_states, smoothed_covs

    def draw_plots(self, smoothed_states, smoothed_covs):

        sigma_x = [sqrt(arr[0,0]) for arr in smoothed_covs]
        sigma_y = [sqrt(arr[1,1]) for arr in smoothed_covs]
        sigma_z = [sqrt(arr[2,2]) for arr in smoothed_covs]

        sigma_vx = [sqrt(arr[3,3]) for arr in smoothed_covs]
        sigma_vy = [sqrt(arr[4,4]) for arr in smoothed_covs]
        sigma_vz = [sqrt(arr[5,5]) for arr in smoothed_covs]
        
        plt.figure('Положение', figsize=(19,10))

        plt.subplot(3, 1, 1)
        plt.plot(self.times, sigma_x, '+')
        plt.grid(True)
        plt.title('СКО х')
        plt.xlabel('Время')
        plt.ylabel('$\sigma_x$')

        plt.subplot(3, 1, 2)
        plt.plot(self.times, sigma_y, '+')
        plt.grid(True)
        plt.title('СКО y')
        plt.xlabel('Время')
        plt.ylabel('$\sigma_y$')

        plt.subplot(3, 1, 3)
        plt.plot(self.times, sigma_z, '+')
        plt.grid(True)
        plt.title('СКО z')
        plt.xlabel('Время')
        plt.ylabel('$\sigma_z$')

        plt.tight_layout()
        plt.show()


        plt.figure('Скорость', figsize=(19,10))

        plt.subplot(3, 1, 1)
        plt.plot(self.times, sigma_vx, '+')
        plt.grid(True)
        plt.title('СКО $V_x$')
        plt.xlabel('Время')
        plt.ylabel('$\sigma_{V_x}$')

        plt.subplot(3, 1, 2)
        plt.plot(self.times, sigma_vy, '+')
        plt.grid(True)
        plt.title('СКО $V_y$')
        plt.xlabel('Время')
        plt.ylabel('$\sigma_{V_y}$')

        plt.subplot(3, 1, 3)
        plt.plot(self.times, sigma_vz, '+')
        plt.grid(True)
        plt.title('СКО $V_z$')
        plt.xlabel('Время')
        plt.ylabel('$\sigma_{V_z}$')

        plt.tight_layout()
        plt.show()
