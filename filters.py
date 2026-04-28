import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

import pyorbs
import _config

_SEC2RAD = np.pi / 180. / 3600.

def create_orbit(v: np.ndarray[np.float64], 
                    t: pyorbs.pyorbs.ephem_time) -> pyorbs.pyorbs.orbit:
    obj = pyorbs.pyorbs.orbit()
    obj.state_v, obj.time = v, t
    return obj

def draw_position(times: List[str], 
                    smoothed_covs: List[np.ndarray[np.float64]]):
    """ Рисует графики отклонения по положению вектора состояния.
    """
    sigma_x = [np.sqrt(arr[0,0] * 1000) * 1000 for arr in smoothed_covs]
    sigma_y = [np.sqrt(arr[1,1] * 1000) * 1000 for arr in smoothed_covs]
    sigma_z = [np.sqrt(arr[2,2] * 1000) * 1000 for arr in smoothed_covs]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True, figsize=(19,10)) # type: ignore
    fig.suptitle('Отклонения положения (метры)', fontsize = 16) # type: ignore

    ax1.plot(times, sigma_x, '+', label = 'Отклонение х')
    ax1.set_ylabel('х')
    ax1.grid(True)

    ax2.plot(times, sigma_y, '+', label = 'Отклонение y')
    ax2.set_ylabel('у')
    ax2.grid(True)

    ax3.plot(times, sigma_z, '+', label = 'Отклонение z')
    ax3.set_ylabel('z')
    ax3.set_xlabel('Время')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


class Trajectories:
    """ Класс, протягивающий траекторию по измерениям для каждой сигма точки.
    """
    
    #: Количество сигма точек
    amount_points: int

    #: Набор сигма точек в момент времени t_start
    sigma_points: np.ndarray[np.float64]

    #: Набор измерений
    measure: pd.Series | None

    #: Момент времени, от которого протягиваем орбиту
    t0: pyorbs.pyorbs.ephem_time

    #: Список орбит протянутых сигма точек до момента времени t_k
    orb_list: list[pyorbs.pyorbs.orbit]

    #: Список протянутых сигма точек в момент времени t_k
    transform_points: np.ndarray[np.float64] = np.zeros((13, 6))

    #: Невязки, помноженные на матрицу весов
    sigma: float = 0.0
    
    def __init__(
            self,
            amount_points: int,
            sigma_points: np.ndarray[np.float64],
            t_start: pyorbs.pyorbs.ephem_time,
            measure: pd.Series | None, 
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
        self.t0 = t_start
        self.orb_list = []
        self.transform_points = np.ndarray((amount_points, (amount_points - 1) // 2 ))
        self.sigma = 0


    def get_transform_sigma_points(self, t_k: pyorbs.pyorbs.ephem_time) -> None:
        """ Протягивает облако сигма точек в заданный момент времени.
        """
        for i in range(self.N):
            orb = create_orbit(self.sigma_points[i], self.t0)
            orb.setup_parameters()
            orb.move(t_k)
            self.transform_points[i] = orb.state_v
            self.orb_list.append(orb)


    def set_residuals(self) -> np.ndarray[np.float64]:
        """ Выдает невязки по измерениям для всех сигма точек
        в момент времени t_cur, взятый из таблицы измерений.
        """
        
        dz: List[np.ndarray[np.float64]] = []

        for (i, orbit) in enumerate(self.orb_list):
            orbit.setup_parameters()
            if self.measure is not None:
                meas = pyorbs.pyorbs_det.Measurements(self.measure.to_frame())
                step = pyorbs.pyorbs_det.newton_step(dim = 6, meas_tab = meas.tracking)
                m, _ = pyorbs.pyorbs_det.process_meas_record(orbit, self.measure, dim = 6, step = step)
                if i == 0:
                    self.sigma = m['ds']
                dz.append(m['res'][:, 0] * _SEC2RAD)
                self.transform_points[i] = orbit.state_v #?

        return np.array(dz)


class UKF:
    """ 
    Класс, реализующий сигматочечный фильтр Калмана с квадратным
    корнем Холецкого. Фильтрация вектора состояния и корня ковариационной
    матрицы по массиву измерений, взятых из ContextOD.
    """
    
    #: Ковариационнная матрица вектора состояния 
    cov_matrix: np.ndarray[np.float64]

    #: Квадратный корень ковариационной матрицы
    SR_cov: np.ndarray[np.float64]

    #: Время t_k начала шага фильтрации
    t_start: pyorbs.pyorbs.ephem_time

    #: Начальное время фильтрации
    t_begin: pyorbs.pyorbs.ephem_time

    #: Вектор состояния
    state_v: np.ndarray[np.float64]

    #: Ковариационная матрица процесса
    cov_process_matrix: np.ndarray[np.float64]

    #: Ковариационная матрица измерений
    cov_matrix_measure: np.ndarray[np.float64]

    #: Параметр отвечающий за разброс от вектора 
    #: состояния. Нужен для определения параметра лямбда
    alpha: float

    #: Второй параметр для определения лямбда
    kappa: float

    #: Параметр для инициализации веса ковариации
    beta: float

    #: Размерность вектора состояния динамической системы
    dim_x: int

    #: Параметр для разложения Холецкого
    par_lambda: float

    #: Матрица преобразованных сигма точек
    transform_points: np.ndarray[np.float64]

    #: Массив сигма точек
    sigma_points: np.ndarray[np.float64] | None = None

    #: Список предсказанных векторов состояния
    pred_states: List[np.ndarray[np.float64]] = []

    #: Список предсказанных ковариационных матриц 
    pred_covs: List[np.ndarray[np.float64]] = []

    #: Список скорректированных векторов состояния
    forward_states: List[np.ndarray[np.float64]] = []

    #: Список скорректированных ковариационных матриц
    forward_sr_covs: List[np.ndarray[np.float64]] = []

    #: Cписок моментов времени, в которые фильтруем данные
    times: List[str] = []

    #: Список всех сигма точек
    all_sigma_points: List[np.ndarray[np.float64]] = []

    #: Список всех сигма точек, протянутых в моменты 
    #: времени из списка times
    all_transform_points: List[np.ndarray[np.float64]] = []

    #: Набор измерений
    meas: pd.DataFrame

    #: Количество попыток фильтрации
    attempt: int

    #: Сигма на одной попытке фильтрации
    sigma: float

    #: Список сигм для каждой попытки фильтрации
    sigma_list: List[float]

    #: Массив весов траекторий
    w_mean: np.ndarray[np.float64]

    def __init__(
            self, 
            P: np.ndarray[np.float64],
            t_begin: pyorbs.pyorbs.ephem_time,
            v: np.ndarray[np.float64],
            meas: pd.DataFrame,
            sigma_points: np.ndarray[np.float64] | None = None,
            R: np.ndarray[np.float64] = _config.R_DEFAULT,
            Q: np.ndarray[np.float64] = _config.DEF_COV_PROC,
            alpha: float = _config.DEFAULT_ALPHA,
            beta: float =  _config.DEFAULT_BETA,
            kappa: float = _config.DEFAULT_KAPPA,
            attempts: int = _config.DEFAULT_ATTEMPT,
            dim_x: int = 6,
            pred_states: List[np.ndarray[np.float64]] = [],
            pred_covs: List[np.ndarray[np.float64]] = [],
            forward_states: List[np.ndarray[np.float64]] = [],
            sr_forward_covs: List[np.ndarray[np.float64]] = [],
            forward_covs: List[np.ndarray[np.float64]] = [],
            times: List[str] = [],
            all_transform_points: List[np.ndarray[np.float64]] = [],
        ) -> None:
            """Конструктор сигматочечного фильтра Калмана
            
            Args:
                P: Начальное приближение ковариационной матрицы вектора состояния.
                t_start: Время начала фильтрации.
                x: Начальное приближение вектора состояния.
                meas: Таблица измерений.
                sigma_points: Набор сигма точек.
                R: Ковариационная матрица наблюдения.
                alpha: Параметр для опредения параметра /lambda для вычисления корня в разложении Холецкого.
                beta: Параметр для инициализации веса w0 ковариации.
                kappa: Параметр для опредения параметра /lambda для вычисления корня в разложении Холецкого.
                dim_x: Размерность динамической системы (по умолчанию 6).
                pred_states: Список предсказанных оценок вектора состояния вдоль эволюции.
                pred_covs: Список предсказанных ковариационных матриц вдоль эволюции.
                forward_states: Список скорректированных векторов состояния вдоль эволюции.
                sr_forward_covs: Список скорректированных корней из ковариационной матрицы вдоль эволюции.
                forward_covs: Список скорректированных ковариационных матриц вдоль эволюции.
                times: Список всех моментов времени, по которым фильтруем данные.
                all_transform_points: Список всех сигма точек, протянутых на следующий момент времени.

            Returns:

            """
            self.cov_matrix = P
            self.t_start = t_begin
            self.t_begin = t_begin
            self.state_v = v
            self.cov_process_matrix = Q
            self.cov_matrix_measure = R
            self.alpha = alpha
            if self.alpha > 1. or self.alpha < 0.:
                print(f'alpha = {self.alpha}. Завершение программы')
                exit()
            self.beta = beta
            self.kappa = kappa

            self.dim_x = dim_x
            self.par_lambda = self.alpha ** 2 * (dim_x + kappa) - dim_x
            self.SR_cov = scipy.linalg.cholesky(P)

            self.set_weights()

            self.pred_states = pred_states
            self.pred_covs = pred_covs
            self.forward_states = forward_states
            self.forward_sr_covs = sr_forward_covs
            self.forward_covs = forward_covs
            self.times = times
            self.sigma_points = sigma_points
            self.all_transform_points = all_transform_points
            self.meas = meas
            self.attempts = attempts
            self.list_sigma = []
            self.sigma = 0

    def set_weights(self) -> None:
        n_sigma = 2 * self.dim_x + 1
        self.w_mean = np.full(n_sigma, 1.0 / (2.0 * (float(self.dim_x) + self.par_lambda)))
        self.w_cov = self.w_mean.copy()

        self.w_mean[0] = self.par_lambda / (self.dim_x + self.par_lambda)
        self.w_cov[0] = self.w_mean[0] + 1 - self.alpha ** 2 + self.beta
    
    def generate_sigma_points(self) -> np.ndarray[np.float64]:
        """ Создает массив, образующий окрестность вокруг
        вектора состояния.
            
        Returns:
            Облако сигма точек с вектором состояния 
            на каждый момент времени оценки.
        """
        sigma_points = np.zeros((2 * self.dim_x + 1, self.dim_x))
        sigma_points[0] = self.state_v

        sqrt_matrix = np.sqrt(self.dim_x + self.par_lambda) * self.SR_cov
        for i in range(self.dim_x):
            sigma_points[i+1] = self.state_v + sqrt_matrix[i,:]
            sigma_points[i+1+self.dim_x] = self.state_v - sqrt_matrix[i,:]

        return sigma_points

    def cholupdate(self, R: np.ndarray[np.float64], 
                   x: np.ndarray[np.float64], alpha: float) -> np.ndarray[np.float64]:

        if alpha > 0:
            P_new = R.T @ R + np.sqrt(alpha) * np.outer(x, x)
        elif alpha < 0 and abs(alpha + 1.0) > 1e-6:
            P_new = R.T @ R - np.sqrt(abs(alpha)) * np.outer(x, x)
        else:
            P_new = R.T @ R - abs(alpha) * np.outer(x, x)

        try:
            return scipy.linalg.cholesky(P_new)
        
        except ValueError:
            k: float = np.sqrt(self.alpha) * np.linalg.norm(P_new.diagonal())
            if np.shape(P_new) == (6,6):
                P_new += k * np.eye(6)
            else:
                P_new += k * np.eye(2)

            return scipy.linalg.cholesky(P_new)

    def prediction(self, t_k: pyorbs.pyorbs.ephem_time) -> tuple[np.ndarray[np.float64], 
                                                           np.ndarray[np.float64], 
                                                           Trajectories]:
        """ Этап предсказания оценки и корня ковариационной матрицы.

        Args:

            t_k: Момент времени, до которого предсказываем (время взято из
            атрибута meas_data класса ContextOD).

        Returns:

            Набор предсказанной оценки и корня ковариационной матрицы и
            класс траекторий.
        """
    
        self.sigma_points = self.generate_sigma_points()
        self.all_sigma_points.append(self.sigma_points)
        
        traj = Trajectories(amount_points = 13, measure = None,
                            sigma_points = self.sigma_points,
                            t_start = self.t_start)
        traj.get_transform_sigma_points(t_k)
        
        pred_state = self.w_mean @ traj.transform_points

        QR_matrix = np.zeros((self.dim_x, 2 * self.dim_x))
        for i in range (2 * self.dim_x):
            QR_matrix[:, i] = np.sqrt(self.w_cov[1]) * (traj.transform_points[i+1] - pred_state)
        QR_matrix = np.hstack([QR_matrix, scipy.linalg.cholesky(self.cov_process_matrix)])

        _, R = np.linalg.qr(QR_matrix.T)
        SR_predict = self.cholupdate(R, traj.transform_points[0] - pred_state, self.w_cov[0])

        return pred_state, SR_predict, traj

    def correction(self, z: pd.Series, pred_state: np.ndarray[np.float64],
                    SR_pred: np.ndarray[np.float64], t_k: pyorbs.pyorbs.ephem_time, 
                    traj: Trajectories) -> None:
        """ Этап коррекции оценки вектора состояния и корня ковариационной матрицы.

        Args:
            z: таблица с измерениями из класса ContextOD.
            pred_state: предсказанный вектор состояния в момент времени t_k.
            SR_predict: предсказанный корень ковариационной матрицы.
            t_k: момент времени коррекции.
            traj: класс протянутых траекторий, образованных сигма точками.
        
        Returns:
            Оценка и корень ковариационной матрицы оценки вектора состояния.
        """

        a_priori_meas = np.array(z['val'])

        state = pred_state.copy()
        SR = SR_pred.copy()

        traj.measure = z
        res_meas = traj.set_residuals()
        self.sigma += traj.sigma
        self.all_transform_points.append(traj.transform_points)
        calc_meas = a_priori_meas - res_meas
        #print(f'res = {res_meas[0][0] / _SEC2RAD}, {res_meas[0][1] / _SEC2RAD}')
        
        pred_meas = self.w_mean @ calc_meas

        QR_matrix = np.zeros((2, 2 * self.dim_x))
        for i in range (2 * self.dim_x):
            QR_matrix[:, i] = np.sqrt(self.w_cov[1]) * (calc_meas[i+1] - pred_meas)
        QR_matrix = np.hstack([QR_matrix, scipy.linalg.cholesky(self.cov_matrix_measure)])
        _, R_meas = np.linalg.qr(QR_matrix.T)
        S_meas = self.cholupdate(R_meas, calc_meas[0] - pred_meas, self.w_cov[0])

        diff_meas = calc_meas - pred_meas
        diff_state = traj.transform_points - state

        Pyz = np.zeros((self.dim_x, 2))
        for i in range(2 * self.dim_x + 1):
            Pyz += self.w_cov[i] * diff_state[i,:].reshape(-1, 1) @ diff_meas[i,:].reshape(1, -1)

        K = Pyz @ np.linalg.inv(S_meas.T @ S_meas)
        
        self.state_v = state + K @ (a_priori_meas - pred_meas)
        self.cov_matrix = SR.T @ SR - K @ S_meas.T @ S_meas @ K.T

        try:
            self.SR_cov = scipy.linalg.cholesky(self.cov_matrix)
        except np.linalg.LinAlgError as e:
            print(f'Ошибка: {e}. Пропуск шага.')
            return

        #self.pred_states.append(pred_state)
        #self.pred_covs.append(self.cov_matrix)
        #self.forward_states.append(self.state_v)
        ##self.forward_sr_covs.append(self.SR_cov)
        #self.forward_covs.append(self.cov_matrix)
        #self.times.append(t_k.__str__())


    def step(self, z: pd.Series, t_k: pyorbs.pyorbs.ephem_time) -> None:
        """ Шаг фильтрации:
        Args:
            z: Таблица с измерениями.
            t_k: Момент времени, до которого происходит коррекция.
        
        Returns:
            Уточненная оценка и ковариационная матрица.
        
        """
        pred_state, SR_predict, traj = self.prediction(t_k)
        self.correction(z, pred_state, SR_predict, t_k, traj)
        self.t_start = t_k


    def od_filtration(self) -> np.ndarray[np.float64] | None:
        """Процесс фильтрации.

        Returns:
            Сглаженный вектор состояния на последний момент времени и 
            список сглаженных ковариационных матриц"""
        
        i = 0
        while i != self.attempts:
            print(f'sigma = {self.sigma}')
            print(f'Фильтрация... Попытка № {i+1}')
            self.sigma = 0
            # Сама фильтрация (проход по моментам измерения в таблице)
            for _, m in reversed(list(self.meas.iterrows())):
                t_k = pyorbs.pyorbs.ephem_time(m['time'].to_pydatetime())
                self.step(m, t_k)

            self.sigma = np.sqrt(self.sigma)
            self.list_sigma.append(self.sigma)
            
            if self.sigma < _config.GOOD_SIGMA:
                self.get_init_orbit()
                break
            elif (i > 1
                and abs((self.list_sigma[-1] - self.list_sigma[-2]) / self.list_sigma[-1]) < _config.EPS_CONVERGE
                and abs((self.list_sigma[-2] - self.list_sigma[-3]) / self.list_sigma[-2]) < _config.EPS_CONVERGE):
                self.get_init_orbit()
                break
            else:
                self.get_init_orbit()
                self.attempts +=1
                i+=1


    def get_init_orbit(self):
        orb = create_orbit(self.state_v.copy(), pyorbs.pyorbs.ephem_time(self.meas.iloc[0]['time'].to_pydatetime()))
        orb.change_param({'calc_partials': True})
        orb.set_initial_point(orb.time)
        orb.setup_parameters()
        orb.move(self.t_begin)
        F = orb.state_v[orb.structure['calc_partials']].reshape(6,6)
        self.state_v, self.t_start = orb.state_v[:6].copy(), self.t_begin
        self.cov_matrix = F.T @ self.cov_matrix @ F
      
    def rts_smoother(self) -> tuple[np.ndarray[np.float64], List[np.ndarray[np.float64]]]:
        """ Процесс сглаживания оценки вектора состояния и ковариационной
            матрицы (Unscented Rauch-Tung-Striebel smoother for the additive
            dynamic system).

            Returns:
                Сглаженный вектор состояния и список сглаженных ковариационных матриц.
        """
        
        #print(f'Cглаживание...')
        smoothed_states = self.forward_states.copy()
        smoothed_covs = self.forward_covs.copy()
        
        for k in range (len(self.forward_states)-2, -1, -1):
            cross_cov = np.zeros((self.dim_x, self.dim_x))
            for i in range (2 * self.dim_x + 1):
                dif = (self.all_transform_points[k][i] - self.forward_states[k]).reshape(-1,1)
                dif_pred = (self.all_transform_points[k+1][i] - self.pred_states[k+1]).reshape(-1, 1)
                cross_cov += self.w_cov[i] * (dif @ dif_pred.T)

            try:
                Gain = cross_cov @ np.linalg.inv(self.pred_covs[k+1])
            except:
                Gain = cross_cov @ np.linalg.pinv(self.pred_covs[k+1])

            smoothed_states[k] = self.forward_states[k] + Gain @ (smoothed_states[k+1] - self.pred_states[k+1])
            smoothed_covs[k] = self.forward_covs[k] + Gain @ (smoothed_covs[k+1] - self.pred_covs[k+1]) @ Gain.T
        print(smoothed_states[0])
        return smoothed_states[-1], smoothed_covs


class LKF:
    
    #: Вектор состояния системы
    state_v: np.ndarray[np.float64]

    #: Ковариационная матрица
    cov: np.ndarray[np.float64]

    def __init__(self,
                 P: np.ndarray[np.float64],
                 t_begin: pyorbs.pyorbs.ephem_time,
                 v: np.ndarray[np.float64],
                 meas: pd.DataFrame,
                 R: np.ndarray[np.float64] = _config.R_DEFAULT,
                 Q: np.ndarray[np.float64] = _config.DEF_COV_PROC,
                 attempts: int = _config.DEFAULT_ATTEMPT):
        
        self.cov = P
        self.t_begin = t_begin
        self.t_start = t_begin
        self.state_v = v
        self.meas = meas
        self.cov_meas = R
        self.cov_process = Q
        
        self.attempts = attempts

    def prediction(self, t_k: pyorbs.pyorbs.ephem_time
                   ) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:

        mean_orb = create_orbit(self.state_v.copy(), self.t_start)
        mean_cur = self.state_v.copy()

        mean_orb.change_param({'calc_partials': True})
        mean_orb.set_initial_point(self.t_start)
        mean_orb.setup_parameters()
        mean_orb.move(t_k)

        F = mean_orb.state_v[mean_orb.structure['calc_partials']].reshape(6,6).copy()

        mean = F.T @ mean_cur
        print(mean)
        cov_mean = F.T @ self.cov @ F + self.cov_process

        return mean, cov_mean, F.T
    
    def correction(self, z: pd.Series, mean: np.ndarray[np.float64], 
                        P: np.ndarray[np.float64], t_k: pyorbs.pyorbs.ephem_time, 
                        F: np.ndarray[np.float64]) -> None:
        
        y = np.array(z['val'])
        orb = create_orbit(mean.copy(), t_k)
        m = pyorbs.pyorbs_det.Measurements(z.to_frame())
        step = pyorbs.pyorbs_det.newton_step(dim = 6, meas_tab = m.tracking)
        orb.setup_parameters()
        _, dPsidX = pyorbs.pyorbs_det.process_meas_record(orb, z, dim = 6, step = step)

        H = dPsidX @ F
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + self.cov_meas)
        self.state_v = mean + K @ (y - H @ mean)
        #print((y - H @ mean) / _SEC2RAD)
        self.cov = (np.eye(6) - K @ H) @ P

    def step(self, m: pd.Series, t_k: pyorbs.pyorbs.ephem_time) -> None:
        mean, P, F = self.prediction(t_k)
        self.correction(m, mean, P, t_k, F)
        #if self.attempts == 1:
            #print(self.state_v)
            #print(f'Коррекция: {t_k.utc()}')
        self.t_start = t_k

    def get_init_orbit(self):

        orb = create_orbit(self.state_v.copy(), pyorbs.pyorbs.ephem_time(self.meas.iloc[0]['time'].to_pydatetime()))
        orb.change_param({'calc_partials': True})
        orb.set_initial_point(orb.time)
        orb.setup_parameters()
        orb.move(self.t_begin)
        F = orb.state_v[orb.structure['calc_partials']].reshape(6,6).copy()
        self.state_v, self.t_start = orb.state_v[:6].copy(), self.t_begin
        self.cov = F.T @ self.cov @ F

    def od_filtration(self) -> np.ndarray[np.float64] | None:
        """Процесс фильтрации.

        Returns:
            Сглаженный вектор состояния на последний момент времени и 
            список сглаженных ковариационных матриц"""
        
        i = 0
        while i != self.attempts:
            print(f'Фильтрация... Попытка № {i+1}')

            for k, m in reversed(list(self.meas.iterrows())):
                t_k = pyorbs.pyorbs.ephem_time(m['time'].to_pydatetime())
                if k == 1:
                    print(self.state_v)
                self.step(m, t_k)
            self.get_init_orbit()
            i+=1


class EKF:

    state_v: np.ndarray[np.float64]

    cov: np.ndarray[np.float64]

    t0: pyorbs.pyorbs.ephem_time

    cov_meas: np.ndarray[np.float64]

    cov_process: np.ndarray[np.float64]

    attempts: int

    def __init__(self,
            P: np.ndarray[np.float64],
            t_begin: pyorbs.pyorbs.ephem_time,
            v: np.ndarray[np.float64],
            meas: pd.DataFrame,
            R: np.ndarray[np.float64] = _config.R_DEFAULT,
            Q: np.ndarray[np.float64] = _config.DEF_COV_PROC,
            attempts: int = _config.DEFAULT_ATTEMPT):
        
        self.cov = P
        self.t0 = t_begin
        self.t_begin = t_begin
        self.state_v = v
        self.meas = meas
        self.cov_meas = R
        self.cov_process = Q
        self.attempts = attempts

    def prediction(self, t_k: pyorbs.pyorbs.ephem_time
                   ) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
        mean_orb = create_orbit(self.state_v, self.t0)
        mean_orb.change_param({'calc_partials': True})
        mean_orb.set_initial_point(mean_orb.time)
        mean_orb.setup_parameters()
        mean_orb.move(t_k)

        F = mean_orb.state_v[mean_orb.structure['calc_partials']].reshape(6,6)
        mean = mean_orb.state_v[:6]
        P_mean = F.T @ self.cov @ F + self.cov_process

        return mean, P_mean, F.T.copy()

    def correction(self, z: pd.Series, x_pred: np.ndarray[np.float64], 
                   P_pred: np.ndarray[np.float64], t_k: pyorbs.pyorbs.ephem_time, F):
        mean = x_pred.copy()
        P = P_pred.copy()

        orb = create_orbit(x_pred, t_k)

        m = pyorbs.pyorbs_det.Measurements(z.to_frame())
        step = pyorbs.pyorbs_det.newton_step(dim = 6, meas_tab = m.tracking)
        m, dPsidX = pyorbs.pyorbs_det.process_meas_record(orb, z, dim = 6, step = step)
        H = dPsidX @ F
        res = m['res'][:, 0] * _SEC2RAD
        #print(res / _SEC2RAD)
        S: np.ndarray[np.float64] = H @ P @ H.T + self.cov_meas
        K = P @ H.T @ np.linalg.inv(S)

        self.state_v = mean + K @ res
        self.cov = (np.eye(6) - K @ H) @ P

    def step(self, z: pd.Series, t_k: pyorbs.pyorbs.ephem_time):
        x, P, F = self.prediction(t_k)
        self.correction(z, x, P, t_k, F)
        self.t0 = t_k

    def get_init_orbit(self):

        orb = create_orbit(self.state_v.copy(), pyorbs.pyorbs.ephem_time(self.meas.iloc[0]['time'].to_pydatetime()))
        orb.change_param({'calc_partials': True})
        orb.set_initial_point(orb.time)
        orb.setup_parameters()
        orb.move(self.t_begin)
        F = orb.state_v[orb.structure['calc_partials']].reshape(6,6).copy()
        self.state_v, self.t0 = orb.state_v[:6].copy(), self.t_begin
        print(f'на правый момент {self.state_v}')
        self.cov = F.T @ self.cov @ F

    def od_filtration(self) -> np.ndarray[np.float64] | None:
        """Процесс фильтрации.

        Returns:
            Сглаженный вектор состояния на последний момент времени и 
            список сглаженных ковариационных матриц"""
        
        i = 0
        while i != self.attempts:
            print(f'Фильтрация... Попытка № {i+1}')
            # Сама фильтрация (проход по моментам измерения в таблице)
            for _, m in reversed(list(self.meas.iterrows())):
                t_k = pyorbs.pyorbs.ephem_time(m['time'].to_pydatetime())
                self.step(m, t_k)
            self.get_init_orbit()
            i+=1
