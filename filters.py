import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from math import sqrt

import pyorbs
import _config

_SEC2RAD = np.pi / 180. / 3600.


class Trajectories:
    """ Класс, протягивающий траекторию по измерениям для каждой сигма точки.
    """
    
    #: Количество сигма точек
    amount_points: int

    #: Набор сигма точек в момент времени t_start
    sigma_points: np.ndarray

    #: Набор измерений
    measure: pd.Series | None

    #: Момент времени, от которого протягиваем орбиту
    t0: pyorbs.pyorbs.ephem_time

    #: Момент времени, до которого тянем орбиту
    t_k: pyorbs.pyorbs.ephem_time

    #: Список орбит протянутых сигма точек до момента времени t_k
    orb_list: list[pyorbs.pyorbs.orbit]

    #: Список протянутых сигма точек в момент времени t_k
    transform_points: np.ndarray = np.zeros((13, 6))

    sigma: float
    
    def __init__(
            self,
            amount_points: int,
            sigma_points: np.ndarray,
            t_start: pyorbs.pyorbs.ephem_time,
            t_k: pyorbs.pyorbs.ephem_time,
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
        self.t_k = t_k
        self.orb_list = []
        self.transform_points = np.ndarray((amount_points, (amount_points - 1) // 2 ))
        self.sigma = 0


    def get_transform_sigma_points(self) -> None:
        """ Протягивает облако сигма точек в заданный момент времени.
        """
        for i in range(self.N):
            orb = pyorbs.pyorbs.orbit()
            orb.state_v, orb.time = self.sigma_points[i], self.t0
            orb.setup_parameters()
            orb.move(self.t_k)
            self.transform_points[i] = orb.state_v
            self.orb_list.append(orb)


    def set_residuals(self) -> np.ndarray:
        """ Выдает невязки по измерениям для всех сигма точек
        в момент времени t_cur, взятый из таблицы измерений.
        """
        
        dz: List[np.ndarray] = []

        for (i, orbit) in enumerate(self.orb_list):
            orbit.setup_parameters()
            if self.measure is not None:
                meas = pyorbs.pyorbs_det.Measurements(self.measure.to_frame())
                step = pyorbs.pyorbs_det.newton_step(dim = 6, meas_tab = meas.tracking)
                m, _ = pyorbs.pyorbs_det.process_meas_record(orbit, self.measure, dim = 6, step = step)
                if i == 0:  
                    self.sigma = m['ds']
                dz.append(m['res'][:, 0] * _SEC2RAD)
                self.transform_points[i] = orbit.state_v

        return np.array(dz)


class UKF:
    """ 
    Класс, реализующий сигматочечный фильтр Калмана с квадратным
    корнем Холецкого. Фильтрация вектора состояния и корня ковариационной
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
    cov_matrix_measure: np.ndarray

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
    transform_points: np.ndarray

    #: Массив сигма точек
    sigma_points: np.ndarray | None = None

    #: Список предсказанных векторов состояния
    pred_states: List[np.ndarray] = []

    #: Список предсказанных ковариационных матриц 
    pred_covs: List[np.ndarray] = []

    #: Список скорректированных векторов состояния
    forward_states: List[np.ndarray] = []

    #: Список скорректированных ковариационных матриц
    forward_sr_covs: List[np.ndarray] = []

    #: Cписок моментов времени, в которые фильтруем данные
    times: List[str] = []

    #: Список всех сигма точек
    all_sigma_points: List[np.ndarray] = []

    #: Список всех сигма точек, протянутых в моменты 
    #: времени из списка times
    all_transform_points: List[np.ndarray] = []

    #: Набор измерений
    meas: pd.DataFrame

    #: Количество попыток фильтрации
    lim_filter: int

    #: Общая невязка за одну фильтрацию
    sigma: float

    def __init__(
            self, 
            P: np.ndarray,
            t_start: pyorbs.pyorbs.ephem_time,
            v: np.ndarray,
            meas: pd.DataFrame,
            sigma_points: np.ndarray | None = None,
            R: np.ndarray = _config.R_DEFAULT,
            alpha: float = _config.DEFAULT_ALPHA,
            beta: float =  _config.DEFAULT_BETA,
            kappa: float = _config.DEFAULT_KAPPA,
            lim_filter: int = _config.DEFAULT_LIM,
            dim_x: int = 6,
            pred_states: List[np.ndarray] = [],
            pred_covs: List[np.ndarray] = [],
            forward_states: List[np.ndarray] = [],
            sr_forward_covs: List[np.ndarray] = [],
            forward_covs: List[np.ndarray] = [],
            times: List[str] = [],
            all_transform_points: List[np.ndarray] = [],
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
            self.t_start = t_start
            self.state_v = v
            self.cov_process_matrix = _config.DEFAULT_COV_PROCESS
            self.cov_matrix_measure = R
            self.alpha = alpha #_config.MAX_VAR / np.sqrt(dim_x) / np.max(np.sqrt(P.diagonal()))
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
            self.lim_filter = lim_filter
            self.sigma = 0


    def set_weights(self) -> None:
        n_sigma = 2 * self.dim_x + 1
        self.w_mean = np.full(n_sigma, 1.0 / (2.0 * (float(self.dim_x) + self.par_lambda)))
        self.w_cov = self.w_mean.copy()

        self.w_mean[0] = self.par_lambda / (self.dim_x + self.par_lambda)
        self.w_cov[0] = self.w_mean[0] + 1 - self.alpha ** 2 + self.beta
    

    def generate_sigma_points(self) -> np.ndarray:
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

    def cholupdate(self, R: np.ndarray, x: np.ndarray, alpha: float) -> np.ndarray:
        if alpha > 0:
            P_new = R.T @ R + sqrt(alpha) * np.outer(x, x)
        elif alpha < 0 and abs(alpha + 1.0) > 1e-6:
            P_new = R.T @ R - sqrt(abs(alpha)) * np.outer(x, x)
        else:
            P_new = R.T @ R - abs(alpha) * np.outer(x, x)

        try:
            return scipy.linalg.cholesky(P_new)
        
        except ValueError:
            k =  np.sqrt(self.alpha) * np.linalg.norm(P_new.diagonal())
            if np.shape(P_new) == (6,6):
                P_new += k * np.eye(6)
            else:
                P_new += k * np.eye(2)

            return scipy.linalg.cholesky(P_new)

    def update_parameters(self):
        alpha_old = np.copy(self.alpha)
        self.alpha = _config.MAX_VAR / np.sqrt(self.dim_x) / np.max(np.sqrt(self.cov_matrix.diagonal()))
        print(self.alpha)
        if self.alpha > 1 or self.alpha < 0.15:
            self.alpha = alpha_old
            print(f'alpha вышел за допустимые значения. Отмена обновления.')
            return
        self.par_lambda = self.alpha ** 2 * (self.dim_x + self.kappa) - self.dim_x
        self.set_weights()

    def prediction(self, t_k: pyorbs.pyorbs.ephem_time) -> tuple[np.ndarray, np.ndarray, Trajectories]:
        """ Этап предсказания оценки и корня ковариационной матрицы.

        Args:

            t_k: Момент времени, до которого предсказываем (время взято из
            атрибута meas_data класса ContextOD).

        Returns:

            Набор предсказанной оценки и корня ковариационной матрицы и
            класс траекторий.
        """
        
        # 1. Создание сигма-точек и добавление их в общий список для 
        #    дальнейшего анализа:
        
        self.sigma_points = self.generate_sigma_points()
        self.all_sigma_points.append(self.sigma_points)

        # 2. Протягиваем сигма-точки по эволюции с помощью класса Trajectories:
        
        traj = Trajectories(amount_points = 2 * self.dim_x + 1,
                            sigma_points = self.sigma_points,
                            t_start = self.t_start, t_k = t_k, measure = None)
        traj.get_transform_sigma_points()

        # 3. Предсказываем оценку и корень ковариационной матрицы:
        
        pred_state = self.w_mean @ traj.transform_points
        #print(traj.transform_points)
        #print(f'pred state = {pred_state}')
        QR_matrix = np.zeros((self.dim_x, 2 * self.dim_x))
        for i in range (2 * self.dim_x):
            QR_matrix[:, i] = sqrt(self.w_cov[1]) * (traj.transform_points[i+1] - pred_state)
        QR_matrix = np.hstack([QR_matrix, scipy.linalg.cholesky(self.cov_process_matrix)])

        _, R = np.linalg.qr(QR_matrix.T)
        SR_predict = self.cholupdate(R, traj.transform_points[0] - pred_state, self.w_cov[0])
        #print('pred cov:')
        #print((SR_predict.T @ SR_predict)[:3, :3])
        return pred_state, SR_predict, traj

    def correction(self, z: pd.Series, pred_state: np.ndarray,
                    SR_predict: np.ndarray, t_k: pyorbs.pyorbs.ephem_time, traj: Trajectories) -> None:
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
        SR = SR_predict.copy()
        #print(f'pred state = {pred_state}')
        # 4. Возвращаем невязки по измерениям с помощью пакета
        #    pyorbs и класса Trajectories и считаем посчитанные по модели
        #    измерения:

        traj.measure = z
        res_meas = traj.set_residuals()
        self.sigma += traj.sigma
        self.all_transform_points.append(traj.transform_points)
        #print(self.alpha)
        #print(f'tranform = {traj.transform_points[0]}')
        calc_meas = a_priori_meas - res_meas
        #print(f'res = {res_meas[0][0] / _SEC2RAD}, {res_meas[0][1] / _SEC2RAD}')
        #print(self.alpha)

        # 5. Вычисляем предсказанную оценку по измерениям и предсказанный
        #    корень ковариационной матрицы с помощью алгоритма cholupdate:
        
        pred_meas = self.w_mean @ calc_meas

        QR_matrix = np.zeros((2, 2 * self.dim_x))
        for i in range (2 * self.dim_x):
            QR_matrix[:, i] = sqrt(self.w_cov[1]) * (calc_meas[i+1] - pred_meas)
        QR_matrix = np.hstack([QR_matrix, scipy.linalg.cholesky(self.cov_matrix_measure)])
        _, R_meas = np.linalg.qr(QR_matrix.T)
        S_meas = self.cholupdate(R_meas, calc_meas[0] - pred_meas, self.w_cov[0])

        # 6. Вычисляем перекрестную ковариацию:

        diff_state = traj.transform_points - state
        diff_meas = calc_meas - pred_meas

        Pyz = np.zeros((self.dim_x, 2))
        for i in range(2 * self.dim_x + 1):
            Pyz += self.w_cov[i] * diff_state[i,:].reshape(-1, 1) @ diff_meas[i,:].reshape(1, -1)
        #print(f'Pyz = {Pyz}')
        #print((np.linalg.inv(SR.T @ SR) @ Pyz).T)

        # 7. Вычисляем матрицу усиления Kalman_gain
        #    в случае неудачи, вычисляем псевдоопределенную матрицу:

        try:
            Kalman_gain = Pyz @ np.linalg.inv(S_meas.T @ S_meas)
        except np.linalg.LinAlgError:
            Kalman_gain = Pyz @ np.linalg.pinv(S_meas.T @ S_meas)
        #print(S_meas)
        #print(Kalman_gain)
        # 8. Вычисляем невязку по измерениям (разница 
        #    между считанными наблюдениями и "среднему" по Калману
        #    наблюдению):

        res_z = a_priori_meas - pred_meas


        # 9. Корректируем вектор состояния и корень ковариационной матрицы.
        #    В случае неудачи обновления Холецкого для корня завершаем
        #    шаг и переходим к следующему моменту времени:

        P_new = SR.T @ SR - Kalman_gain @ S_meas.T @ S_meas @ Kalman_gain.T
        #print(f'{(SR.T @ SR)[0,0]} - {(Kalman_gain @ S_meas.T @ S_meas @ Kalman_gain.T)[0,0]} = {P_new[0,0]}')
        try:
            self.SR_cov = scipy.linalg.cholesky(P_new)
        except np.linalg.LinAlgError as e:
            #k =  np.sqrt(self.alpha) * np.linalg.norm(P_new.diagonal())
            #P_new += k * np.eye(6)
            #self.SR_cov = scipy.linalg.cholesky(P_new)
            #print(f'Ошибка: {e}, регуляризация с k = {k}')
            print(f'Ошибка: {e}. Пропуск шага.')
            return
        
        self.state_v = state + Kalman_gain @ res_z
        self.cov_matrix = self.SR_cov.T @ self.SR_cov

        # 10. Добавка в списки всей необходимой информации для 
        #     дальнейшего сглаживания оценки и построения графиков:

        self.pred_states.append(pred_state)
        self.pred_covs.append(SR_predict.T @ SR_predict)
        self.forward_states.append(self.state_v)
        self.forward_sr_covs.append(self.SR_cov)
        self.forward_covs.append(self.cov_matrix)
        self.times.append(t_k.__str__())
        self.t_start = t_k


    def step(self, z: pd.Series, t_k: pyorbs.pyorbs.ephem_time) -> None:
        """ Шаг фильтрации:
        Args:
            z: Таблица с измерениями.
            t_k: Момент времени, до которого происходит коррекция.
        
        Returns:
            Уточненная оценка и ковариационная матрица.
        
        """
        try:
            pred_state, SR_predict, traj = self.prediction(t_k)
            self.correction(z, pred_state, SR_predict, t_k, traj)
        except pyorbs.exceptions.ReentryException as e:
            print(f'Шаг фильтрации не удался: {e}')
            self.t_start = t_k

    def filtration(self) -> tuple[np.ndarray, List[np.ndarray]]:
        """Процесс фильтрации.

        Returns:
            Сглаженный вектор состояния на последний момент времени и 
            список сглаженных ковариационных матриц"""

        for _, m in reversed(list(self.meas.iterrows())):
            t_k = pyorbs.pyorbs.ephem_time(m['time'].to_pydatetime())
            #if abs(self.t_start - t_k) * 1000 > 300:
            #    alpha_old = self.alpha
                #self.update_parameters()
            #    print(f'Обновление параметра alpha: {alpha_old} --- {self.alpha}')
                    
            self.step(m, t_k)
            #print(f'state = {self.state_v}')
            #print(f'cov = {self.cov_matrix[:3, :3]}')
            #print(np.sqrt(self.cov_matrix.diagonal()))
            if self.lim_filter == 1:
                print(f'Коррекция: {t_k.utc()}')
            #print('=' * 90)

        return self.state_v #self.rts_smoother()

    def several_filtrations(self, t0:pyorbs.pyorbs.ephem_time, P0: np.ndarray) -> np.ndarray:
        i = 0
        while i != self.lim_filter:
            print(f'Фильтрация... Попытка № {i+1}')
            vec = self.filtration()
            self.sigma = sqrt(self.sigma)
            print(f'sigma = {self.sigma}')
            self.sigma = 0
            orb = pyorbs.pyorbs.orbit()
            orb.state_v = vec
            orb.time = pyorbs.pyorbs.ephem_time(self.meas.iloc[0]['time'].to_pydatetime())
            orb.setup_parameters()
            orb.move(t0)
            self.state_v, self.t_start = orb.state_v, t0
            #print(self.state_v)
            self.cov_matrix = P0
            i+=1
        #return vec
        
    def rts_smoother(self) -> tuple[np.ndarray, List[np.ndarray]]:
        """ Процесс сглаживания оценок вектора состояния и ковариационной
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

        return smoothed_states[-1], smoothed_covs

    def draw_position_std(self, smoothed_covs: List[np.ndarray]):
        """ Рисует график СКО по положению вектора состояния.
        """
        sigma_x = [sqrt(arr[0,0] * 1000) * 1000 for arr in smoothed_covs]
        sigma_y = [sqrt(arr[1,1] * 1000) * 1000 for arr in smoothed_covs]
        sigma_z = [sqrt(arr[2,2] * 1000) * 1000 for arr in smoothed_covs]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True, figsize=(19,10))
        fig.suptitle('СКО положения (метры)', fontsize = 16) # type: ignore

        ax1.plot(self.times, sigma_x, '+', label = 'СКО х')
        ax1.set_ylabel('х')
        ax1.grid(True)

        ax2.plot(self.times, sigma_y, '+', label = 'СКО y')
        ax2.set_ylabel('у')
        ax2.grid(True)

        ax3.plot(self.times, sigma_z, '+', label = 'СКО z')
        ax3.set_ylabel('z')
        ax3.set_xlabel('Время')
        ax3.grid(True)

        plt.tight_layout()
        plt.show()


class LKF:
    
    def __init__(self,
            P: np.ndarray,
            t_start: pyorbs.pyorbs.ephem_time,
            v: np.ndarray,
            meas: pd.DataFrame,
            R: np.ndarray = _config.R_DEFAULT):
        
        self.cov = P
        self.t0 = t_start
        self.state_v = v
        self.meas = meas
        self.cov_meas = R


    def prediction(self, t_k: pyorbs.pyorbs.ephem_time) -> tuple[np.ndarray, np.ndarray]:

        num_orbits = 800
        orb = pyorbs.pyorbs.orbit()
        orb.state_v, orb.time = self.state_v, self.t0
        orbits = np.random.multivariate_normal(orb.state_v, self.cov, num_orbits)
        samples = np.ndarray((6, num_orbits))
        pyorbs.bal.uim_flushed()
        orb.setup_parameters()
        orb.move(t_k)
        for j in range(len(orbits)):
            pyorbs.bal.uim_flushed()
            transform_orbit = pyorbs.pyorbs.orbit()
            transform_orbit.state_v, transform_orbit.time = orbits[j], self.t0
            transform_orbit.setup_parameters()
            transform_orbit.move(t_k)
            samples[:, j] = transform_orbit.state_v
        P_pred = np.cov(samples)
        mean = orb.state_v
        self.t0 = t_k
        #print(f'pred cov = {P_pred[:3, :3]}')

        return mean, P_pred, orb
    
    def correction(self, z: pd.Series, center: np.ndarray, 
                        P_pred: np.ndarray, orb:pyorbs.pyorbs.orbit) -> None:
        mean = center.copy()
        m = pyorbs.pyorbs_det.Measurements(z.to_frame())
        step = pyorbs.pyorbs_det.newton_step(dim = 6, meas_tab = m.tracking)
        m, H = pyorbs.pyorbs_det.process_meas_record(orb, z, dim = 6, step = step)
        res = m['res'][:, 0] * _SEC2RAD
        #print(f'pred state = {mean}')
        print(f'res = {res[0] / _SEC2RAD}, {res[1] / _SEC2RAD}')
        Kalman_gain = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + self.cov_meas)
        #print(f'mat = {scipy.linalg.cholesky(P_pred).T[:3, :3]}')
        #print(H @ scipy.linalg.cholesky(P_pred).T)
        #print(f'H = {H}')
        #print(f'P * H^T = {P_pred @ H.T}')
        #print(Kalman_gain)
        #print('pred cov:')
        #print(P_pred[:3, :3])
        self.state_v = mean + Kalman_gain @ res.T
        #print(f'state = {self.state_v}')
        self.cov = (np.eye(6) - Kalman_gain @ H) @ P_pred
        print(f'{P_pred[0,0]} - {(Kalman_gain @ H @ P_pred)[0,0]} = {self.cov[0,0]}')
        print('corr cov:')
        print(self.cov[:3, :3])
        #print(np.sqrt(self.cov.diagonal()))


    def step(self, m: pd.Series, t_k: pyorbs.pyorbs.ephem_time) -> None:
        mean, P, orbit = self.prediction(t_k)
        self.correction(m, mean, P, orbit)
        print(f'Коррекция: {t_k.utc()}')

    def filtration(self) -> None:

        for _, m in reversed(list(self.meas.iterrows())):
            t_k = pyorbs.pyorbs.ephem_time(m['time'].to_pydatetime())
            self.step(m, t_k)
            #print(f'Коррекция: {t_k}')
            #print('=' * 90)


class EKF:

    def __init__(self,
            P: np.ndarray,
            t_start: pyorbs.pyorbs.ephem_time,
            v: np.ndarray,
            meas: pd.DataFrame,
            R: np.ndarray = _config.R_DEFAULT):
        
        self.cov = P
        self.t0 = t_start
        self.state_v = v
        self.meas = meas
        self.cov_meas = R

    def prediction(self, t_k):
        mean_orb = pyorbs.pyorbs.orbit()
        mean_orb.state_v, mean_orb.time = self.state_v, self.t0

        mean_orb.change_param({'calc_partials': True})
        mean_orb.set_initial_point(self.t0)
        mean_orb.setup_parameters()
        mean_orb.move(t_k)

        F = mean_orb.state_v[mean_orb.structure['calc_partials']].reshape(6,6)
        mean = mean_orb.state_v[:6]
        P_mean = F.T @ self.cov @ F

        return mean, P_mean

    def correction(self, z, x_pred, P_pred, t_k):
        mean = x_pred.copy()
        P = P_pred.copy()
        orb = pyorbs.pyorbs.orbit()
        orb.state_v, orb.time = x_pred, t_k

        m = pyorbs.pyorbs_det.Measurements(z.to_frame())
        step = pyorbs.pyorbs_det.newton_step(dim = 6, meas_tab = m.tracking)
        m, H = pyorbs.pyorbs_det.process_meas_record(orb, z, dim = 6, step = step)
        res = m['res'][:, 0] * _SEC2RAD
        S = H @ P @ H.T + self.cov_meas
        K = P @ H.T @ np.linalg.inv(S)

        self.state_v = mean + K @ res
        self.cov = (np.eye(6) - K @ H) @ P
        self.t0 = t_k

    def step(self, z, t_k):
        x, P = self.prediction(t_k)
        self.correction(z, x, P, t_k)

    def filtration(self):
        for _, m in reversed(list(self.meas.iterrows())):
            t_k = pyorbs.pyorbs.ephem_time(m['time'].to_pydatetime())
            self.step(m, t_k)
            print(f'Коррекция: {t_k}')
        