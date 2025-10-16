import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter():
    def __init__(self, F, H, Q, R, P, x0):
        """Инициализация фильтра Калмана:
        
            Parameters:
            F - матрица перехожа состояния;
            H - матрица наблюдений;
            Q - ковариационная матрица процесса;
            R - ковариационная матрица измерений;
            P - ковариационная матрица ошибки;
            x0 - начальное состояние;
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x0

    def prediction(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x
    
    def update(self,z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        y = z - self.H @ self.x
        self.x = self.x + K @ y

        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

        return self.x
    
    def get_state(self):
        return self.x.copy()
    
def example_2d_tracking():
    dt = 0.5
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    
    process_noise = 0.1
    q = process_noise ** 2
    Q = np.array([[dt**4/4, 0, dt**3/2, 0],
                  [0, dt**4/4, 0, dt**3/2],
                  [dt**3/2, 0, dt**2, 0],
                  [0, dt**3/2, 0, dt**2]]) * q
    
    measurement_noise = 0.8
    R = np.eye(2) * measurement_noise ** 2

    P = np.eye(4)
    x0 = np.array([0, 0, 1, 0.5])

    kf = KalmanFilter(F, H, Q, R, P, x0)

    num_steps = 200
    true_trajectory = []
    measurements = []
    estimates = []

    radius = 10
    angular_velocity = 0.1

    for i in range(num_steps):
        t = i * dt
        x = radius * np.cos(angular_velocity * t)
        y = radius * np.sin(angular_velocity * t)
        vx = -radius * angular_velocity * np.sin(angular_velocity * t)
        vy = radius * np.cos(angular_velocity * t)

        true_state = np.array([x, y, vx, vy])
        true_trajectory.append(true_state)

        measurement = true_state[:2] + np.random.normal(0, measurement_noise, 2)
        measurements.append(measurement)

        kf.prediction()
        estimate = kf.update(measurement)
        estimates.append(estimate[:2])
    
    true_trajectory = np.array(true_trajectory)
    measurements = np.array(measurements)
    estimates = np.array(estimates)

    plt.figure(figsize=(10,8))
    plt.plot(true_trajectory[:,0], true_trajectory[:,1], 'g-',
             label = 'Истинная траектория', linewidth = 2)
    plt.plot(measurements[:,0], measurements[:,1], 'ro',
             label = 'Измерения', markersize = 2, alpha = 0.5)
    plt.plot(estimates[:,0], estimates[:,1], 'b-',
             label = 'Оценка Калмана', linewidth = 2)
    plt.xlabel('X координата')
    plt.ylabel('Y координата')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

#example_2d_tracking()
