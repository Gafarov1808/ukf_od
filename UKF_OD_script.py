import numpy as np
from UKF_class import UKF
from kiamdb.orbits import OrbitSolution, OrbitStr, SessionOrbits
from datetime import timedelta, datetime
from sqlalchemy import select
from pyorbs.pyorbs import orbit
from kiamdb.od import ContextOD

def get_initial_params(obj_id: int):
    sq = select(OrbitSolution).where(
        OrbitSolution.obj_id == obj_id
    ).order_by(OrbitSolution.time_obtained.asc()).limit(1)
        
    with SessionOrbits() as session:
        res = session.scalar(sq)

    b_initial = np.array(res.state)
    P_initial = res.cov
    epoch = res.epoch

    return b_initial, P_initial, epoch

def main():
    obj_id = 56703
    dim_x = 6
    dt = 3600

    x0, P0, t_start = get_initial_params(obj_id = obj_id)
    t_end = t_start + timedelta(days = 30)

    num_steps = int((t_end - t_start).total_seconds() / dt)
    Q = np.zeros((dim_x, dim_x))
    R = np.eye(1) * 5e-6
    x0 = orbit(x = x0, time = t_start)
    P0 = np.array(P0).reshape(7,7)[:6,:6]

    ukf = UKF(
        t_start = t_start,
        P = P0, x = x0.state_v, Q = Q, R = R,
        alpha = 1e-3, beta = 2.0, kappa = 0.0)

    ctx = ContextOD(obj_id = obj_id, initial_orbit = x0, 
                    t_start = t_start, t_stop = t_end)
    meas = ctx.meas_data

    states_history = np.zeros((num_steps, dim_x))
    cov_history = np.zeros((num_steps, dim_x, dim_x))
    true_states = np.zeros((num_steps, dim_x))
    measurements = np.zeros((num_steps, 1))

    #true_state = np.array(x0).copy()
    for k in range(num_steps):
        #true_state = ... #зададим
        #true_state += np.random.multivariate_normal(np.zeros(dim_x), Q * 0.1)
        #true_states[k] = true_state

        #true_range = np.linalg.norm(true_state[:3])
        #measurement = true_range + np.random.normal(0, np.sqrt(R[0,0]))
        #measurements[k] = measurement

        #state_est, cov_est = ukf.step(measurement, k+1, dt)
        ukf.step(meas, k+1, dt)
        #states_history[k] = state_est
        #cov_history[k] = cov_est

        #print(f'Шаг {k+1}: Оценка положения = {state_est[:3]}, измерение = {measurement[0]:.2f}')
        print(f'Шаг {k+1}')

if __name__ == "__main__":
    main()