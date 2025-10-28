import numpy as np
from datetime import timedelta
from sqlalchemy import select

from UKF_class import UKF

from kiamdb.orbits import OrbitSolution, SessionOrbits
from kiamdb.od import ContextOD
from pyorbs.pyorbs import orbit

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
    obj_id = 26629
    dim_x = 6

    x0, P0, t_start = get_initial_params(obj_id = obj_id)
    t_end = t_start + timedelta(days = 30)

    Q = np.zeros((dim_x, dim_x))
    R = np.eye(1) * 5e-6
    x0 = orbit(x = x0, time = t_start)
    P0 = np.array(P0).reshape(7,7)[:6,:6]

    ukf = UKF(
        t_start = t_start,
        P = P0, Q = Q, R = R,
        x = x0.state_v, 
        alpha = 0.4
    )

    ctx = ContextOD(obj_id = obj_id, initial_orbit = x0, 
                    t_start = t_start, t_stop = t_end)
    
    meas = ctx.meas_data
    meas_time = meas['time'].tolist()

    for m, t in zip(meas, meas_time):
        ukf.step(m, t)
        print(f'Уточнились на {t}: Оценка вектора состояния = {ukf.state}, '
              f'оценка ковариационной матрицы = {ukf.cov_matrix}')

if __name__ == "__main__":
    main()