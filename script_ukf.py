import numpy as np
from typing import List
from datetime import timedelta, datetime
from sqlalchemy import select

from models import UKF, smoothing

from kiamdb.orbits import OrbitSolution, SessionOrbits
from kiamdb.od import ContextOD
from pyorbs.pyorbs import orbit

def get_initial_params(obj_id: int) \
    -> tuple[np.ndarray[np.float64], List[np.float64] | None, datetime | None]:
    
    sq = select(OrbitSolution).where(
        OrbitSolution.obj_id == obj_id
    ).order_by(OrbitSolution.time_obtained.asc()).limit(1)
        
    with SessionOrbits() as session:
        res = session.scalar(sq)

    if res is not None:
        b_initial = np.array(res.state)
        P_initial = res.cov
        epoch = res.epoch

    return b_initial, P_initial, epoch

def main():
    obj_id = 26629
    x0, P0, t_start = get_initial_params(obj_id = obj_id)
    if t_start is not None:
        t_end = t_start + timedelta(days = 15)

    ukf = UKF(
        t_start = t_start,
        P = np.array(P0).reshape(7,7)[:6,:6],
        x = orbit(x = x0, time = t_start).state_v,
        alpha = 0.5, kappa = -3.0
    )
    print(ukf.state_v)
    ctx = ContextOD(obj_id = obj_id, initial_orbit = x0, 
                    t_start = t_start, t_stop = t_end) 
    meas = ctx.meas_data

    forward_states = []
    forward_covs = []

    for m in meas.iterrows():
        t = m['time']
        ukf.step(m, t)

        forward_states.append(ukf.state_v)
        forward_covs.append(ukf.cov_matrix)
        print(f'Уточнились на {t}: {ukf.state_v}')

    smoothing_states, smoothing_covs = smoothing(ukf, forward_states, forward_covs, meas)
    
    time_list = meas['time'].tolist()
    for l in range(len(smoothing_states)):

        print(f'Время {time_list[l]}:\n'
              f'Оценка вектора состояния: {smoothing_states[l]}\n'
              f'Оценка ковариационной матрицы: {smoothing_covs[l]}')

if __name__ == "__main__":
    main()