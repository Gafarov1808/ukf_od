import numpy as np
from typing import List
from datetime import timedelta, datetime
from sqlalchemy import select
from models import UKF

from kiamdb.orbits import OrbitSolution, SessionOrbits
from kiamdb.od import ContextOD
from pyorbs.pyorbs import orbit

def get_initial_params(obj_id: int) \
    -> tuple[np.ndarray, List[np.float64], datetime] | None:
    
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
    t_end = t_start + timedelta(days = 15)

    ukf = UKF(
        t_start = t_start,
        P = np.array(P0).reshape(7,7)[:6,:6],
        x = orbit(x = x0, time = t_start).state_v,
        alpha = 0.52
    )
    ctx = ContextOD(obj_id = obj_id, initial_orbit = x0, 
                    t_start = t_start, t_stop = t_end) 
    meas = ctx.meas_data

    for _, m in meas.iterrows():
        t = m['time'].to_pydatetime()
        ukf.step(m, t)
        print(f'Уточнились на {t}')
    #ukf.draw_plots()
    smoothing_states, smoothing_covs = ukf.rts_smoother()

if __name__ == "__main__":
    main()