import numpy as np
from datetime import timedelta, datetime
from sqlalchemy import select, cast, DateTime
from models import UKF

from kiamdb.orbits import OrbitSolution, SessionOrbits
from kiamdb.od import ContextOD
from pyorbs.pyorbs import orbit

def get_initial_params(obj_id: int):
    
    sq = select(OrbitSolution).where(
        OrbitSolution.obj_id == obj_id,
        cast(OrbitSolution.epoch, DateTime).between(datetime(2025,9,1), datetime(2025,9,2))  
    ).order_by(OrbitSolution.time_obtained.desc()).limit(1)
        
    with SessionOrbits() as session:
        res = session.scalar(sq)

    if res is not None:
        b_initial = np.array(res.state)
        P_initial = res.cov
        epoch = res.epoch
        return b_initial, P_initial, epoch

def main():
    obj_id = 58356
    x0, P0, t_start = get_initial_params(obj_id = obj_id)
    t_end = t_start + timedelta(days = 4)

    ctx = ContextOD(obj_id = obj_id, initial_orbit = x0, 
                    t_start = t_start, t_stop = t_end) 
    meas = ctx.meas_data
    print(meas)
    ukf = UKF(
        t_start = meas['time'][0].to_pydatetime(),
        P = np.array(P0).reshape(7,7)[:6,:6],
        x = orbit(x = x0, time = t_start).state_v)
    measurements = [row for _, row in meas.iterrows()] 
    times = meas['time'].tolist()[1:] + [None]
    for m, t in zip(measurements, times):
        if t is not None:
            t = t.to_pydatetime()
            ukf.step(m, t)
            print(f'Уточнились на {t}')
    ukf.draw_plots()
    smoothing_states, smoothing_covs = ukf.rts_smoother()

if __name__ == "__main__":
    main()