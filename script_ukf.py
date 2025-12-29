import numpy as np
from datetime import timedelta, datetime
from sqlalchemy import select, cast, DateTime
from models import UKF, SquareRootUKF

from kiamdb.orbits import OrbitSolution, SessionOrbits
from kiamdb.od import ContextOD
from pyorbs.pyorbs import orbit
from pyorbs.pyorbs_det import od_step
from pyorbs.vis import plot_res

def get_initial_params():
    sq = select(OrbitSolution).where(
        OrbitSolution.obj_id == 41105,
        cast(OrbitSolution.epoch, DateTime).between(datetime(2025, 7, 1), datetime(2025, 7, 10))  
    ).order_by(OrbitSolution.time_obtained.desc()).limit(1)
        
    with SessionOrbits() as session:
        res = session.scalar(sq)

    obj = res.obj_id
    b_initial = np.array(res.state)
    P_initial = np.array(res.cov).reshape(6,6) #np.array(res.cov).reshape(7,7)[:6,:6]
    epoch = res.epoch
    return obj, b_initial, P_initial, epoch

def check_residuals(state, meas):
    orb = orbit(x = state, time = meas.iloc[-1]['time'].to_pydatetime())
    step = od_step(orb, meas)
    plot_res(step.meas_tab)

def main():
    obj, x0, P0, t0 = get_initial_params()
    t = t0 + timedelta(days = 10)

    ctx = ContextOD(obj_id = obj, initial_orbit = x0, t_start = t0, t_stop = t)
    meas = ctx.meas_data.sort_values('time')

    filter = SquareRootUKF(t_start = t0, P = P0, 
                                x = orbit(x = x0, time = t0).state_v)
    for _, m in meas.iterrows():
        t_k = m['time'].to_pydatetime()
        filter.step(m, t_k)
        print(f'Уточнились на {t_k}')
    
    smoothing_states, smoothing_covs = filter.rts_smoother()
    filter.draw_position_std(smoothing_covs)
    check_residuals(smoothing_states[-1], meas)

    state = orbit(x = x0, time = t0)
    ctx.single_od(state )

if __name__ == "__main__":
    main()