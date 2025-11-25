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
        OrbitSolution.obj_id == 20026,
        cast(OrbitSolution.epoch, DateTime).between(datetime(2025,10,20), datetime(2025,11,2))  
    ).order_by(OrbitSolution.time_obtained.desc()).limit(1)
        
    with SessionOrbits() as session:
        res = session.scalar(sq)

    obj = res.obj_id
    b_initial = np.array(res.state)
    P_initial = np.array(res.cov).reshape(7,7)[:6,:6] #np.array(res.cov).reshape(6,6) 
    epoch = res.epoch
    return obj, b_initial, P_initial, epoch

def check_residuals(state, meas):
    orb = orbit(x = state, time = meas.iloc[-1]['time'].to_pydatetime())
    step = od_step(orb, meas)
    plot_res(step.meas_tab)

def main():
    obj, x0, P0, t_start = get_initial_params()
    t_end = t_start + timedelta(days = 15)

    ctx = ContextOD(obj_id = obj, initial_orbit = x0, 
                    t_start = t_start, t_stop = t_end) 
    meas = ctx.meas_data
    #x0 += np.array([1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6])
    filter = SquareRootUKF(
        t_start = t_start,
        x = orbit(x = x0, time = t_start).state_v,
        P = P0
    )

    for _, m in meas.iterrows():
        t = m['time'].to_pydatetime()
        filter.step(m, t)
        print(filter.state_v)
        print(f'Уточнились на {t}')
        
    smoothing_states, smoothing_covs = filter.rts_smoother()
    filter.draw_plots(smoothing_states, smoothing_covs)
    check_residuals(smoothing_states[-1], meas)

if __name__ == "__main__":
    main()