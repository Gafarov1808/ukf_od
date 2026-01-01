import numpy as np
from datetime import timedelta, datetime
from sqlalchemy import select, cast, DateTime
from models import SquareRootUKF

from kiamdb.orbits import OrbitSolution, SessionOrbits
from kiamdb.od import ContextOD
from pyorbs.pyorbs import orbit
from pyorbs.pyorbs_det import od_step
from pyorbs.vis import plot_res

def get_initial_params():
    sq = select(OrbitSolution).where(
        OrbitSolution.obj_id == 43109,
        cast(OrbitSolution.epoch, DateTime).between(datetime(2025, 12, 1), datetime(2025, 12, 30))
        ).order_by(OrbitSolution.time_obtained.desc()).limit(1)
        
    with SessionOrbits() as session:
        res = session.scalar(sq)

    if np.shape(res.cov) == (49,):
        P_initial = np.array(res.cov).reshape(7,7)[:6,:6]
    else:
        P_initial = np.array(res.cov).reshape(6,6)

    return res.obj_id, np.array(res.state), P_initial, res.epoch

def check_residuals(state, meas):
    orb = orbit(x = state, time = meas.iloc[-1]['time'].to_pydatetime())
    step = od_step(orb, meas)
    plot_res(step.meas_tab)

def main():
    obj, x0, P0, t0 = get_initial_params()
    t_start = datetime(2025, 12, 1)
    t = t_start + timedelta(days = 30)
    ctx = ContextOD(obj_id = obj, initial_orbit = x0, t_start = t_start, t_stop = t)
    #ctx.single_od()
    #plot_res(ctx.current_step.meas_tab)
    meas = ctx.meas_data.sort_values('time')

    filter = SquareRootUKF(t_start = t0, P = P0, x = orbit(x = x0, time = t0).state_v)
    for _, m in meas.iterrows():
        t_k = m['time'].to_pydatetime()
        filter.step(m, t_k)
        print(f'Уточнились на {t_k}')
    
    smoothing_states, smoothing_covs = filter.rts_smoother()
    filter.draw_position_std(smoothing_covs)
    check_residuals(smoothing_states[-1], meas)

if __name__ == "__main__":
    main()