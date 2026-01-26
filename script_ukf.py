import numpy as np
from datetime import timedelta, datetime
from sqlalchemy import select, cast, DateTime
from models import UKF, SquareRootUKF

from kiamdb.orbits import OrbitSolution, SessionOrbits
from kiamdb.od import ContextOD
from pyorbs.pyorbs import orbit
from pyorbs.pyorbs_det import od_step
from pyorbs.vis import plot_res
import _config

def get_initial_params(obj, t_start):
    sq = select(OrbitSolution).where(
        OrbitSolution.obj_id == obj,
        cast(OrbitSolution.epoch, DateTime).between(t_start, t_start + timedelta(days = 15))
        ).order_by(OrbitSolution.time_obtained.asc()).limit(1)
        
    with SessionOrbits() as session:
        res = session.scalar(sq)

    if np.shape(res.cov) == (49,):
        P_initial = np.array(res.cov).reshape(7,7)[:6,:6]
    else:
        P_initial = np.array(res.cov).reshape(6,6)

    return np.array(res.state), P_initial, res.epoch

def check_residuals(state, meas):
    orb = orbit(x = state, time = meas.iloc[-1]['time'].to_pydatetime())
    step = od_step(orb, meas)
    plot_res(step.meas_tab)

def main():
    obj = 43109
    t_start = datetime(2025, 12, 1)
    t = t_start + timedelta(days = 30)

    x0, P0, t0 = get_initial_params(obj, t_start)
    ctx = ContextOD(obj_id = obj, initial_orbit = x0, t_start = t_start, t_stop = t)
    #ctx.single_od()
    #plot_res(ctx.current_step.meas_tab)
    meas = ctx.meas_data.sort_values('time')
    x0 += np.array([0.01, 0.00, 0.0, 0.0, 0.0, 0.0])
    filter = SquareRootUKF(t_start = t0, x = orbit(x = x0, time = t0).state_v, P = P0)
    """for i in range(len(meas) // _config.LEN_BLOCK_MEAS):
        block_meas = meas[i * _config.LEN_BLOCK_MEAS : (i+1) * _config.LEN_BLOCK_MEAS]
        filter.new_filter_step(block_meas)"""

    for _, m in meas.iterrows():
        t_k = m['time'].to_pydatetime()
        filter.step(m, t_k)
        print(filter.state_v)
        print(f'Коррекция: {t_k}')

    #smoothing_states, smoothing_covs = filter.rts_smoother()
    #check_residuals(filter.state_v, meas)
    filter.draw_position_std()

if __name__ == "__main__":
    main()