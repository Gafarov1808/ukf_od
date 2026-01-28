import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from sqlalchemy import select, cast, DateTime
from models import SquareRootUKF

from kiamdb.orbits import OrbitSolution, SessionOrbits
from kiamdb.od import ContextOD
from pyorbs.pyorbs import orbit, ephem_time
from pyorbs.pyorbs_det import od_step
from pyorbs.vis import plot_res

def get_initial_params(obj:int, t_start: datetime)->tuple[orbit, np.ndarray] | None:
    sq = select(OrbitSolution).where(
        OrbitSolution.obj_id == obj,
        cast(OrbitSolution.epoch, DateTime).between(t_start, t_start + timedelta(days = 15))
        ).order_by(OrbitSolution.time_obtained.asc()).limit(1)
        
    with SessionOrbits() as session:
        res = session.scalar(sq)

    if res is not None:
        if np.shape(res.cov) == (49,):
            P_initial = np.array(res.cov).reshape(7,7)[:6,:6]
        else:
            P_initial = np.array(res.cov).reshape(6,6)
    else:
        print('Не удалось найти объект с данными параметрами. Завершение')
        return

    init_orbit = orbit()
    init_orbit.state_v, init_orbit.time = np.array(res.state), ephem_time(res.epoch)
    return init_orbit, P_initial

def check_residuals(state: np.ndarray, meas: pd.DataFrame):
    orb = orbit(x = state, time = ephem_time(meas.iloc[-1]['time'].to_pydatetime()))
    step = od_step(orb, meas)
    plot_res(step.meas_tab)

def main():
    obj = 43109
    t_start = datetime(2025, 12, 1)
    t = t_start + timedelta(days =28)

    init_orbit, P0 = get_initial_params(obj, t_start)
    ctx = ContextOD(obj_id = obj, initial_orbit = init_orbit, t_start = t_start, t_stop = t)
    #ctx.single_od()
    #plot_res(ctx.current_step.meas_tab)
    meas = ctx.meas_data.sort_values('time')
    init_orbit.state_v += np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    filter = SquareRootUKF(t_start = init_orbit.time, x = init_orbit.state_v, P = P0)
    
    meas['date'] = meas['time'].dt.date
    daily_group = meas.groupby('date')
    daily_dfs = {str(date): group.drop(columns=['date']) 
             for date, group in daily_group}
    for _, block_meas in daily_dfs.items():
        filter.new_filter_step(block_meas)
        print(filter.state_v)




    for _, m in meas.iterrows():
        t_k = ephem_time(m['time'].to_pydatetime())
        filter.step(m, t_k)
        print(f'Коррекция: {t_k}')

    smoothing_states, smoothing_covs = filter.rts_smoother()
    check_residuals(smoothing_states[-1], meas)
    #filter.draw_position_std(smoothing_covs)

if __name__ == "__main__":
    main()