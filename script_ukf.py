import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from sqlalchemy import select, cast, DateTime

from models import LinearKalman, SquareRootUKF

from kiamdb.orbits import OrbitSolution, SessionOrbits
from kiamdb.od import ContextOD
from pyorbs.pyorbs import orbit, ephem_time
from pyorbs.pyorbs_det import od_step
from pyorbs.vis import plot_res

sigma_pos = 0.1
sigma_v = 0
P_const = np.diag([1e-5, 1e-5, 1e-5, 1e-10, 1e-10, 1e-10])

def get_initial_params(obj: int, t_start: datetime) -> tuple[orbit, np.ndarray]:
    sq = select(OrbitSolution).where(
        OrbitSolution.obj_id == obj,
        cast(OrbitSolution.epoch, DateTime).between(t_start, t_start + timedelta(days = 15))
        ).order_by(OrbitSolution.time_obtained.asc()).limit(1)
        
    with SessionOrbits() as session:
        res = session.scalar(sq)

    if res is not None:
        if len(res.cov) == 49:
            P_initial = np.array(res.cov).reshape(7,7)[:6,:6]
        else:
            P_initial = np.array(res.cov).reshape(6,6)
    else:
        print('Не удалось найти объект с данными параметрами. Завершение')
        exit()

    init_orbit = orbit()
    init_orbit.state_v, init_orbit.time = np.array(res.state), ephem_time(res.epoch)
    
    return init_orbit, P_initial

def check_residuals(state: np.ndarray, meas: pd.DataFrame, t: ephem_time):
    orb = orbit()
    orb.state_v, orb.time = state, t # ephem_time(meas.iloc[0]['time'].to_pydatetime())
    orb.setup_parameters()
    orb.change_param({'calc_partials': True})
    step = od_step(orb, meas)
    plot_res(step.meas_tab)

def LSM(orb: orbit, ctx: ContextOD) -> None:
    orb.setup_parameters()
    orb.change_param({'calc_partials': True})
    ctx.single_od()
    plot_res(ctx.current_step.meas_tab)

def main():
    obj = 43087 #43109
    t0 = datetime(2026, 1, 1)
    t = t0 + timedelta(days = 28)
    orb, P0 = get_initial_params(obj, t0)

    print(f'correct state = {orb.state_v}')
    orb.state_v += np.array([sigma_pos, 0, 0, sigma_v, sigma_v, sigma_v])

    ctx = ContextOD(obj_id = obj, initial_orbit = orb, t_start = t0, t_stop = t)
    meas = ctx.meas_data.sort_values('time')

    P0 = np.zeros([6,6])
    P0[0,0] = sigma_pos ** 2
    P0[3,3] = P0[4, 4] = P0[5, 5] = sigma_v ** 2
    P0 += P_const

    #LSM(orb, ctx)
    filter = SquareRootUKF(t_start = orb.time, v = orb.state_v, 
                                P = P0, meas = meas, lim_filter = 1)
    filter.several_filtrations(orb.time, P0)
    #filter = LinearKalman(t_start = orb.time, v = orb.state_v, P = P0, meas = meas)
    #filter.filtration()

    check_residuals(filter.state_v, meas, orb.time)

if __name__ == "__main__":
    main()