#!/usr/bin/env python3
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from sqlalchemy import select 

from filters import UKF, EKF, LKF

from kiamdb.orbits import OrbitSolution, SessionOrbits
from kiamdb.od import ContextOD
from pyorbs.pyorbs import orbit, ephem_time
from pyorbs.pyorbs_det import od_step
from pyorbs.vis import plot_res

obj = 40258
t0 = datetime(2026, 2, 17)
t = t0 + timedelta(days = 25)

sigma_pos = 0.1
sigma_v = 0
P_const = np.diag([1e-5, 1e-5, 1e-5, 1e-10, 1e-10, 1e-10])

def get_initial_params() -> tuple[np.ndarray[np.float64], ephem_time, 
                                  np.ndarray[np.float64], pd.DataFrame]:
    sq = select(OrbitSolution).where(OrbitSolution.id == 1277174)
        
    with SessionOrbits() as session:
        res = session.scalar(sq)

    if res is not None:
        if len(res.cov) == 49:
            P_initial = np.array(res.cov).reshape(7,7)[:6,:6]
        else:
            P_initial = np.array(res.cov).reshape(6,6)
    else:
        print('Не удалось найти объект с данными параметрами. Завершение программы.')
        exit()

    init_orbit = orbit()
    init_orbit.state_v, init_orbit.time = np.array(res.state), ephem_time(res.epoch)
    ctx = ContextOD(obj_id = obj, initial_orbit = init_orbit, t_start = t0, t_stop = t)
    v = init_orbit.state_v.copy()

    return v, init_orbit.time, P_initial, ctx.meas_data.sort_values('time')

def check_residuals(state: np.ndarray[np.float64], meas: pd.DataFrame, time: ephem_time):
    orb = orbit()
    orb.state_v, orb.time = state, time #ephem_time(meas.iloc[0]['time'].to_pydatetime())
    orb.setup_parameters()
    orb.change_param({'calc_partials': True})
    step = od_step(orb, meas)
    plot_res(step.meas_tab)

def main():
    v0, t_start, P0, meas = get_initial_params()
    v0 += np.array([sigma_pos, 0, 0, sigma_v, sigma_v, sigma_v])
    
    P0 = np.zeros([6,6])
    P0[0,0] = sigma_pos ** 2
    P0[3,3] = P0[4, 4] = P0[5, 5] = sigma_v ** 2
    P0 += P_const

    filter = UKF(t_start=t_start, v=v0, P=P0, meas=meas)
    filter.several_filtrations(t_start, P0)

    #mat = pyorbs.bal.to_rnb_mat(vec2)
    #print(f'{mat @ dv[:3] * 1e3} км')

    check_residuals(filter.state_v, meas, t_start)

if __name__ == "__main__":
    main()
