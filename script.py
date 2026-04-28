#!/usr/bin/env python3
import numpy as np, kiamdb, pyorbs
from datetime import timedelta, datetime
from sqlalchemy import select

from filters import UKF, LKF, EKF

obj, t0 = 40258, datetime(2026, 3, 10)
t = t0 + timedelta(days = 5)

sigma_pos = 0
P_const = np.diag([1e-5, 1e-5, 1e-5, 1e-8, 1e-8, 1e-8])

def get_initial_params():
    sq = select(kiamdb.orbits.OrbitSolution).where(kiamdb.orbits.OrbitSolution.id == 1277174)
    with kiamdb.orbits.SessionOrbits() as session:
        res = session.scalar(sq)

    if len(res.cov) == 49:
        P_initial = np.array(res.cov).reshape(7,7)[:6,:6]
    else:
        P_initial = np.array(res.cov).reshape(6,6)

    init_orbit = pyorbs.pyorbs.orbit()
    init_orbit.state_v, init_orbit.time = np.array(res.state), pyorbs.pyorbs.ephem_time(res.epoch)
    ctx = kiamdb.od.ContextOD(obj_id = obj, initial_orbit = init_orbit, t_start = t0, t_stop = t)

    return init_orbit.state_v.copy(), init_orbit.time, P_initial, ctx.meas_data.sort_values('time')

def check_residuals(state, meas, time):
    orb = pyorbs.pyorbs.orbit()
    orb.state_v, orb.time = state, time #pyorbs.pyorbs.ephem_time(meas.iloc[0]['time'].to_pydatetime())
    orb.setup_parameters()
    orb.change_param({'calc_partials': True})
    step = pyorbs.pyorbs_det.od_step(orb, meas)
    pyorbs.vis.plot_res(step.meas_tab)

def main():
    v0, t_start, P0, meas = get_initial_params()

    v0 += np.array([sigma_pos, 0, 0, 0, 0, 0])
    P0 = np.zeros([6,6])
    P0[0,0] = sigma_pos ** 2
    P0 += P_const
    print(v0)
    filter = LKF(t_begin=t_start, v=v0, P=P0, meas=meas, attempts=3)
    filter.od_filtration()

    #mat = pyorbs.bal.to_rnb_mat(vec2)
    #print(f'{mat @ dv[:3] * 1e3} км')
    check_residuals(filter.state_v, meas, t_start)

if __name__ == "__main__":
    main()
