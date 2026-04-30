#!/usr/bin/env python3
import numpy as np, kiamdb, pyorbs
from datetime import timedelta, datetime
from sqlalchemy import select

from filters import UKF, LKF, EKF

obj, t0 = 40258, datetime(2026, 3, 7)
t = t0 + timedelta(days = 8)

sigma_pos = 0
P_const = np.diag([1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5])

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

    return init_orbit.state_v.copy(), init_orbit.time, P_initial, ctx.meas_data.sort_values('time'), ctx

def check_residuals(state, meas, t):
    orb = pyorbs.pyorbs.orbit()
    orb.state_v, orb.time = state, t #pyorbs.pyorbs.ephem_time(meas.iloc[0]['time'].to_pydatetime())
    orb.setup_parameters()
    orb.change_param({'calc_partials': True})
    step = pyorbs.pyorbs_det.od_step(orb, meas)
    pyorbs.vis.plot_res(step.meas_tab)

def LSM(v, t, ctx):
    orb = pyorbs.pyorbs.orbit()
    orb.state_v, orb.time = v, t
    orb.setup_parameters()
    orb.change_param({'calc_partials': True})
    ctx.single_od(orb)
    plt = pyorbs.vis.plot_res(ctx.current_step.meas_tab)
    plt.savefig('LSM.png    ')

def main():
    v0, t0, P0, meas, ctx = get_initial_params()

    v0 += np.array([sigma_pos, 0, 0, 0, 0, 0])
    P0 = np.zeros([6,6])
    P0[0,0] = sigma_pos ** 2
    P0 += P_const
    print(v0)
    LSM(v0, t0, ctx)
    filter = LKF(t_begin=t0, v=v0, P=P0, meas=meas, attempts=7)
    filter.od_filtration()

    #mat = pyorbs.bal.to_rnb_mat(vec2)
    #print(f'{mat @ dv[:3] * 1e3} км')
    check_residuals(filter.state_v, meas, t0)

if __name__ == "__main__":
    main()


"""
[-1.23687151e-05  5.42418125e-04  2.82455218e-07  1.58082969e-05 1.55969414e-05  4.21929570e-06]
[ 6.85820986e-04  6.47531501e-05  3.98946224e-05  3.42704044e-05 2.89017980e-05 -7.51855080e-06]
[ 1.42851228e-06 -1.02634209e-06 -2.43914756e-07 -7.12612064e-08 1.10491381e-07 -1.78970300e-08]
[-4.07093911e-08 -4.56040154e-08  4.24892261e-09 -3.80450752e-09 -1.12210145e-09  1.87868283e-10]
[ 2.90033707e-04  1.23672776e-04  1.38065272e-05  1.20904599e-05 1.90489590e-05 -4.48917201e-07]
[ 1.64071735e-07 -6.43647975e-08 -8.83488213e-09 -3.27193451e-09 1.12677349e-08 -1.36844137e-11]
[-8.22282165e-07  5.15401925e-08  2.59026843e-08 -4.97279888e-09 -5.86271963e-08  1.15718906e-10]
"""