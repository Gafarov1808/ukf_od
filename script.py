#!/usr/bin/env python3
import numpy as np, kiamdb, pyorbs
from datetime import timedelta, datetime
from sqlalchemy import select

from filters import UKF, LKF, EKF

obj, t0 = 40258, datetime(2026, 3, 8)
t = t0 + timedelta(days = 7)

sigma_pos = 0
P_const = np.diag([1e10, 1e10, 1e10, 1e-5, 1e-5, 1e-5])
lim = 1

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
    ctx = kiamdb.od.ContextOD(obj_id = obj, initial_orbit = init_orbit, 
                                    t_start = t0, t_stop = t, mle_limit = lim)

    return init_orbit.state_v.copy(), init_orbit.time, P_initial, ctx.meas_data.sort_values('time'), ctx

def check_residuals(state, meas, t):
    orb = pyorbs.pyorbs.orbit()
    orb.state_v, orb.time = state, pyorbs.pyorbs.ephem_time(meas.iloc[0]['time'].to_pydatetime())
    orb.setup_parameters()
    orb.change_param({'calc_partials': True})
    step = pyorbs.pyorbs_det.od_step(orb, meas)
    plt = pyorbs.vis.plot_res(step.meas_tab)
    plt[0].savefig('LKF.png', dpi = 1000)

def LSM(v, t, ctx):
    orb = pyorbs.pyorbs.orbit()
    orb.state_v, orb.time = v, t
    orb.setup_parameters()
    orb.change_param({'calc_partials': True})
    orb.set_initial_point(t)
    ctx.single_od(orb)
    plt = pyorbs.vis.plot_res(ctx.current_step.meas_tab)
    plt[0].savefig('LSM.png', dpi = 1000)
    print("=" * 60)

def main():
    v0, t0, P0, meas, ctx = get_initial_params()

    v0 += np.array([sigma_pos, 0, 0, 0, 0, 0])
    P0 = np.zeros([6,6])
    P0[0,0] = sigma_pos ** 2
    P0 += P_const
    #LSM(v0, t0, ctx)
    filter = LKF(t_begin=t0, v=v0, P=P0, meas=meas, attempts=lim)
    filter.od_filtration()

    #mat = pyorbs.bal.to_rnb_mat(vec2)
    #print(f'{mat @ dv[:3] * 1e3} км')
    check_residuals(filter.state_v, meas, t0)

if __name__ == "__main__":
    main()
