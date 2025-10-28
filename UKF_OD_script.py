import numpy as np
from datetime import timedelta
from sqlalchemy import select

from UKF_class import UKF

from kiamdb.orbits import OrbitSolution, SessionOrbits
from kiamdb.od import ContextOD
from pyorbs.pyorbs import orbit

def get_initial_params(obj_id: int):
    sq = select(OrbitSolution).where(
        OrbitSolution.obj_id == obj_id
    ).order_by(OrbitSolution.time_obtained.asc()).limit(1)
        
    with SessionOrbits() as session:
        res = session.scalar(sq)

    b_initial = np.array(res.state)
    P_initial = res.cov
    epoch = res.epoch

    return b_initial, P_initial, epoch

def main():
    obj_id = 26629
    dim_x = 6

    x0, P0, t_start = get_initial_params(obj_id = obj_id)
    t_end = t_start + timedelta(days = 30)

    Q = np.zeros((dim_x, dim_x))
    R = np.eye(1) * 5e-6
    x0 = orbit(x = x0, time = t_start)
    P0 = np.array(P0).reshape(7,7)[:6,:6]

    ukf = UKF(
        t_start = t_start,
        P = P0, Q = Q, R = R,
        x = x0.state_v, 
        alpha = 0.5200000100000001, #0.52  
        kappa = -6.700000000000005e-07 #-3.0
    )
    #print(ukf.state)
    ctx = ContextOD(obj_id = obj_id, initial_orbit = x0, 
                    t_start = t_start, t_stop = t_end) 
    meas = ctx.meas_data
    ukf.res_z = 1
    #0.5200000200000001 -1.1399999999999901e-05
    #res = [ 1.09313964 -0.35062547]

    alpha = 0.5200000100000001
    kappa = -6.700000000000005e-07
    for index, m in meas.iterrows():
        t = m['time']
        for k in range (0, 1000000):
            alpha += 1e-8
            for j in range (0, 10000):
                kappa -= 1e-8
                print(ukf.alpha, ukf.kappa)
                try:
                    ukf = UKF(
                        t_start = t_start,
                        P = P0, Q = Q, R = R,
                        x = x0.state_v,
                        alpha= alpha, kappa = kappa 
                    )
                    ukf.step(m, t)
                    print(f'res = {ukf.res_z}')
                    if (np.linalg.norm(ukf.res_z) < 1e-6):
                        print(f'Успех! {ukf.alpha}, {ukf.kappa}')
                        exit()
                except:
                    pass

        print(f'Уточнились на {t}:\n'
              f'Оценка вектора состояния = {ukf.state}\n'
              f'Oценка ковариационной матрицы = {ukf.cov_matrix}')

if __name__ == "__main__":
    main()