import numpy as np
from pyorbs.pyorbs import orbit, ephem_time, load
from datetime import datetime

_SEC2RAD = np.pi / 180. / 3600.

orb = orbit()
#orb = load('test_orbit')
orb.state_v = np.array([20.20687328, 0.89828793, 19.94422809, -1.66755229,  2.98893043, 1.57641073])
orb.time = ephem_time(datetime(2025,12,1,17,49,36,950000))
orb.change_param({'calc_partials': True})
orb.set_initial_point(orb.time)

cov = np.array([
    [ 4.62198741e-04, -2.36402971e-06,  1.04013410e-05, -1.20072825e-08, -7.62372924e-09,  1.12828091e-07],
    [-2.36402971e-06,  9.57410809e-06, -1.01219303e-07, -1.24129069e-10, -1.36913665e-09, -8.27802444e-10],
    [ 1.04013410e-05, -1.01219303e-07,  9.95374010e-06,  2.11863584e-09, -3.66222414e-10,  4.82542060e-09],
    [-1.20072825e-08, -1.24129069e-10,  2.11863584e-09,  1.07590074e-10,  4.74171897e-12, -7.28588109e-11],
    [-7.62372924e-09, -1.36913665e-09, -3.66222414e-10,  4.74171897e-12,  1.03459126e-10, -4.96543503e-11],
    [ 1.12828091e-07, -8.27802444e-10,  4.82542060e-09, -7.28588109e-11, -4.96543503e-11,  8.63204766e-10]
  ])

print(orb.state_v[:6], orb.time.utc())
t = datetime(2025, 12, 3, 12, 28, 6, 700000)
orb.move(t)
F = orb.state_v[orb.structure['calc_partials']].reshape(6,6)
cov = F @ cov @ F.T
print(orb.state_v[:6], orb.time.utc())
#print(cov)
print(f'std = {np.sqrt(cov[:3, :3].diagonal())}')

