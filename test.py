import numpy as np
from pyorbs.pyorbs import orbit, ephem_time
from datetime import datetime

_SEC2RAD = np.pi / 180. / 3600.

orb = orbit()
orb.state_v = np.array([22.11068642, -3.09987973, 17.53192939,-1.17270878,2.96316509,2.01474894]) 
orb.time = ephem_time(datetime(2025,12,1,17,27,17))

#orb.change_param({'calc_partials': True})
#orb.set_initial_point(orb.time)
'''cov = np.array([[ 6.29303426e-04, -5.80808134e-07,  4.86962339e-06,  8.57543995e-08,
   1.09986662e-08,  2.42027251e-07],
 [-5.80808134e-07,  9.71127319e-06, -2.85697493e-08, -1.66393247e-10,
  -1.66446914e-09, -4.25600825e-10],
 [ 4.86962339e-06, -2.85697493e-08,  9.93678122e-06,  3.77692748e-09,
  -1.21850590e-10,  5.62322405e-09],
 [ 8.57543995e-08, -1.66393247e-10,  3.77692748e-09,  2.85951731e-11,
  -3.49283676e-12, -4.76341828e-11],
 [ 1.09986662e-08, -1.66446914e-09, -1.21850590e-10, -3.49283676e-12,
   1.07881460e-10, -5.87178311e-11],
 [ 2.42027251e-07, -4.25600825e-10,  5.62322405e-09, -4.76341828e-11,
  -5.87178311e-11,  1.61827568e-09]]
)'''

t = datetime(2025,12,26,14,44,31)
orb.move(t)
#F = orb.state_v[orb.structure['calc_partials']].reshape(6,6)
#cov = F @ cov @ F.T
print(orb.state_v)
#print(cov)
#print(np.sqrt(cov[:3, :3].diagonal()))
