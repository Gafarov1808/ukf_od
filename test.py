import numpy as np
from pyorbs.pyorbs import orbit, ephem_time, load
import pyorbs
from datetime import datetime, timedelta
from kiamdb.od import ContextOD
from models import LinearKalman, SquareRootUKF
from script_ukf import get_initial_params

_SEC2RAD = np.pi / 180. / 3600.
obj = 43087
t0 = datetime(2026, 1, 1)
t = t0 + timedelta(days = 28)
orb, _ = get_initial_params(obj, t0)

t_start = ephem_time(datetime(2026,1,1,19,17,55))
t_k = ephem_time(datetime(2026,1,2,18,0,29, 710 ))
v0 = np.array([29.31671358, 30.77124676, -2.11061921, -2.19041675,  2.10544439,  0.35919605])
P0 = np.array([
  [ 2.42765722e-01,  9.17646249e-01,  6.63171419e-03, -9.31071135e-02,6.48563753e-02,  1.14968000e-02],
  [ 9.17646249e-01,  3.91423675e+00,  4.38056283e-02 ,-6.82496768e-01,1.67138275e-01 , 7.71090516e-02],
  [ 6.63171419e-03,  4.38056283e-02,  1.09885337e-03, -1.54284538e-02,-1.25838605e-03,  1.63158146e-03],
  [-9.31071135e-02, -6.82496768e-01, -1.54284538e-02  ,1.68791343e+00,5.20939660e-01 ,-1.68559405e-01],
  [ 6.48563753e-02,  1.67138275e-01, -1.25838605e-03  ,5.20939660e-01,2.01288128e-01 ,-5.10656957e-02],
  [ 1.14968000e-02,  7.71090516e-02 , 1.63158146e-03, -1.68559405e-01,-5.10656957e-02,  1.68556850e-02]
  ])

ctx = ContextOD(obj_id = obj, initial_orbit = orb, t_start = t0, t_stop = t, mle_limit = 1)
meas = ctx.meas_data.sort_values('time')
z = meas.iloc[173].copy()
z['sta_id'] = int(z['sta_id'])

def partial_der():
  orb = orbit()
  orb = load('test_orbit')
  #orb.state_v = np.array([20.20687328, 0.89828793, 19.94422809, -1.66755229,  2.98893043, 1.57641073])
  #orb.time = ephem_time(datetime(2025,12,1,17,49,36,950000))
  orb.change_param({'calc_partials': True})
  orb.set_initial_point(orb.time)

  cov = np.array([
      [ 4.32185767e-04, -3.44894839e-06,  1.01525917e-05, -1.15224203e-08, -7.13990932e-09,  1.05472672e-07],
      [-3.44894839e-06,  9.57788630e-06, -9.37244806e-08, -1.15389730e-10, -1.35245118e-09, -1.08164274e-09],
      [ 1.01525917e-05, -9.37244806e-08,  9.95957132e-06,  2.10867490e-09, -3.67900988e-10,  4.78079196e-09],
      [-1.15224203e-08, -1.15389730e-10,  2.10867490e-09,  1.07991025e-10,  5.02619813e-12, -7.73282725e-11],
      [-7.13990932e-09, -1.35245118e-09, -3.67900988e-10,  5.02619813e-12,  1.03650065e-10, -5.26347484e-11],
      [ 1.05472672e-07, -1.08164274e-09,  4.78079196e-09, -7.73282725e-11, -5.26347484e-11,  9.09310075e-10]
  ])

  t = datetime(2025, 12, 3, 12, 28, 6, 700000)

  orb.move(t)
  F = orb.state_v[orb.structure['calc_partials']].reshape(6,6)
  print(orb.state_v)
  print(F)
  cov = F.T @ cov @ F
  print(orb.state_v[:6], orb.time.utc())
  #print(cov)
  print(f'std part_der = {np.sqrt(cov.diagonal())}')

def monte_carlo(n):

  orb = orbit()
  orb = load('test_orbit')
  t0 = orb.time
  t = ephem_time(datetime(2025, 12, 3, 12, 28, 6, 700000))
  cov = np.array([
      [ 4.32185767e-04, -3.44894839e-06,  1.01525917e-05, -1.15224203e-08, -7.13990932e-09,  1.05472672e-07],
      [-3.44894839e-06,  9.57788630e-06, -9.37244806e-08, -1.15389730e-10, -1.35245118e-09, -1.08164274e-09],
      [ 1.01525917e-05, -9.37244806e-08,  9.95957132e-06,  2.10867490e-09, -3.67900988e-10,  4.78079196e-09],
      [-1.15224203e-08, -1.15389730e-10,  2.10867490e-09,  1.07991025e-10,  5.02619813e-12, -7.73282725e-11],
      [-7.13990932e-09, -1.35245118e-09, -3.67900988e-10,  5.02619813e-12,  1.03650065e-10, -5.26347484e-11],
      [ 1.05472672e-07, -1.08164274e-09,  4.78079196e-09, -7.73282725e-11, -5.26347484e-11,  9.09310075e-10]
  ])

  orbits = np.random.multivariate_normal(orb.state_v[:6], cov, n)

  P = np.zeros((6, 6))
  orb.move(t)
  samples = np.ndarray((6, n))
  for j in range(len(orbits)):
    pyorbs.bal.uim_flushed()
    transform_orbit = orbit()
    transform_orbit.state_v, transform_orbit.time = orbits[j], t0
    transform_orbit.setup_parameters()
    transform_orbit.move(t)
    samples[:, j] = transform_orbit.state_v
  
  P = np.cov(samples)
  print(f'std mon-carl = {np.sqrt(P.diagonal())}')

def test_ukf(t_start, v0, P0, meas, z, t_k):
  filter = SquareRootUKF(t_start = t_start, v = v0, P = P0, meas = meas)
  filter.step(z, t_k)

def test_lin(t_start, v0, P0, meas, z, t_k):
    
  filter = LinearKalman(t_start = t_start, v = v0, P = P0, meas = meas)
  filter.step(z, t_k)

#monte_carlo(1000)
#partial_der()
test_ukf(t_start, v0, P0, meas, z, t_k)
test_lin(t_start, v0, P0, meas, z, t_k)