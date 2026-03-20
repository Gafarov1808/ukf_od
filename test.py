import numpy as np
from pyorbs.pyorbs import orbit, ephem_time, load
import pyorbs
from datetime import datetime, timedelta
from kiamdb.od import ContextOD
from models import LinearKalman, SquareRootUKF
from script_ukf import get_initial_params

_SEC2RAD = np.pi / 180. / 3600.

obj = 43235
t0 = datetime(2026, 2, 1)
t = t0 + timedelta(days = 28)
orb, _ = get_initial_params(obj, t0)

ctx = ContextOD(obj_id = obj, initial_orbit = orb, t_start = t0, t_stop = t, mle_limit = 1)
meas = ctx.meas_data.sort_values('time')
z = meas.iloc[63].copy()
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

  orb.setup_parameters()
  orb.move(t)
  F = orb.state_v[orb.structure['calc_partials']].reshape(6,6)
  print(orb.state_v)
  print(F)
  cov = F.T @ cov @ F
  print(orb.state_v[:6], orb.time.utc())
  #print(cov)
  print(f'std part_der = {np.sqrt(cov.diagonal())}')

def monte_carlo(n):
  print('MONTE CARLO METHOD:')
  orb = orbit()
  orb.state_v, orb.time = v0, t_start
  #orb = load('test_orbit')
  t0 = orb.time

  orbits = np.random.multivariate_normal(v0, P0, n)

  P = np.zeros((6, 6))
  orb.setup_parameters()
  orb.move(t_k)
  samples = np.ndarray((6, n))
  for j in range(len(orbits)):
    pyorbs.bal.uim_flushed()
    transform_orbit = orbit()
    transform_orbit.state_v, transform_orbit.time = orbits[j], t0
    transform_orbit.setup_parameters()
    transform_orbit.move(t_k)
    samples[:, j] = transform_orbit.state_v
  
  state = np.mean(samples, axis = 1)
  P = np.cov(samples)
  print(f'state = {state}')
  print(f'std mon-carl = {np.sqrt(P.diagonal())}')
  print('='*60)

def test_ukf(t_start, v0, P0, meas, z, t_k):
  print('UNSCENTED KALMAN:')
  filter = SquareRootUKF(t_start = t_start, v = v0, P = P0, meas = meas)
  filter.step(z, t_k)
  #print(filter.state_v)
  print('corr cov:')
  print(filter.cov_matrix[:3,:3])
  print('='*60)

def test_lin():
  print('LINEAR KALMAN:')
  #filter = LinearKalman(t_start = t_start, v = v0, P = P0, meas = meas)
  #filter.step(z, t_k)
  v0 = np.array([-3.94405353e+01,  1.57866146e+01,  1.23987720e+00, -1.13654747e+00,-2.84343778e+00,  2.70338495e-02])
  t_start = ephem_time(datetime(2026,2,17,7,14,20))
  t_k = ephem_time(datetime(2026,3,13,12,7,35,980000))
  orb = orbit()
  orb.state_v, orb.time = v0, t_start
  orb.setup_parameters()
  orb.move(t_k)
  print(orb.state_v)
  P = np.array([
  [ 3.47287919e+01,  9.52713751e+00, -4.04102133e-02, -3.45535899e+00, 1.31066327e+01,  1.87407983e-02],
 [ 9.52713751e+00,  8.37713684e+00, -5.04510839e-03, -3.15620268e+00, 3.62570813e+00,  8.38003345e-03],
 [-4.04102133e-02, -5.04510839e-03,  6.04143579e-05,  1.70604285e-03, -1.52191619e-02, -1.87290631e-05],
 [-3.45535899e+00, -3.15620268e+00 , 1.70604285e-03 , 1.18989733e+00, -1.31561103e+00, -3.10559060e-03],
 [ 1.31066327e+01,  3.62570813e+00 ,-1.52191619e-02, -1.31561103e+00, 4.94659643e+00,  7.08972423e-03],
 [ 1.87407983e-02,  8.38003345e-03 ,-1.87290631e-05, -3.10559060e-03, 7.08972423e-03,  1.19476928e-05]
])
  
  #filter.correction(z, center, P, orb)
  print('='*60)

#monte_carlo(1000)
#partial_der()
test_lin()
#test_ukf(t_start, v0, P0, meas, z, t_k)
