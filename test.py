import numpy as np
from pyorbs.pyorbs import orbit, ephem_time, load
import pyorbs
from datetime import datetime

_SEC2RAD = np.pi / 180. / 3600.

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

  #print(orb.state_v[:6], orb.time.utc())
  t = datetime(2025, 12, 3, 12, 28, 6, 700000)
  #t = datetime(2025, 12, 1, 18, 19, 6, 700000)

  orb.move(t)
  F = orb.state_v[orb.structure['calc_partials']].reshape(6,6)
  print(orb.state_v)
  print(F)
  cov = F.T @ cov @ F
  #print(orb.state_v[:6], orb.time.utc())
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

monte_carlo(1000)
partial_der()