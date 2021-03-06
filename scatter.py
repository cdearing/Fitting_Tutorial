#!/usr/bin/env python
"""
This file implements a python function that is intended to be used as part of
a tutorial on mathematical optimization.

Originally written for the UNC-CH CAP-REU program, 
summer of 2013.

Author: Zane Beckwith

In the toy example given in the tutorial, we imagine that this function 
represents black box simulation software for nuclear cross-section
calculations. The students are instructed to find the maximum of this function
using a Nelder-Mead optimizer method (written for the tutorial, and 
included in this directory) to find the 'parameter values' 
('energy' and 'helicity') that give the largest cross-section.

In reality, this is simply the negative of Himmelblau's function
(it's negative in order to turn the minima into maxima).
"""

def simulate(params):
  """
  Argument:
    params - 2-d arraylike [energy, helicity]:
      energy = COM energy of beam of particles allegedly being simulated
      helicity = helicity of the alleged beam
  Return value:
           - float: 
      cross-section of scattering
  """

  himmel = (params[0]**2 + params[1] - 11)**2 + (params[0] + params[1]**2 - 7)**2
  return himmel

if __name__ == '__main__':

  import numpy as np
  import matplotlib.pyplot as plt
  # Test one known minimum
  test_out_1 = simulate(np.array([-2.805118,3.131312]))
  assert test_out_1 > -1e-10

  # Test the known maximum
  test_out_2 = simulate(np.array([-0.270845, -0.923039]))
  assert test_out_2 < -180
 

  # Test a very-precisely-known minimum
  # (even accounting for floating-point
  # comparison, this should test fine)
  test_out_3 = simulate(np.array([3.0, 2.0]))
  assert test_out_3 == 0.0

 
