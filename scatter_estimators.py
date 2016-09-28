# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 17:06:38 2016

@author: cdearing
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr


xvals = np.linspace(1,10,100)
yvals = np.linspace(20,40,100)
errs = 1
narr = 100

yvals = yvals + npr.normal(0,errs,narr)

plt.plot(xvals,yvals)