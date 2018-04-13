#!/usr/bin/python
# -*- coding: utf-8 -*-

# imports
import os
import math
import copy
import numpy as np

# set matplotlib backend
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import constants
from uncertainties import ufloat

def coreSpin(gf, gj):
	return (gj)/(2*gf) - (1)/(2)
	
# define electron g factor
gj = ufloat(2.00231930436182, 0.00000000000052)

# define isotop g factor
gF1 = ufloat(0.488, 0.005)
gF2 = ufloat(0.331, 0.002)

# print results for core spin
print("CoreSpin for gF1: ")
print(coreSpin(gF1, gj))

print("CoreSpin for gF2: ")
print(coreSpin(gF2, gj))

# quadratic zeeman effect