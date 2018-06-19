#! /usr/bin/env python3.4
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

freq, U_A, phi = np.genfromtxt('../data/a_R1_10-02k_RN_100k_U1_233mV.txt', unpack=True)

# fit function
def fit(x, A, B):
    return A * x + B

# linear fit
params, cov = curve_fit(fit, np.log10(freq[7:]), np.log10(U_A[7:]))
errors = np.sqrt(np.diag(cov))

# print out fit results
print("Fit Function")
print('A =', params[0], '+/-', errors[0])
print('B =', params[1], '+/-', errors[1])

# plot
plt.plot(freq[7:], 10**(fit(np.log10(freq[7:]), *params)), 'g-', label="fit")
plt.plot(freq, U_A, 'rx', label='data')

# save fig
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Frequenz $\nu$ / Hz')
plt.ylabel(r'Spannung $U_A$ / mV')
plt.savefig('../img/a_R1_10-02k_RN_100k_U1_233mV..png')
