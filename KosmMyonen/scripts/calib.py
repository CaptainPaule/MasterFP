#! /usr/bin/env python3.4
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

channel_number = []
data = np.genfromtxt('../data/Kalibrierung_TAC.txt', unpack=True)

# get the relevant data from calibration measurement
y = []
for id in range(len(data)):
    if data[id]!=0:
        y.append(id)
    if (data[id]==0 and y):
        channel_number.append(round(np.mean(y)))
        y = []

# define delay times in ns
delay = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.4, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 9.9]

def fit(x, A):
    return A*x

# linear fit
params, cov = curve_fit(fit, channel_number, delay)
errors = np.sqrt(np.diag(cov))

print("Fit Function")
print('A =', params[0], '+/-', errors[0])

plt.plot(channel_number, fit(channel_number, params), 'g-', label="fit")
plt.plot(channel_number, delay, 'rx', label='data')
plt.legend()
plt.xlabel('Kanalnummer')
plt.ylabel('Laufzeit in ns')
plt.savefig('../img/calib.png')
