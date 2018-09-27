#! /usr/bin/env python3.4
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat

# define
theorie_lifetime = 2.196981

def exp_fit(t, N0, lamb, U):
	return(N0*np.exp(-lamb*t) + U)

data = np.genfromtxt('../data/Messung.txt', unpack=True)

# fit
params, cov = curve_fit(exp_fit, np.arange(len(data)), data, p0=[60, 0.5, 2])
errors = np.sqrt(np.diag(cov))

# print out results
print("Fit Function")
print(f'N0 = {params[0]:.2f} \pm {errors[0]:.2f}')
print(f'lambda = {params[1]:.4f} \pm {errors[1]:.4f}')
print(f'U = {params[2]:.2f} \pm {errors[2]:.2f}')

err_cal = ufloat(0.0454, 0.00003)
lifetime = err_cal / abs(params[1])
err_lifetime = ufloat(1/abs(params[1]), errors[1])
print(f"Myon Lifetime: {(err_cal * err_lifetime):.4f} micro sec")
print(f"relative Error: {(abs((theorie_lifetime-lifetime)/lifetime)):.4f}")

plt.plot(np.arange(len(data)), data, 'bx', label="Messwerte")
plt.plot(np.arange(len(data)), exp_fit(np.arange(len(data)), *params), 'r-', label="exponentieller Fit")

plt.legend()
plt.yscale('log')
plt.xlabel(r"Kanal")
plt.ylabel(r"Anzahl Ereignisse")
plt.savefig('../img/myons.pdf')
