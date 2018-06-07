#! /usr/bin/env python3.4
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat

# define
theorie_lifetime = 2.196981

def build_bins(start, end, num):
	return np.linspace(start, end, num)

#def fit(x, A, B, C):
#	return A * np.exp(-B * x) + C

def lin_fit(x, A, B):
	return A * x + B

data = np.genfromtxt('../data/Messung.txt', unpack=True)
U = 0.7

# substract underground from data
data = data - 0.7

bins = 512
start = 0
end = 512
num = 512

# plot data
ax = build_bins(start, end, num)
print("Lenght Data: %s" % len(data))
print("Length axis: %s" % len(ax))
plt.plot(ax[3:200], np.log(data[3:200]), 'bx', label='Daten')
plt.plot(ax[200:400], np.log(data[200:400]), 'kx')


# print histogram
#hist, edges = np.histogram(data, bins=build_bins(start, end, bins))

# evaluate bin centers
#bin_centers = 0.5 * (edges[1:] + edges[:-1])

# calc width
#bin_width = (bin_centers[-1] - bin_centers[0]) / bins

#print("bin width: %s" % bin_width)

# plot bin centers
#plt.bar(edges[:-1], hist, width=bin_width, align='edge')
#plt.plot(bin_centers, hist, 'rx')

# fit
params, cov = curve_fit(lin_fit, ax[3:200], np.log(data[3:200]), p0=[1, 1])
errors = np.sqrt(np.diag(cov))

# print out results
print("Fit Function")
print('A =', params[0], '+/-', errors[0])
print('B =', params[1], '+/-', errors[1])

lifetime = 0.0454 * 1/abs(params[0])
err_cal = ufloat(0.0454, 0.00003)
err_lifetime = ufloat(1/abs(params[0]), errors[0])
print("Myon Lifetime: %s mu s" % (err_cal * err_lifetime))
print("relative Error: %s " % abs((theorie_lifetime-lifetime)/lifetime))


# scale x axis
ax_x = np.linspace(0, 400, 100)
#scale = 0.0454
#ax_x = ax_x * scale

plt.plot(ax_x, lin_fit(ax_x, *params), 'r-', label="linearer Fit")

plt.xlim(1, 400)
plt.legend()
plt.xlabel(r"Kanal")
plt.ylabel(r"log(Counts)")
#plt.ylim(0, 1600)
plt.savefig('../img/myons.pdf')