#! /usr/bin/env python3.4
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def build_bins(start, end, num):
	return np.linspace(start, end, num)

def fit(x, A, B):
	return A * np.exp(-B * x)

data = np.genfromtxt('../data/Messung.txt', unpack=True)
bins = 30.0
start = 1.0
end = 512.0

data = data - 0.2

# print histogram
hist, edges = np.histogram(data, bins=build_bins(start, end, bins))

print(edges)

# evaluate bin centers
bin_centers = 0.5 * (edges[1:] + edges[:-1])

# calc width
bin_width = (bin_centers[-1] - bin_centers[0]) / bins

print("bin width: %s" % bin_width)
print(hist)

# plot bin centers
plt.bar(edges[:-1], hist, width=bin_width, align='edge')
plt.plot(bin_centers, hist, 'rx')

# fit
params, cov = curve_fit(fit, hist[:len(hist)/3], bin_centers[:len(bin_centers)/3], p0 = [200, 1])
errors = np.sqrt(np.diag(cov))

# print out fit results
print("Fit Function")
print('A =', params[0], '+/-', errors[0])
print('B =', params[1], '+/-', errors[1])
#print('C =', params[2], '+/-', errors[2])

plt.plot(np.linspace(1, 512, 1000), fit(np.linspace(1, 512, 1000), *params), 'r-')

plt.xlim(1, 512)
plt.show()