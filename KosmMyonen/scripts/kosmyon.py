#! /usr/bin/env python3.4
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def build_bins(start, end, num):
	return np.linspace(start, end, num)


data = np.genfromtxt('../data/Messung.txt', unpack=True)
bins = 256.0
start = 1.0
end = 512.0

# print histogram
hist, edges = np.histogram(data, bins=build_bins(start, end, bins))

print(hist)

# evaluate bin centers
bin_centers = 0.5 * (edges[1:] + edges[:-1])

# calc width
bin_width = 0.5 * (bin_centers[-1] - bin_centers[0]) / bins

print("bin width: %s" % bin_width)
print("centers: ")

# plot bin centers
plt.bar(edges[:-1], hist, width=end/bins)
plt.plot(bin_centers, hist, 'rx')


plt.xlim(1, 512)
plt.show()