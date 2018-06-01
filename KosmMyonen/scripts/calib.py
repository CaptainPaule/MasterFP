#! /usr/bin/env python3.4
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

compressed_data = []
data = np.genfromtxt('../data/Kalibrierung_TAC.txt', unpack=True)

x = 0
y = 0

for i, val in zip(range(len(data)), data):
    if val!=0:
        y = y + val
        x = 0
    if (val==0 and x==0):
        compressed_data.append(y)
        x = x + 1
        y = 0

channel = range(len(compressed_data))
plt.plot(channel[1:], compressed_data[1:], 'rx')
plt.savefig('../img/calib.png')
