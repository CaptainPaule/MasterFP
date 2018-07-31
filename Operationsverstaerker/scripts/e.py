#! /usr/bin/env python3.4
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# fit function
def fit(x, A, B):
    return A * x + B

def make_linear_fit(input_file):
    freq, U_E, U_A, phi = np.genfromtxt('{}'.format(input_file), unpack=True)

    # linear fit
    params, cov = curve_fit(fit, np.log10(freq), np.log10(U_A))
    errors = np.sqrt(np.diag(cov))

    # print out results
    print('\nInputfile: {}'.format(input_file))
    print('A =', params[0], '+/-', errors[0])
    print('B =', params[1], '+/-', errors[1])

    # plot
    plt.plot(freq, 10**(fit(np.log10(freq), *params)), 'g-')
    plt.plot(freq, U_A, 'rx', label='{}F'.format(input_file.replace("_", " ").replace("-", ".")[10:-4]))

if __name__ == '__main__':
    make_linear_fit('../data/e_R1_0-2k_C1_23-1n.txt')
    # save fig
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Frequenz $\nu$ / Hz')
    plt.ylabel(r'Ausgangsspannung $U_A$')
    plt.savefig('../img/e.png')
    print('')
