#! /usr/bin/env python3.4
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# fit function
def fit(x, A, B):
    return A * x + B

def make_linear_fit(input_file, omit):
    freq, U_E, U_A, phi = np.genfromtxt('{}'.format(input_file), unpack=True)

    # linear fit
    params, cov = curve_fit(fit, np.log10(freq[omit:]), np.log10(U_A[omit:]/U_E[omit:]))
    errors = np.sqrt(np.diag(cov))

    # print out results
    print('\nInputfile: {}'.format(input_file))
    print('A =', params[0], '+/-', errors[0])
    print('B =', params[1], '+/-', errors[1])

    # plot
    plt.plot(freq[omit:], 10**(fit(np.log10(freq[omit:]), *params)), 'g-')
    plt.plot(freq, U_A/U_E , 'rx', label='{}F'.format(input_file.replace("_", " ").replace("-", ".")[10:-4]))

if __name__ == '__main__':
    make_linear_fit('../data/d_R1_10-02k_C1_23-1n.txt', 3)
    # save fig
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Frequenz $\nu$ / Hz')
    plt.ylabel(r'Verstärkung $V^\prime$')
    plt.savefig('../img/d.png')
    print('')
