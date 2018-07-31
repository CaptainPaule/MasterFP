#! /usr/bin/env python3.4
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# fit function
def fit(x, A, B):
    return A * x + B

def make_linear_fit(input_file, data_sym, fit_sym, omit):
    freq, U_A, phi, U_1 = np.genfromtxt('{}'.format(input_file), unpack=True)

    # linear fit
    params, cov = curve_fit(fit, np.log10(freq[omit:]), np.log10(U_A[omit:]/U_1[omit:]))
    errors = np.sqrt(np.diag(cov))

    # compute critical frequency
    nu_G = 10**((np.log10((U_A[0]/U_1[0])/np.sqrt(2)) - params[1]) / params[0])

    # print out results
    print('\nInputfile: {}, nu_G*V: {}'.format(input_file, nu_G*U_A[0]/U_1[0]))
    print('A =', params[0], '+/-', errors[0])
    print('B =', params[1], '+/-', errors[1])
    print('Grenzfrequenz:', nu_G, 'Hz; Verstärkung: ', np.mean(U_A[freq<10]/U_1[freq<10]), '+/-', np.std(U_A[freq<10]/U_1[freq<10]))

    # plot
    plt.plot(freq[omit:], 10**(fit(np.log10(freq[omit:]), *params)), '{}'.format(fit_sym))
    plt.plot(freq, U_A/U_1, '{}'.format(data_sym), label='{}'.format(input_file.replace("_", " ").replace("-", ".")[10:-4]))
    plt.axhline(y=np.mean(U_A[freq<10]/U_1[freq<10])/np.sqrt(2), color='y')
    plt.axvline(x=nu_G, color='y')
    # save fig
    plt.legend(loc='lower left')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Frequenz $\nu$ / Hz')
    plt.ylabel(r'Verstärkung $V^\prime$')
    plt.savefig('../img/{}.png'.format(input_file[8:-4]))
    print('')
    plt.clf()

if __name__ == '__main__':
    make_linear_fit('../data/a_R1_10-02k_RN_100k.txt', 'gx', 'g-', 7)
    make_linear_fit('../data/a_R1_10-02k_RN_33-2k.txt', 'b.', 'b-', 11)
    make_linear_fit('../data/a_R1_0-2k_RN_1k.txt', 'r1', 'r-', 9)
    make_linear_fit('../data/a_R1_33-3k_RN_100k.txt', 'k|', 'k-', 11)
