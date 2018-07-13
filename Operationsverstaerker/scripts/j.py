#! /usr/bin/env python3.4
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plot_phase(input_file, data_sym):
    freq, U_A, phi, U_1 = np.genfromtxt('{}'.format(input_file), unpack=True)
    plt.plot(freq, phi, '{}'.format(data_sym), label='{}'.format(input_file.replace("_", " ")[10:-4]))

if __name__ == '__main__':
    plot_phase('../data/a_R1_10-02k_RN_100k.txt', 'gx')
    plot_phase('../data/a_R1_10-02k_RN_33-2k.txt', 'b.')
    plot_phase('../data/a_R1_0-2k_RN_1k.txt', 'r1')
    plot_phase('../data/a_R1_33-3k_RN_100k.txt', 'k|')
    # save fig
    plt.legend()
    plt.xscale('log')
    plt.xlabel(r'Frequenz $\nu$ / Hz')
    plt.ylabel(r'Phase $\varphi$')
    plt.savefig('../img/j.png')
