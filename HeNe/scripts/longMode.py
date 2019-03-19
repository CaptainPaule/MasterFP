from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# defines
c = 3 # speed of light in m/s
L = [1.5, 1.3, 1.1] # length of resonator

v150 = np.genfromtxt("../data/LongModeFreq150.csv", unpack=True)
v130 = np.genfromtxt("../data/LongModeFreq130.csv", unpack=True)
v110 = np.genfromtxt("../data/LongModeFreq110.csv", unpack=True) 

lenV150 = len(v150) # num of samples in v150 dataset
lenV130 = len(v130) # num of samples in v130 dataset
lenV110 = len(v110) # num of samples in v110 dataset

vecNum150 = [1, 2, 3 ,4, 5, 6, 7, 8, 9]
vecNum130 = [1, 2, 3, 4, 5, 6, 7]
vecNum110 = [1, 2 , 3, 4, 5]

def freq(N, C, L):
    C = np.asarray(C)
    N = np.asarray(N)
    L = np.asarray(L)
    return (C * N)/(2.0 * L)


plt.plot(v150, freq(vecNum150, c, 1.5), "rx-", label='L=150')
plt.plot(v130, freq(vecNum130, c, 1.5), "bx-", label='L=130')
plt.plot(v110, freq(vecNum110, c, 1.5), "kx-", label='L=110')

plt.xlim(100, 900)
plt.xlabel(r'Freq \ MHz')
plt.ylabel(r'Freq \ GHz')
plt.legend()

plt.legend()
plt.savefig('../img/longMode.png')