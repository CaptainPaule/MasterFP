from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

theta, M_1, M_2, M_3 = np.genfromtxt('../data/n_luft.txt', unpack=True)
M = np.mean([M_1[:16], M_2[:16], M_3[:16]], axis=0)
M_error = np.std([M_1[:16], M_2[:16], M_3[:16]], axis=0)
M = np.append(M, np.mean([M_1[16:], M_2[16:]], axis=0))
M_error = np.append(M_error, np.std([M_1[16:], M_2[16:]], axis=0))
M_error[M_error == 0] = 1.
print(M_error)

def n_glas(x, a, b):
    return a * x + b

params, covariance_matrix = curve_fit(n_glas, theta, M, p0=[0.05, -1.3], sigma=M_error)
errors = np.sqrt(np.diag(covariance_matrix))

print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])

plt.errorbar(theta, M, yerr=M_error, fmt='+', label='Messwerte')
plt.plot(theta, n_glas(theta, *params), label='Fit')
plt.xlabel(r'Druck/mbar')
plt.ylabel(r'#Intensitätsmaxima')
plt.legend(loc='upper left')
plt.savefig('../img/n_luft.png')
