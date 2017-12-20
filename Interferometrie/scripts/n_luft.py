from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
import uncertainties.unumpy as unp

# Länge von Gas in Meter
L = 0.1
# Wellenlänge vom Licht
lamb = 633e-9

druck, M_1, M_2, M_3 = np.genfromtxt('../data/n_luft.txt', unpack=True)
M_1 = M_1 * lamb / L + 1
M_2 = M_2 * lamb / L + 1
M_3 = M_3 * lamb / L + 1
M = np.mean([M_1[:16], M_2[:16], M_3[:16]], axis=0)
M_error = np.std([M_1[:16], M_2[:16], M_3[:16]], axis=0)
M = np.append(M, np.mean([M_1[16:], M_2[16:]], axis=0))
M_error = np.append(M_error, np.std([M_1[16:], M_2[16:]], axis=0))
M_error[M_error == 0.] = 1e-8
print(M_1, M_error)

def n_glas(x, a):
    return np.sqrt(a * x + 1)

params, covariance_matrix = curve_fit(n_glas, druck, M, sigma=M_error)
errors = np.sqrt(np.diag(covariance_matrix))

print('a =', params, '±', errors)

F = ufloat(params, errors)
T0 = ufloat(15, 0)
T = ufloat(22.2, 0)
F_norm = F * T / T0
n_norm = unp.sqrt(1 - F_norm * 1013)
print('Brechungsindex bei Normalbedingungen: ', n_norm)

plt.errorbar(druck, M, yerr=M_error, fmt='+', label='gemittelte Messwerte')
plt.plot(druck, n_glas(druck, *params), label='Fit an gemittelte Werte')
plt.plot(druck, M_1, 'x', label='Messwerte 1.Messung')
plt.plot(druck, M_2, 'x', label='Messwerte 2.Messung')
plt.plot(druck, M_3, 'x', label='Messwerte 3.Messung')
plt.xlabel(r'Druck/mbar')
plt.ylabel(r'$n_{Luft}$')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('../img/n_luft.png')
