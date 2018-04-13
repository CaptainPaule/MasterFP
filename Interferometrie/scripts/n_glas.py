from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# Eigenschaften Glas
T = 0.001
lamba = 633*10**(-9)
alpha = 160*(20*np.pi)/360

# Daten einlesen
theta, M_1, M_2, M_3 = np.genfromtxt('../data/n_glas.txt', unpack=True)
M = np.mean([M_1, M_2, M_3], axis=0)
M_error = np.std([M_1, M_2, M_3], axis=0)

nL = 0.999547
# Theoriekurve M(teta)
def n_glas(x, a):
    ta = (10 + x)/180*np.pi
    ta2 = (10 - x)/180*np.pi
    ti = np.arcsin(nL/a*np.sin(ta))
    ti2 = np.arcsin(nL/a*np.sin(ta2))
    return T/lamba * ( (a - np.cos(ta - ti))/np.cos(ti)*nL - (a - np.cos(ta2 - ti2))/np.cos(ti2)*nL )

params, covariance_matrix = curve_fit(n_glas, theta, M, sigma=M_error)
errors = np.sqrt(np.diag(covariance_matrix))

print('a =', params[0], '±', errors[0])

plt.errorbar(theta, M, yerr=M_error, fmt='+', label='gemittelte Messwerte')
plt.plot(theta, n_glas(theta, *params), label='Fit an gemittelte Werte')
plt.plot(theta, M_1, 'x', label='Messwerte 1.Messung')
plt.plot(theta, M_2, 'x', label='Messwerte 2.Messung')
plt.plot(theta, M_3, 'x', label='Messwerte 3.Messung')
plt.xlabel(r'$\theta/°$')
plt.ylabel(r'$M(\theta)$')
plt.legend()
plt.tight_layout()
plt.savefig('../img/n_glas.png')
