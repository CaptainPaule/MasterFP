import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fit(x, A, B):
	return A*x + B

# import data
omega_T, U, U_0 = np.genfromtxt('../data/aufgabe_e.txt', unpack=True)

# linear fit
params, cov = curve_fit(fit, omega_T, np.arccos(U/U_0))
errors = np.sqrt(np.diag(cov))

print("Fit Function")
print('A =', params[0], '+/-', errors[0])
print('B =', params[1], '+/-', errors[1])

plt.plot(omega_T, np.arccos(U/U_0), 'kx', label="data")
plt.plot(omega_T, fit(omega_T, *params), 'r-', label="fit")
plt.xlabel(r'$\omega_T$')
plt.ylabel(r'$\arccos(U/U_0)$')
plt.legend()
plt.show()
