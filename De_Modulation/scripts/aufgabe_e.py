import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# rescale
delta_T = 250e-9

def fit(x, A, B):
	return A*x + B

# import data
omega_T, U, U_0 = np.genfromtxt('../data/aufgabe_e.txt', unpack=True)

omega_T = omega_T * 1e6
# linear fit
params, cov = curve_fit(fit, omega_T * delta_T, np.arccos(U/U_0))
errors = np.sqrt(np.diag(cov))

print("Fit Function")
print('A =', params[0], '+/-', errors[0])
print('B =', params[1], '+/-', errors[1])

plt.plot(omega_T*delta_T, np.arccos(U/U_0), 'kx', label="data")
plt.plot(omega_T*delta_T, fit(omega_T*delta_T, *params), 'r-', label="fit")
plt.xlabel(r'$\varphi$')
plt.ylabel(r'$\arccos(U/U_0)$')
plt.legend()
plt.show()
