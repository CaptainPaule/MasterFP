from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

theta, M_1, M_2, M_3 = np.genfromtxt('../data/n_glas.txt', unpack=True)
M = np.mean([M_1, M_2, M_3], axis=0)
print(M)

def n_glas(x, a, b):
    return a * x + b

params, covariance_matrix = curve_fit(n_glas, theta, M)
errors = np.sqrt(np.diag(covariance_matrix))

print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])

plt.plot(theta, M, '+')
plt.plot(theta, n_glas(theta, *params))
plt.show()
