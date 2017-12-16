from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import correlated_values

phi, v_max, v_min = np.genfromtxt('../data/kontrast.txt', unpack=True)
K = -(v_max - v_min) / (v_max + v_min)

def kontrast(x, a, b):
    return a * abs(np.sin(2 * np.deg2rad(x) + np.deg2rad(b)))

def mse(y, y_hat):
    return np.sum((y-y_hat) ** 2)

params, covariance_matrix = curve_fit(kontrast, phi, K, p0=[1.0, 0.0])
errors = np.sqrt(np.diag(covariance_matrix))
#errors = correlated_values(covariance_matrix)
a_vec = np.linspace(-45.0, 45.0, 100)
mse_vec = np.zeros(len(a_vec))
for i in range(len(a_vec)):
    mse_vec[i] = mse(K, kontrast(phi, 1.0, a_vec[i]))

plt.plot(a_vec, mse_vec)
plt.show()
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
xx=np.linspace(0,1)*180
plt.plot(phi, K, '+')
plt.plot(xx, kontrast(xx, 1.0, 0.0))
plt.show()
