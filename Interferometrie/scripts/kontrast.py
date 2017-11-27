from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

phi, v_max, v_min = np.genfromtxt('../data/kontrast.txt', unpack=True)
K = -(v_max - v_min) / (v_max + v_min)

def kontrast(x, a, b):
    return a * abs(np.sin (2*x + b))

Phi = np.arange(0,195,15)*2*np.pi/360
params, covariance_matrix = curve_fit(kontrast, Phi, K, p0=[1., 0.008])
errors = np.sqrt(np.diag(covariance_matrix))

print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])

plt.plot(phi, K, '+')
plt.plot(phi, kontrast(phi, *params))
#plt.show()
