from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

a, s = np.genfromtxt("../data/LP00.csv", unpack=True)

def lp00(r, xr, w, I):
    return I * np.exp((-2 * (r-xr)**2)/(w)**2)


params, covariance_matrix = curve_fit(lp00, a, s)
errors = np.sqrt(np.diag(covariance_matrix))

print('xr =', params[0], '±', errors[0])
print('w =', params[1], '±', errors[1])
print('I =', params[2], '±', errors[2])

xx = np.linspace(-20, 12, 100)
plt.plot(a, s, '+', label='Messwerte')
plt.plot(xx, lp00(xx, *params), label='Fit')

plt.xlim(-20, 12)
plt.xlabel(r'x \ mm')
plt.ylabel(r'Photostrom \ uA')
plt.legend()
plt.savefig('../img/lp00.png')