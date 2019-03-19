from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

a, s = np.genfromtxt("../data/LP01.csv", unpack=True)

def lp00(r, xr, w, I):
    return I * (r - xr)**2 * np.exp((-4 * (r-xr)**2)/(w)**2)


params, covariance_matrix = curve_fit(lp00, a, s, p0=[0, 10, 10])
errors = np.sqrt(np.diag(covariance_matrix))

print('xr =', params[0], '±', errors[0])
print('w =', params[1], '±', errors[1])
print('I =', params[2], '±', errors[2])

xx = np.linspace(-17, 30, 200)
plt.plot(a, s, '+', label='Messwerte')
plt.plot(xx, lp00(xx, *params), label='Fit')


plt.xlim(-17, 17)
plt.xlabel(r'$x \ mm$')
plt.ylabel(r'$Photostrom \ uA$')
plt.legend()
plt.savefig('../img/lp01.png')