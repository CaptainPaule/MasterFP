import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

delay_l, counts_l = np.genfromtxt('../data/delay_l.txt', unpack=True)
delay_r, counts_r = np.genfromtxt('../data/delay_r.txt', unpack=True)
delay_l = - delay_l

delay = np.append(np.flip(delay_l, axis=0), delay_r)
counts = np.append(np.flip(counts_l, axis=0), counts_r)
print(delay)

def gauss(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

# gaussian fit
params, cov = curve_fit(gauss, delay, counts, p0=[250., 2., 10.])
errors = np.sqrt(np.diag(cov))

# print out fit results
print("Fit Function")
print('A =', params[0], '+/-', errors[0])
print('mu =', params[1], '+/-', errors[1])
print('sigma =', params[2], '+/-', errors[2])

# plot
plt.plot(delay, gauss(delay, *params), 'r-', label="fit")
plt.plot(delay, counts, 'g+')
plt.xlabel('Verzögerung in ns')
plt.ylabel('Zählrate')
plt.show()
