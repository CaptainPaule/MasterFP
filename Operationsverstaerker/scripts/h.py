import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from detect_peaks import detect_peaks

C1 = 20e-9
R = 10e3

print('f1 = ' + str(1/(2*np.pi*R*C1)))
print('tau = ' + str(20*R*C1))

t, U, _ = np.genfromtxt('../usb/scope_240.csv',unpack=True, delimiter=',')

t_p = t[np.logical_and(t > 0.0, t < 0.02)]
U_p = U[np.logical_and(t > 0.0, t < 0.02)]
t_p = t_p[ U_p > 0.0 ]
U_p = U_p[ U_p > 0.0 ]
U_p_log = np.log(U_p)
indexes = detect_peaks(U_p_log, mph=-4.5, mpd=0.002)

def fit(x,a,b):
	return x * a + b

# linear fit
params, cov = curve_fit(fit, t_p[indexes], np.log10(U_p[indexes]))
errors = np.sqrt(np.diag(cov))

print('A =', params[0], '+/-', errors[0])
print('B =', params[1], '+/-', errors[1])

plt.plot(t_p, U_p,'rx',label=r'Messdaten')
# plt.plot(t_p[indexes], U_p[indexes],'gx',label=r'Peaks')
plt.plot(t_p[indexes], 10**(fit(t_p[indexes], *params)), 'g-', label='Fit')

plt.grid()
plt.yscale('log')
plt.xlabel(r'$t/\mathrm{s}$')
plt.ylabel(r'$U/\mathrm{V}$')
plt.legend(loc="best")
plt.savefig('../img/h.png')
