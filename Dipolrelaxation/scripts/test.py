import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import convert_temperature, Boltzmann, e
import numpy as np

def exponential_fit(x, a, b):
    return a * np.exp(b * x)

def linear_fit(x, a, b):
    return (a * x + b)

T, I = np.genfromtxt('../data/heizrate_15C.txt', unpack=True)
T = convert_temperature(T, 'C', 'K')
T1 = T[:15]
T2 = T[57:78]
T_fit = np.append(T1, T2)
I1 = I[:15]
I2 = I[57:78]
I_fit = np.append(I1, I2)

# fit
params_exp, cov_exp = curve_fit(exponential_fit, T_fit, I_fit, p0=[2, 0.1])
errors_exp = np.sqrt(np.diag(cov_exp))

print('Background Fit:')
print(f'a = {params_exp[0]} \pm {errors_exp[0]}')
print(f'b = {params_exp[1]} \pm {errors_exp[1]}')

# Strom ohne Untergrund
I_clean = I - exponential_fit(T, *params_exp)

plt.figure()
# plt.plot(T, exponential_fit(T, *params_exp), 'r-', label='Untergrund')
# plt.plot(T, I, 'b+', label='Messwerte')
# plt.plot(T_fit, I_fit, 'g+', label='Untergrundfit-Daten')
# plt.xlabel('T/°C')
# plt.legend()
# plt.savefig('../img/heizrate_15C.pdf')
# plt.figure()
# plt.plot(T, I_clean, 'b+')
# plt.savefig('../img/heizrate_15C_no-bg.pdf')

# fit
params_lin, cov_lin = curve_fit(linear_fit, 1/T[:20], np.log(I_clean[:20]))
errors_lin = np.sqrt(np.diag(cov_lin))

print('Einfacher linearer Fit für W:')
print(f'a = {params_lin[0]} \pm {errors_lin[0]}')
print(f'b = {params_lin[1]} \pm {errors_lin[1]}')
print(f'Aktivierungsenergie W = {- params_lin[0] * Boltzmann / e} eV')

plt.plot(1/T, np.log(I_clean), 'r*')
plt.plot(1/T[:20], np.log(I_clean[:20]), 'g*')
plt.plot(1/T[:20], linear_fit(1/T[:20], *params_lin), 'k-')
plt.savefig('../img/W-aus-Methode1.pdf')
