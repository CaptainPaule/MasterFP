import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import convert_temperature, Boltzmann, e
import numpy as np

def polynomal_fit(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

def linear_fit(x, a, b):
    return (a * x + b)

T, I = np.genfromtxt('../data/heizrate_15C.txt', unpack=True)
T = convert_temperature(T, 'C', 'K')
T1 = T[:15]
T2 = T[57:80]
T_fit = np.append(T1, T2)
I1 = I[:15]
I2 = I[57:80]
I_fit = np.append(I1, I2)

# fit
params_exp, cov_exp = curve_fit(polynomal_fit, T_fit, I_fit)
errors_exp = np.sqrt(np.diag(cov_exp))

print('Background Fit:')
print(f'a = {params_exp[0]} \pm {errors_exp[0]}')
print(f'b = {params_exp[1]} \pm {errors_exp[1]}')

# Strom ohne Untergrund
T_clean = T[:80]
I_clean = I[:80] - polynomal_fit(T_clean, *params_exp) + params_exp[4]
I_clean = I_clean - np.min(I_clean) + np.min(I_clean)*1e-55
print(np.argmin(np.log(I_clean)))

# noch den Offset (vom Messgerät) bestimmen und abziehen!

# plt.figure()
# plt.plot(T, polynomal_fit(T, *params_exp), 'r-', label='Untergrund')
# plt.plot(T, I, 'b+', label='Messwerte')
# plt.plot(T_fit, I_fit, 'g+', label='Untergrundfit-Daten')
# plt.xlabel('T/K')
# plt.ylabel('I/A')
# plt.legend()
# plt.savefig('../img/heizrate_15C.pdf')
# plt.figure()
# plt.plot(T_clean, I_clean, 'b+', label='Bereinigte Daten')
# plt.xlabel('T/K')
# plt.ylabel('I/A')
# plt.legend()
# plt.savefig('../img/heizrate_15C_no-bg.pdf')

# fit
params_lin, cov_lin = curve_fit(linear_fit, 1/T[:25], np.log(I_clean[:25]))
errors_lin = np.sqrt(np.diag(cov_lin))

print('Einfacher linearer Fit für W:')
print(f'a = {params_lin[0]} \pm {errors_lin[0]}')
print(f'b = {params_lin[1]} \pm {errors_lin[1]}')
print(f'Aktivierungsenergie W = {- params_lin[0] * Boltzmann / e} eV')

# plt.figure()
# plt.plot(1/T_clean, np.log(I_clean), 'r*', label='Messwerte')
# plt.plot(1/T[:25], np.log(I_clean[:25]), 'g*', label='Fitdaten')
# plt.plot(1/T[:25], linear_fit(1/T[:25], *params_lin), 'k-', label='Ausgleichsgerade')
# plt.xlabel('1/T / 1/K')
# plt.ylabel('ln(I/1A)')
# plt.legend()
# plt.savefig('../img/W-aus-Methode1.pdf')

integrated = np.array([])
for i in range(len(T_clean)):
    integrated = np.append(integrated, np.log(np.trapz(I_clean[i:], T_clean[i:]) / I_clean[i]))

# fit
params_linin, cov_linin = curve_fit(linear_fit, 1/T_clean[:-25], integrated[:-25])
errors_linin = np.sqrt(np.diag(cov_linin))

print('Linearer Fit an Integral für W:')
print(f'a = {params_linin[0]} \pm {errors_linin[0]}')
print(f'b = {params_linin[1]} \pm {errors_linin[1]}')
print(f'Aktivierungsenergie W = {params_linin[0] * Boltzmann / e} eV')

plt.figure()
plt.plot(1/T_clean[:-5], integrated[:-5], 'bx', label='Daten')
plt.plot(1/T_clean[:-25], integrated[:-25], 'gx', label='Fitdaten')
plt.plot(1/T_clean[:-25], linear_fit(1/T_clean[:-25], *params_linin), 'k-', label='Ausgleichsgerade')
plt.xlabel('1/T / 1/K')
plt.ylabel(r'$\ln(\int I/I)$')
plt.legend()
plt.savefig('../img/W-aus-Methode2.pdf')
