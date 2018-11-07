import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import convert_temperature, Boltzmann, elementary_charge
import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat

def polynomal_fit(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

def linear_fit(x, a, b):
    return (a * x + b)

# T_max: temperature at which current has maximum; b: heating rate in K/s; W: activation energy
def relaxationszeit(T_max, b, W):
    tau_0 = Boltzmann / elementary_charge * T_max**2 / (W * b) * unp.exp(- W / (Boltzmann / elementary_charge * T_max))
    return(tau_0)

@np.vectorize
def tau_vs_T(tau, W, T):
    tau = unp.nominal_values(tau)
    W = unp.nominal_values(W)
    tau_T = tau * unp.exp( W / (Boltzmann / elementary_charge * T))
    return(tau_T)

print('\nHeizrate: 1.5 K/s')
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

a_exp = ufloat(params_exp[0], errors_exp[0])
b_exp = ufloat(params_exp[1], errors_exp[1])
c_exp = ufloat(params_exp[2], errors_exp[2])
d_exp = ufloat(params_exp[3], errors_exp[3])
e_exp = ufloat(params_exp[4], errors_exp[4])

print('Background Fit:')
print(f'a = {a_exp}')
print(f'b = {b_exp}')
print(f'c = {c_exp}')
print(f'd = {d_exp}')
print(f'e = {e_exp}')

# Strom ohne Untergrund
T_clean = T[:80]
I_clean = I[:80] - polynomal_fit(T_clean, *params_exp) + unp.nominal_values(e_exp)
offset = np.min(I_clean) - np.min(I_clean)*1e-55
I_clean = I_clean - offset
T_max = T_clean[np.argmax(I_clean)]

plt.figure()
plt.plot(T, polynomal_fit(T, *params_exp), 'r-', label='Untergrund')
plt.plot(T, I, 'b+', label='Messwerte')
plt.plot(T_fit, I_fit, 'g+', label='Untergrundfit-Daten')
plt.xlabel('T/K')
plt.ylabel('I/A')
plt.legend()
plt.savefig('../img/heizrate_15C.pdf')
# plt.show()
plt.figure()
plt.plot(T_clean, I_clean, 'b+', label='Bereinigte Daten')
plt.xlabel('T/K')
plt.ylabel('I/A')
plt.legend()
plt.savefig('../img/heizrate_15C_no-bg.pdf')
# plt.show()

# fit
params_lin, cov_lin = curve_fit(linear_fit, 1/T[8:33], np.log(I_clean[8:33]))
errors_lin = np.sqrt(np.diag(cov_lin))

a_lin = ufloat(params_lin[0], errors_lin[0])
b_lin = ufloat(params_lin[1], errors_lin[1])

W_meth1 = - a_lin * Boltzmann / elementary_charge
tau0_1 = relaxationszeit(T_max, 1.5, W_meth1)

print('Einfacher linearer Fit für W:')
print(f'a = {a_lin}')
print(f'b = {b_lin}')
print(f'Aktivierungsenergie W = {W_meth1} eV')
print(f'Relaxationszeit tau0 = {tau0_1} s')

cut = np.log(I_clean) > -40

plt.figure()
plt.plot(1/T_clean[cut], np.log(I_clean[cut]), 'r*', label='Messwerte')
plt.plot(1/T[8:33], np.log(I_clean[8:33]), 'g*', label='Fitdaten')
plt.plot(1/T[8:33], linear_fit(1/T[8:33], *params_lin), 'k-', label='Ausgleichsgerade')
plt.xlabel('1/T / 1/K')
plt.ylabel('ln(I/1A)')
plt.legend()
plt.savefig('../img/hr15_W-aus-Methode1.pdf')

integrated = np.array([])
for i in range(len(T_clean)):
    integrated = np.append(integrated, np.log(np.trapz(I_clean[i:], T_clean[i:]) / I_clean[i]))

# fit
params_linin, cov_linin = curve_fit(linear_fit, 1/T_clean[7:-25], integrated[7:-25])
errors_linin = np.sqrt(np.diag(cov_linin))

a_linin = ufloat(params_linin[0], errors_linin[0])
b_linin = ufloat(params_linin[1], errors_linin[1])

W_meth2 = a_linin * Boltzmann / elementary_charge
tau0_2 = relaxationszeit(T_max, 1.5, W_meth2)

print('Linearer Fit an Integral für W:')
print(f'a = {a_linin}')
print(f'b = {b_linin}')
print(f'Aktivierungsenergie W = {W_meth2} eV')
print(f'Relaxationszeit tau0 = {tau0_2} s')
print(f'T_max = {T_max} K')
print(f'Offset des Messgeraets: I_off = {offset}')

plt.figure()
plt.plot(1/T_clean[:-5], integrated[:-5], 'bx', label='Daten')
plt.plot(1/T_clean[7:-25], integrated[7:-25], 'gx', label='Fitdaten')
plt.plot(1/T_clean[7:-25], linear_fit(1/T_clean[7:-25], *params_linin), 'k-', label='Ausgleichsgerade')
plt.xlabel('1/T / 1/K')
plt.ylabel(r'$\ln(\int I/I)$')
plt.legend()
plt.savefig('../img/hr15_W-aus-Methode2.pdf')

tau_T1 = tau_vs_T(tau0_1, W_meth1, T)
tau_T2 = tau_vs_T(tau0_2, W_meth2, T)

plt.figure()
plt.plot(T, tau_T1, 'rx', label='Methode 1')
plt.plot(T, tau_T2, 'gx', label='Methode 2')
plt.yscale('log')
plt.xlabel('T / K')
plt.ylabel(r'$\tau(T)$ / s')
plt.legend()
plt.savefig('../img/hr15_tau.pdf')

print('\nHeizrate: 2 K/s')
T, I = np.genfromtxt('../data/heizrate_2C.txt', unpack=True)
T = convert_temperature(T, 'C', 'K')
T1 = T[:18]
T2 = T[57:80]
T_fit = np.append(T1, T2)
I1 = I[:18]
I2 = I[57:80]
I_fit = np.append(I1, I2)

# fit
params_exp, cov_exp = curve_fit(polynomal_fit, T_fit, I_fit)
errors_exp = np.sqrt(np.diag(cov_exp))

a_exp = ufloat(params_exp[0], errors_exp[0])
b_exp = ufloat(params_exp[1], errors_exp[1])
c_exp = ufloat(params_exp[2], errors_exp[2])
d_exp = ufloat(params_exp[3], errors_exp[3])
e_exp = ufloat(params_exp[4], errors_exp[4])

print('Background Fit:')
print(f'a = {a_exp}')
print(f'b = {b_exp}')
print(f'c = {c_exp}')
print(f'd = {d_exp}')
print(f'e = {e_exp}')

# Strom ohne Untergrund
T_clean = T[:72]
I_clean = I[:72] - polynomal_fit(T_clean, *params_exp) + unp.nominal_values(e_exp)
offset = np.min(I_clean) - np.min(I_clean)*1e-55
I_clean = I_clean - offset
T_max = T_clean[np.argmax(I_clean)]

plt.figure()
plt.plot(T[:-10], polynomal_fit(T[:-10], *params_exp), 'r-', label='Untergrund')
plt.plot(T, I, 'b+', label='Messwerte')
plt.plot(T_fit, I_fit, 'g+', label='Untergrundfit-Daten')
plt.xlabel('T/K')
plt.ylabel('I/A')
plt.legend()
plt.savefig('../img/heizrate_2C.pdf')
plt.figure()
plt.plot(T_clean, I_clean, 'b+', label='Bereinigte Daten')
plt.xlabel('T/K')
plt.ylabel('I/A')
plt.legend()
plt.savefig('../img/heizrate_2C_no-bg.pdf')

# fit
params_lin, cov_lin = curve_fit(linear_fit, 1/T[15:35], np.log(I_clean[15:35]))
errors_lin = np.sqrt(np.diag(cov_lin))

a_lin = ufloat(params_lin[0], errors_lin[0])
b_lin = ufloat(params_lin[1], errors_lin[1])

W_meth1 = - a_lin * Boltzmann / elementary_charge
tau0_1 = relaxationszeit(T_max, 2, W_meth1)

print('Einfacher linearer Fit für W:')
print(f'a = {a_lin}')
print(f'b = {b_lin}')
print(f'Aktivierungsenergie W = {W_meth1} eV')
print(f'Relaxationszeit tau0 = {tau0_1} s')

cut = np.log(I_clean) > -40

plt.figure()
plt.plot(1/T_clean[cut], np.log(I_clean[cut]), 'r*', label='Messwerte')
plt.plot(1/T[15:35], np.log(I_clean[15:35]), 'g*', label='Fitdaten')
plt.plot(1/T[15:35], linear_fit(1/T[15:35], *params_lin), 'k-', label='Ausgleichsgerade')
plt.xlabel('1/T / 1/K')
plt.ylabel('ln(I/1A)')
plt.legend()
plt.savefig('../img/hr2_W-aus-Methode1.pdf')

integrated = np.array([])
for i in range(len(T_clean)):
    integrated = np.append(integrated, np.log(np.trapz(I_clean[i:], T_clean[i:]) / I_clean[i]))

# fit
params_linin, cov_linin = curve_fit(linear_fit, 1/T_clean[15:-25], integrated[15:-25])
errors_linin = np.sqrt(np.diag(cov_linin))

a_linin = ufloat(params_linin[0], errors_linin[0])
b_linin = ufloat(params_linin[1], errors_linin[1])

W_meth2 = a_linin * Boltzmann / elementary_charge
tau0_2 = relaxationszeit(T_max, 2, W_meth2)

print('Linearer Fit an Integral für W:')
print(f'a = {a_linin}')
print(f'b = {b_linin}')
print(f'Aktivierungsenergie W = {W_meth2} eV')
print(f'Relaxationszeit tau0 = {tau0_2} s')
print(f'T_max = {T_max} K')
print(f'Offset des Messgeraets: I_off = {offset}')

plt.figure()
plt.plot(1/T_clean[:-5], integrated[:-5], 'bx', label='Daten')
plt.plot(1/T_clean[15:-25], integrated[15:-25], 'gx', label='Fitdaten')
plt.plot(1/T_clean[15:-25], linear_fit(1/T_clean[15:-25], *params_linin), 'k-', label='Ausgleichsgerade')
plt.xlabel('1/T / 1/K')
plt.ylabel(r'$\ln(\int I/I)$')
plt.legend()
plt.savefig('../img/hr2_W-aus-Methode2.pdf')

tau_T1 = tau_vs_T(tau0_1, W_meth1, T)
tau_T2 = tau_vs_T(tau0_2, W_meth2, T)

plt.figure()
plt.plot(T, tau_T1, 'rx', label='Methode 1')
plt.plot(T, tau_T2, 'gx', label='Methode 2')
plt.yscale('log')
plt.xlabel('T / K')
plt.ylabel(r'$\tau(T)$ / s')
plt.legend()
plt.savefig('../img/hr2_tau.pdf')
