import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def three_temp_model(x, a, b):
    return a * np.exp(b * x)

T, I = np.genfromtxt('../data/heizrate_15C.txt', unpack=True)
T1 = T[:15]
T2 = T[57:78]
T_fit = np.append(T1, T2)
I1 = I[:15]
I2 = I[57:78]
I_fit = np.append(I1, I2)

# fit
params, cov = curve_fit(three_temp_model, T_fit, I_fit, p0=[2, 0.1])
errors = np.sqrt(np.diag(cov))

print(f'a = {params[0]} \pm {errors[0]}')
print(f'b = {params[1]} \pm {errors[1]}')

plt.figure()
plt.plot(T, three_temp_model(T, *params), 'r-', label="Untergrund")
plt.plot(T, I, 'b+', label='Messwerte')
plt.plot(T_fit, I_fit, 'g+', label='Untergrundfit-Daten')
plt.xlabel('T/°C')
plt.legend()
plt.savefig('../img/heizrate_15C.pdf')
plt.figure()
plt.plot(T, I - three_temp_model(T, *params), 'b+')
plt.savefig('../img/heizrate_15C_no-bg.pdf')
