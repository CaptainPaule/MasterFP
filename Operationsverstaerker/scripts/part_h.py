import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy.optimize import curve_fit
from detect_peaks import detect_peaks
from textable import table

C1 = 20e-9
R = 10e3

print('f1 = ' + str(1/(2*np.pi*R*C1)))
print('tau = ' + str(20*R*C1))

t, U, _ = np.genfromtxt('../usb/scope_240.csv',unpack=True, delimiter=',')
# 12 aufloesbare Maxima in Plot log(U) gegen t
N = 12
val = np.zeros(N)
val_x = np.zeros(N)

t_p = t[np.logical_and(t > 0.0, t < 0.02)]
U_p = U[np.logical_and(t > 0.0, t < 0.02)]
t_p = t_p[ U_p > 0.0 ]
U_p = U_p[ U_p > 0.0 ]
# indexes = detect_peaks(cb, mph=0.04, mpd=100)
data_array = np.stack((t_p, U_p), axis=-1)
#
# def func(x,a,b):
# 	return x/a+b
#
# def func2(x,a,b,c):
# 	return a*np.exp(x/b)+c
#
# popt, pcov = curve_fit(func,val_x[0:6],val_log[0:6])
# popt1, pcov1 = curve_fit(func2,val_x,val)
#
# print(ufloat(popt[0],pcov[0][0]))
# print(ufloat(popt[1],pcov[1][1]))
#
# x = np.linspace(val_x[0]-0.0015,val_x[5]+0.00001,1e4)
# x1 = np.linspace(val_x[0],val_x[len(val_x)-1],1e4)
#
# plt.plot(x,func(x,*popt),'g-',label=r'Regression $f_h(x)$')
# plt.plot(val_x,val_log,'rx',label=r'Messpunkte')
plt.plot(data_array[:, 0], data_array[:, 1],'rx',label=r'Messdaten')

plt.grid()
plt.xlabel(r'$t/\mathrm{s}$')
plt.ylabel(r'$\log U/\mathrm{V}$')
plt.legend(loc="best")
plt.show()

# plt.savefig('build/h_plot.pdf')
#
# with open('build/h_table.tex', 'w', 'utf-8') as f:
# 	f.write(
# 		table(
# 			[r'$t/\si{\second}$', r'$U/\si{\volt}$', r'$t/\si{\second}$', r'$U/\si{\volt}$'],
# 			[val_x[0:19], val[0:19], val_x[19:38], val[19:38]]
# 			)
# 		)
#
# val = val*1e3
# val_x = val_x*1e3
#
# print(val)
