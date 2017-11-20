#!/usr/bin/python
# -*- coding: utf-8 -*-

# imports
import os
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import constants
from uncertainties import ufloat

# read data
f, m, b1, b2 = np.genfromtxt('../data/messwerteLandeFaktor.txt', unpack=True)

# define functions
# calc B for sweepSpule
def sweepSpule(Umdrehnungsanzahl):
	I = 0.1 * Umdrehnungsanzahl
	N = 11
	r = 0.1639 # in Meter
	B = 4 * math.pi * 10**(-7) * 0.716 * (I * N)/r
	return B

# calc B for horizontalSpule
def horizontalSpule(Umdrehnungsanzahl):
	I = 0.3 * Umdrehnungsanzahl
	N = 154
	r = 0.1570 # in Meter
	B = 4 * math.pi * 10**(-7) * 0.716 * (I * N)/r
	return B

# linear fit function
def fitfunction(x, A, B):
	return A*x + B

# export latex table
def exportLatexTable(table, fileName):
	with open('../tex/' + fileName + '.tex', 'wb') as texfile:
		texfile.write("\\begin{tabular}{ccc} \\toprule \n")
		texfile.write("\\centering\n")
		texfile.write(" & A / $\\mu$T/kHz  & B / $\\mu$T \\\\ \\midrule\n")
		texfile.write("Isotop 1 & " + str(round(table[0], 3)) + " $\pm$ " + str(round(table[2], 3)) + " & " + str(round(table[1], 1)) + " $\pm$ " + str(round(table[3], 1)) +  " " + "\\\\\n")
		texfile.write("Isotop 2 & " + str(round(table[4], 3)) + " $\pm$ " + str(round(table[6], 3)) + " & " + str(round(table[5], 1)) + " $\pm$ " + str(round(table[7], 1)) +  " " + "\\\\\n")
		texfile.write("\\bottomrule\n\\end{tabular}\n")

# export result as latex equation
def exportResultAsEquation(name, value, roundMean, roundStd, unit, fileName):
	with open('../tex/' + fileName + '.tex', 'wb') as texfile:
		texfile.write("\\begin{equation} \n")
		equation = name + " = " + str(round(value.n, roundMean)) + "\\pm" + str(round(value.s, roundMean)) + "\\," + unit
		texfile.write(equation + "\n")
		texfile.write("\\end{equation} \n")

# create linspace for fit
x_lin = np.linspace(0, 1000)

# reserve memory to store result
table = list()

# calc and plot fit isotop 1
params, cov = curve_fit(fitfunction, f, sweepSpule(b1) + horizontalSpule(m))
errors = np.sqrt(np.diag(cov))
print("Fit Isotop 1")
print('a =', params[0], '+/-', errors[0])
print('b =', params[1], '+/-', errors[1])
print("")
table.append(params[0]  * 10**6)
table.append(params[1]  * 10**6)
table.append(errors[0]  * 10**6)
table.append(errors[1]  * 10**6)
plt.plot(x_lin, fitfunction(x_lin, *params)  * 10**6, 'r-', label='linearer Fit 1', linewidth=3)

# calc and plot fit isotop 2
params, cov = curve_fit(fitfunction, f, sweepSpule(b2) + horizontalSpule(m))
errors = np.sqrt(np.diag(cov))
print("Fit Isotop 2")
print('a =', params[0], '+/-', errors[0])
print('b =', params[1], '+/-', errors[1])
table.append(params[0]  * 10**6)
table.append(params[1]  * 10**6)
table.append(errors[0]  * 10**6)
table.append(errors[1]  * 10**6)
plt.plot(x_lin, fitfunction(x_lin, *params)  * 10**6, 'b-', label='linearer Fit 2', linewidth=3)

# Plot data
plt.plot(f, (sweepSpule(b1) + horizontalSpule(m)) * 10**6, 'o', label="Isotop 1")
plt.plot(f, (sweepSpule(b2) + horizontalSpule(m)) * 10**6, 'o', label="Isotop 2")
plt.xlabel('Frequenz / kHz')
plt.ylabel(r'B-Feld / $\mu$ T')

# set xlim
plt.xlim(0, 1000)

# create grid and legend
plt.grid()
plt.legend(loc='best')

# save file
plt.savefig('../img/plotLande.pdf')
plt.clf()

# print out results as latex table
exportLatexTable(table, "isotopFitTable")

# calc g_F for both isotops
# electron mass
m = constants.value(u'electron mass')

# electron charge
e = constants.value(u'elementary charge')

# create ufloat values for A1 and A2 and magnetic field components
A1 = ufloat(table[0], table[2])
A2 = ufloat(table[4], table[6])
B1 = ufloat(table[1], table[3])
B2 = ufloat(table[5], table[7])

# calculate
g1 = (4 * math.pi)/(A1 * 10**(-9)) * m/e
g2 = (4 * math.pi)/(A2 * 10**(-9)) * m/e
B = (B1 + B2) / 2

# note:
# g1 is Rb87
# g2 is Rb85

# export results as latexEquation
exportResultAsEquation("g_{F1}", g1, 3, 3, "", "resG1")
exportResultAsEquation("g_{F2}", g2, 3, 3, "", "resG2")
exportResultAsEquation("B_{h}", B, 1, 1, "\\mu\\text{T}", "resB")

# amplitude ratio
amp1 = ufloat(108, 108 * 0.01) #in pixel
amp2 = ufloat(213, 213 * 0.01) #in pixel
ratio = amp1/amp2

# export Equations
exportResultAsEquation("h_1", amp1, 1, 1, "\\text{px}", "ratioRes1")
exportResultAsEquation("h_2", amp2, 1, 1, "\\text{px}", "ratioRes2")
exportResultAsEquation("r", ratio, 3, 3, "", "ratio")

# calculate core spin
# and J = 1/2 (Alkali)
# define electron g factor
ge = ufloat(2.00231930436182, 0.00000000000052)

# the core spin is a discret value
I = np.linspace(0, 5, 11)

# define quantum numbers for selected transition
F = 2
J = 1/2

# calc the ratio of the electron g factor and the spectific isotop g factor
# and build a numpy array
lhs1 = np.full(11, (g1/ge).n)
lhs2 = np.full(11, (g2/ge).n)

# calculate the secound side of equation and look for grahical solution
#define function
def rhs(I, F, J):
	a = copy.copy(I)
	b = copy.copy(F)
	c = copy.copy(J)

	return (b*(b+1)+c*(c+1)-a*(a+1))/(2*b*(b+1))

# plot Isotop 1 coreSpin
plt.plot(I, rhs(I, F, J), 'bo', label="xxx")
plt.plot(I, lhs1, 'rx', label="ddd")

plt.axvline(x = (1.5), linewidth=2, color = 'k')
plt.xlabel('I')
plt.ylabel(r'$\frac{g_F}{g_J}$')

plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
           [r"$0$", r"$\frac{1}{2}$", r"$1$", r"$\frac{3}{2}$", r"$2$", r"$\frac{5}{2}$",
		    r"$3$", r"$\frac{7}{2}$", r"$4$", r"$\frac{9}{2}$", r"$5$"])

# create grid and legend
plt.grid()
plt.legend(loc='best')

# save file
plt.savefig('../img/coreSpin1.pdf')

#clear plot
plt.clf()

# plot Isotop 2 coreSpin
F = 3
plt.plot(I, rhs(I, F, J), 'bo', label="xxx")
plt.plot(I, lhs2, 'rx', label="ddd")

plt.axvline(x = 2.5, linewidth=2, color = 'k')
plt.xlabel('I')
plt.ylabel(r'$\frac{g_F}{g_J}$')

plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
           [r"$0$", r"$\frac{1}{2}$", r"$1$", r"$\frac{3}{2}$", r"$2$", r"$\frac{5}{2}$",
		    r"$3$", r"$\frac{7}{2}$", r"$4$", r"$\frac{9}{2}$", r"$5$"])

# create grid and legend
plt.grid()
plt.legend(loc='best')

# save file
plt.savefig('../img/coreSpin2.pdf')

# clear plot
plt.clf()

# quadratischer Zeeman Effekt
Ehyp = 10**(-24)
B = sweepSpule(2.50) + horizontalSpule(0.25)
BohrMag = 10**(-24)

UHqz = (g1.n)**2 * BohrMag**2 * B**2 / Ehyp

print("Abgeschätzte Größe des quadratischen Zeeman Effektes")
print(UHqz)
print("")

# read data
A1, P1, deltaT1 = np.genfromtxt('../data/messwertePeriodenResonanz1.txt', unpack=True)
A2, P2, deltaT2 = np.genfromtxt('../data/messwertePeriodenResonanz2.txt', unpack=True)

# define function
def hyp(x, a, b, c):
	return a + b * np.power(x-c, -1)

# calc and plot fit isotop 1
#params, cov = curve_fit(hyp, P1, A1)
#errors = np.sqrt(np.diag(cov))

#print("Fit Amplitude/Periode Isotop 1")
#print('a =', params[0], '+/-', errors[0])
#print('b =', params[1], '+/-', errors[1])
#print("")

#x = np.linspace(0, 15, 50)
#plt.plot(P1, A1, 'bo')
#plt.plot(x, hyp(x, *params), 'r-')

# create grid and legend
#plt.grid()
#plt.legend(loc='best')

# save file
#plt.savefig('../img/trans1.pdf')

#clear plot
#plt.clf()

# calc and plot fit isotop 2
#params, cov = curve_fit(hyp, P2, A2)
#errors = np.sqrt(np.diag(cov))

#print("Fit Amplitude/Periode Isotop 2")
#print('a =', params[0], '+/-', errors[0])
#print('b =', params[1], '+/-', errors[1])
#print("")

#x = np.linspace(0, 15, 500)
#plt.plot(P2, A2, 'bo')
#plt.plot(x, hyp(x, *params), 'r-')

# create grid and legend
#plt.grid()
#plt.legend(loc='best')

# save file
#plt.savefig('../img/trans1.pdf')

#clear plot
#plt.clf()
