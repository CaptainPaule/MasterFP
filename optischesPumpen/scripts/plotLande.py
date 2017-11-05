import numpy as np 
import matplotlib.pyplot as plt

f, m, b1, b2 = np.genfromtxt('messwerte.txt', unpack=True)

def sweepSpule(Umdrehnungsanzahl):
	I = 0.1 * Umdrehnungsanzahl
	N = 11
	r = 0.1639 # in Meter
	B = 4 * np.pi * 10**(-1) * 0.716 * (I * N)/r
	return B

def horizontalSpule(Umdrehnungsanzahl):
	I = 0.3 * Umdrehnungsanzahl
	N = 154
	r = 0.1570 # in Meter
	B = 4 * np.pi * 10**(-1) * 0.716 * (I * N)/r
	return B

plt.plot(f, sweepSpule(b1) + horizontalSpule(m), 'o', label="Isotop 1")
plt.plot(f, sweepSpule(b2) + horizontalSpule(m), 'o', label="Isotop 2")

plt.xlabel('Frequenz / kHz')
plt.ylabel(r'B-Feld / $\mu$ T')

plt.grid()
plt.legend(loc='best')
plt.savefig('../img/plotLande.pdf')

