from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

a, s = np.genfromtxt("../data/pol.csv", unpack=True)


plt.plot(a, s, '+', label='Messwerte')

plt.xlabel(r'$\phi$ / °')
plt.ylabel(r'Photostrom \ uA')
plt.legend()
plt.savefig('../img/pol.png')