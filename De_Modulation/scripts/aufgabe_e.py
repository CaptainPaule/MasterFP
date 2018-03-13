import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

omega_T, U, U_0 = np.genfromtxt('../data/aufgabe_e.txt', unpack=True)

plt.plot(omega_T, U, 'C0')
plt.plot(omega_T, U_0, 'C1')
plt.show()
