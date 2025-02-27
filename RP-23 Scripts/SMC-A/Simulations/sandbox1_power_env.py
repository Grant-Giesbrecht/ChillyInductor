import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 301)
i = complex(0, 1)

def complex_sin(theta):
	i = complex(0, 1)
	return (np.exp(i*theta) - np.exp(-i*theta))/(2*i)

def complex_cos(theta):
	i = complex(0, 1)
	return (np.exp(i*theta) + np.exp(-i*theta))/(2)

freq = 0.25
# volt = complex_sin(t*2*np.pi*freq)
volt = np.exp(i*t*2*np.pi*freq)

Zload = complex(0, 1)

curr = volt/Zload

pwr1 = np.real(volt*curr)
pwr2 = np.real(volt)*np.real(curr)

plt.plot(t, np.real(volt), label='Voltage', linestyle=':', marker='.')
plt.plot(t, np.real(curr), label='Current', linestyle=':', marker='.')
plt.plot(t, np.real(pwr1), label='pwr1', linestyle=':', marker='s')
plt.plot(t, np.real(pwr2), label='pwr2', linestyle=':', marker='+')
plt.grid()
plt.legend()

plt.show()
