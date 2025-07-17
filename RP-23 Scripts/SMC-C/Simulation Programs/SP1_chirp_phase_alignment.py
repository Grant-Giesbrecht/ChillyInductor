import matplotlib.pyplot as plt
import numpy as np


length = 0.45
L2 = 30
Y0 = 74.074e-6
wavelength = 0.055
sigma = 50e-9
tau = sigma
w0 = 100e6*2*np.pi
t = np.linspace(-5*tau, 5*tau, 1000)

w = w0 + (4*np.pi*length*L2*Y0)/(wavelength*tau**2) * t * np.exp(-t**2/(tau**2))

envelope = np.exp( -t**2/(2*tau**2))


fig1 = plt.figure(1)
gs1 = fig1.add_gridspec(2, 1)
ax1a = fig1.add_subplot(gs1[0, 0])
ax1b = fig1.add_subplot(gs1[1, 0])

ax1a.plot(t*1e9, envelope, linewidth=2, color=(1, 165/255, 0), label='Approx pulse shape')
ax1b.plot(t*1e9, w/1e6/2/np.pi, linewidth=2, color=(135/255, 206/255, 235/255), label='Chirp frequency')

ax1a.set_title("Approx Pulse Shape")
ax1a.grid(True)
ax1a.set_xlabel("Time (ns)")

ax1b.set_title("Chirp Frequency")
ax1b.grid(True)
ax1b.set_xlabel("Time (ns)")
ax1b.set_ylabel("Frequency (MHz)")

plt.tight_layout()
plt.show()