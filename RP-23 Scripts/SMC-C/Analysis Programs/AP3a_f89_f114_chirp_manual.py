import matplotlib.pyplot as plt
import numpy as np
import mplcursors

f_min = np.array([ 0.2915, 0.28697, 0.2895, 0.28953, 0.28693, 0.2817 ])
f_max = np.array([ 0.3018, 0.30448, 0.3054, 0.30735, 0.32015, 0.32085  ])
power_dBm = np.array([ -6.62, -3, 0, 6, 9, 12 ])

power_W = 10**(power_dBm/10)*1e-3

f_delta = f_max - f_min

plt.figure(1)
plt.plot(power_dBm, f_min*1e3, color=(0.7, 0, 0), linestyle='--', marker='+')
plt.plot(power_dBm, f_max*1e3, color=(0, 0, 0.7), linestyle='--', marker='+')
plt.grid(True)
plt.xlabel("Power (dBm)")
plt.ylabel("Frequency (MHz)")

plt.figure(2)
plt.plot(power_W, f_min*1e3, color=(0.7, 0, 0), linestyle='--', marker='+')
plt.plot(power_W, f_max*1e3, color=(0, 0, 0.7), linestyle='--', marker='+')
plt.grid(True)
plt.xlabel("Power (W)")
plt.ylabel("Frequency (MHz)")

plt.figure(3)
plt.plot(power_dBm, f_delta*1e3, color=(0, 0.6, 0), linestyle='--', marker='+')
plt.grid(True)
plt.xlabel("Power (dBm)")
plt.ylabel("Frequency Change (MHz)")

plt.figure(4)
plt.plot(power_W, f_delta*1e3, color=(0, 0.6, 0), linestyle='--', marker='+')
plt.grid(True)
plt.xlabel("Power (W)")
plt.ylabel("Frequency Change (MHz)")

mplcursors.cursor(multiple=True)

plt.show()

