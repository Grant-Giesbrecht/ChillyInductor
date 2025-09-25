import matplotlib.pyplot as plt
import numpy as np
import mplcursors

f_min = np.array([0.09894, 0.09939, 0.09885, 0.09776, 0.09737, 0.09651, 0.09599])
f_max = np.array([0.10037, 0.10009,  0.10075, 0.10135, 0.10198, 0.1033, 0.10385])
power_dBm = np.array([-6, -3, 0, 3, 6, 9, 12])
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
plt.title("Traditional, $\sigma$=50 ns, IF=100MHz")
plt.xlabel("Power (dBm)")
plt.ylim([0, 10])
plt.ylabel("Frequency Change (MHz)")

plt.figure(4)
plt.plot(power_W, f_delta*1e3, color=(0, 0.6, 0), linestyle='--', marker='+')
plt.grid(True)
plt.xlabel("Power (W)")
plt.ylim([0, 10])
plt.ylabel("Frequency Change (MHz)")

mplcursors.cursor(multiple=True)

plt.show()

