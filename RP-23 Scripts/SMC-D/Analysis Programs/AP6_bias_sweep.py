# Basing data off note: obsidian://open?vault=Endurance&file=01%20-%20Dilution%20Refrigerator%20Measurements%2FRP-23%2FSMC-D%20September%20Campaign%2FFinal%20Results%2FSubharmonic%20Measurement%2C%20Bias%20Sweep
#
# Common parameters: 10ns, 600 pulses, using freq doubler.
import matplotlib.pyplot as plt
import numpy as np

bias_volts = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
rb_result = np.array([0.0037452,  0.0058679, 0.005008, 3.614e-3 , 0.003705, 0.0047784])

power_RF_dBm = [-2.23, -3.806, -4.98, -6.29, -6.714, -7.78]

fig1 = plt.figure(1)
gs1 = fig1.add_gridspec(2, 1)
ax1a = fig1.add_subplot(gs1[0, 0])
ax1b = fig1.add_subplot(gs1[1, 0])

ax1a.plot(bias_volts/1039*1e3, rb_result*1e3, linestyle=':', marker='.')
ax1a.grid(True)
ax1a.set_ylim([0, 7])
ax1a.set_xlim([0.1, 0.7])
ax1a.set_xlabel("Bias Current (mA)")
ax1a.set_ylabel(f"Error Per Gate ($x10^3$)")

ax1b.plot(bias_volts/1039*1e3, power_RF_dBm, linestyle=':', marker='.')
ax1b.grid(True)
ax1b.set_xlim([0.1, 0.7])
ax1b.set_xlabel("Bias Current (mA)")
ax1b.set_ylabel(f"RF Power (dBm)")


ax1a.set_title(f"Randomized Benchmark Result over Bias Sweep")

plt.tight_layout()

fig1.savefig("AP6_fig1.pdf")

max_delta = np.max(rb_result)/np.min(rb_result)
print(f"Max delta: {max_delta}")

plt.show()