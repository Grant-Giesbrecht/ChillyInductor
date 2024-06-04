import matplotlib.pyplot as plt
import numpy as np


def force0y():
	yl_ = plt.ylim()
	plt.ylim(0, yl_[1])

# Import data
V_da_mV = np.array([23, 79, 135, 190, 246, 298, 437, 574, 630, 360])
I_bias_mA = np.array([0, 1.5, 2.99, 4.48, 5.83, 7.27, 10.93, 14.58, 16.07, 9.1])
Vset_V = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.1, 1.2])

# Convert to SI
V_da_V = V_da_mV/1e3
I_bias_A = I_bias_mA/1e3

# Calculate estimated system impedance
Z_sys = Vset_V[1:]/I_bias_A[1:]

# Plot result
# plt.subplots(layout="constrained")

fig = plt.figure(1)
plt.subplot(3, 1, 1)
plt.scatter(I_bias_mA, V_da_mV)
plt.xlabel("Bias Current (mA)")
plt.ylabel("Output Voltage (V)")
# plt.title("Diff Amp Output Scaling")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.scatter(I_bias_mA[1:], V_da_V[1:]/I_bias_A[1:])
plt.xlabel("Bias Current (mA)")
plt.ylabel("Transconductance (Ohms)")
# plt.title("Diff Amp Transconductance")
plt.grid(True)
force0y()

plt.subplot(3, 1, 3)
plt.scatter(Vset_V[1:], Z_sys)
plt.xlabel("Set Voltage (V)")
plt.ylabel("System Impedance (Ohms)")
# plt.title("System Impedance Estimate")
plt.grid(True)
force0y()

plt.tight_layout()

plt.show()
