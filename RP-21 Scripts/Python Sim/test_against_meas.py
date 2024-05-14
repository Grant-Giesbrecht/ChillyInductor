from core import *
import time
import pickle
import os

Pgen = 0 # dBm
C_ = 150.2e-12
l_phys = 0.5
freq = 10e9
L0 = 900e-9*l_phys
q = 0.19


filename = "last_simulation.pkl"

# Open S21 data File
with open("cryostat_sparams.pkl", 'rb') as fh:
	S21_data = pickle.load(fh)

# Get target values from MATLAB (SC4)

V_2H = np.array([0.022715, 0.019268, 0.018873, 0.020213, 0.023781, 0.017135, 0.01579, 0.014872, 0.012785, 0.0098328, 0.0081696, 0.0070002, 0.0055488, 0.003095, 1.5521e-05, 0.003104, 0.0056001, 0.0070699, 0.0082481, 0.0098724, 0.012929, 0.015455, 0.016188, 0.01738, 0.024912, 0.020964, 0.019668, 0.019591, 0.023608])
Idc_A = np.array([-0.03325, -0.030875, -0.0285, -0.026125, -0.02375, -0.021375, -0.019, -0.016625, -0.01425, -0.011875, -0.0095, -0.007125, -0.00475, -0.002375, 0, 0.002375, 0.00475, 0.007125, 0.0095, 0.011875, 0.01425, 0.016625, 0.019, 0.021375, 0.02375, 0.026125, 0.0285, 0.030875, 0.03325])
Ifund_mA = np.array([1.5131, 1.6246, 1.4932, 1.6711, 1.7109, 1.817, 1.7177, 1.6502, 1.6852, 1.7652, 1.8488, 1.9048, 1.9497, 1.9705, 1.9756, 1.9717, 1.9531, 1.9116, 1.8525, 1.7779, 1.6928, 1.6577, 1.7247, 1.8184, 1.7306, 1.6805, 1.5131, 1.6378, 1.5375])
Ifund_A = Ifund_mA/1e3
I2H_matlab = V_2H/50

mid_idx = int(np.floor(len(Idc_A)/2))

# Initialize system
lks = LKSystem(Pgen, C_, l_phys, freq, q, L0)

# Change options (applies to all simulators)
lks.setopt('start_guess_method', GUESS_USE_LAST)
lks.setopt('max_iter', 200)
lks.configure_loss(sparam_data=S21_data)

# Select simulator for system to use
lks.select_simulator(SIMULATOR_ABCD)

# Run hybrid simulation
lks.solve(Idc_A)
lks.solve(Idc_A, simulator=SIMULATOR_P0)

IL_mA = lks.get_solution(simulator=SIMULATOR_ABCD, parameter='IL_w', element=1)*1e3
IL_mAp0 = lks.get_solution(simulator=SIMULATOR_P0, parameter='Ig_w', element=0)*1e3

IL2_mA = lks.get_solution(simulator=SIMULATOR_ABCD, parameter='IL_w', element=2)*1e3
IL2_mAp0 = lks.get_solution(simulator=SIMULATOR_P0, parameter='Ig_w', element=1)*1e3

plt.figure(1)
plt.plot(Idc_A, Ifund_mA, color=(0.7, 0, 0), label="Measurement", linestyle='dashed', marker='o')
plt.plot(Idc_A, IL_mA, color=(0, .7, 0.4), label="ABCD Simulation", linestyle='dashed', marker='o')
plt.plot(Idc_A, IL_mAp0, color=(0, 0.4, 0.7), label="P0 Simulation", linestyle='dashed', marker='o')
plt.grid()
plt.legend()
plt.xlabel("Bias Current (mA)")
plt.ylabel("AC Current Amplitude (mA)")
plt.title(f"Load Current at Fundamental")
plt.legend()

plt.figure(2)
plt.plot(Idc_A, I2H_matlab*1e3, color=(0.7, 0, 0), label="Measurement", linestyle='dashed', marker='+')
plt.plot(Idc_A, IL2_mA, color=(0, .7, 0.4), label="ABCD Simulation", linestyle='dashed', marker='+')
plt.plot(Idc_A, IL2_mAp0, color=(0, 0.4, 0.7), label="P0 Simulation", linestyle='dashed', marker='+')
plt.grid()
plt.legend()
plt.xlabel("Bias Current (mA)")
plt.ylabel("AC Current Amplitude (mA)")
plt.title(f"Load Current at 2nd Harmonic")
plt.legend()

plt

# Force0y
ax = plt.gca()
yl = ax.get_ylim()
ax.set_ylim([0, yl[1]])

plt.show()


