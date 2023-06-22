from core import *
import os

# Get target values from MATLAB
Idc_A = np.array([-0.03325, -0.030875, -0.0285, -0.026125, -0.02375, -0.021375, -0.019, -0.016625, -0.01425, -0.011875, -0.0095, -0.007125, -0.00475, -0.002375, 0, 0.002375, 0.00475, 0.007125, 0.0095, 0.011875, 0.01425, 0.016625, 0.019, 0.021375, 0.02375, 0.026125, 0.0285, 0.030875, 0.03325])
Ifund_mA = np.array([1.5131, 1.6246, 1.4932, 1.6711, 1.7109, 1.817, 1.7177, 1.6502, 1.6852, 1.7652, 1.8488, 1.9048, 1.9497, 1.9705, 1.9756, 1.9717, 1.9531, 1.9116, 1.8525, 1.7779, 1.6928, 1.6577, 1.7247, 1.8184, 1.7306, 1.6805, 1.5131, 1.6378, 1.5375])
Ifund_A = Ifund_mA/1e3


Pgen = 0 # dBm
C_ = 121e-12
l_phys = 0.5
freq = 10.04e9
q = 0.190
L0 = 271e-9

Ibias = Idc_A

# Configure simulation
lks = LKSystem(Pgen, C_, l_phys, freq, q, L0)
lks.opt.start_guess_method = GUESS_USE_LAST
lks.opt.use_S21_loss = False

lks.solve(Ibias, show_plot_on_conv=False)

Iac = np.array([x.Iac for x in lks.solution])

# Configure 2nd simulation with system loss
lksL = LKSystem(Pgen, C_, l_phys, freq, q, L0)
lksL.opt.start_guess_method = GUESS_USE_LAST
lksL.configure_loss(file="cryostat_sparams.pkl")

lksL.solve(Ibias, show_plot_on_conv=False)

IacL = np.array([x.Iac for x in lksL.solution])
Iac_2H = np.array([x.spec_Ig[1] for x in lksL.solution])

plt.figure(1)
plt.plot(Ibias*1e3, Iac*1e3, color='b', linestyle='dashed', marker='o')
plt.plot(Ibias*1e3, IacL*1e3, color='g', linestyle='dashed', marker='o')
plt.plot(Ibias*1e3, Ifund_mA, color='r', linestyle='dashed', marker='o')
plt.grid()
plt.xlabel("Bias Current (mA)")
plt.ylabel("AC Current Amplitude (mA)")
plt.title("Simulation Summary")
plt.legend(["Lossless", "Lossy", "Measured"])

plt.figure(2)
plt.plot(Ibias*1e3, Iac_2H, color='g', linestyle='dashed', marker='o')
plt.grid()
plt.xlabel("Bias Current (mA)")
plt.ylabel("AC Current Amplitude (mA)")
plt.title("2nd Harmonic - Simulated with System Loss")

plt.show()