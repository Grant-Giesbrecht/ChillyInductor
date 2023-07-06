from core import *

############################### CONFIGURE SYSTEM PARAMETERS ##################
Pgen = 0 # dBm
C_ = 121e-12
l_phys = 0.5
freq = 10e9
q = .19
L0 = 1e-6

Ibias = np.linspace(1, 60, 61)*1e-3

Idc_A = np.array([-0.03325, -0.030875, -0.0285, -0.026125, -0.02375, -0.021375, -0.019, -0.016625, -0.01425, -0.011875, -0.0095, -0.007125, -0.00475, -0.002375, 0, 0.002375, 0.00475, 0.007125, 0.0095, 0.011875, 0.01425, 0.016625, 0.019, 0.021375, 0.02375, 0.026125, 0.0285, 0.030875, 0.03325])

# Read file if no data provided
with open("cryostat_sparams.pkl", 'rb') as fh:
	S21_data = pickle.load(fh)

######################### CONFIGURE BASIC SIMULATION ##################

# Initialize system
lks = LKSystem(Pgen, C_, l_phys, freq, q, L0)

# Change options (applies to all simulators)
lks.setopt('start_guess_method', GUESS_ZERO_REFLECTION)
lks.setopt('max_iter', 200)
lks.configure_loss()

# Select simulator for system to use
lks.select_simulator(SIMULATOR_ABCD)

# Run simulation
lks.solve(Idc_A)

# Get results
Idc = lks.get_solution(parameter='Ibias_c')
I1_ = lks.get_solution(parameter='IL_w', element=0)
I2_ = lks.get_solution(parameter='IL_w', element=1)
I3_ = lks.get_solution(parameter='IL_w', element=2)

ce = lks.calculate(parameter=LKSystem.CPARAM_CONVERSION_EFFIC)

plt.figure(1)
plt.plot(Idc*1e3, abs(I1_*1e3), linestyle='dashed', marker='o', color=(0, 0, 0.7), label='Fundamental')
plt.plot(Idc*1e3, abs(I2_*1e3), linestyle='dashed', marker='o', color=(0, 0.7, 0), label='2nd Harm.')
plt.plot(Idc*1e3, abs(I3_*1e3), linestyle='dashed', marker='o', color=(0.7, 0, 0), label='3rd Harm.')
plt.grid()
plt.legend()
plt.xlabel("Bias Current (mA)")
plt.ylabel("Current Amplitude (mA)")
plt.title("Spectral Power")

plt.figure(2)
plt.plot(Idc*1e3, ce, linestyle='dashed', marker='+', color=(0.4, 0, 0.7))
plt.grid()
plt.legend()
plt.xlabel("Bias Current (mA)")
plt.ylabel("Efficiency (%)")
plt.title("Conversion Efficiency")
plt.show()

plt.show()