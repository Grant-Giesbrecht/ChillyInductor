from core import *

############################### CONFIGURE SYSTEM PARAMETERS ##################
Pgen = 0 # dBm
C_ = 121e-12
l_phys = 0.5
freq = 10e9
q = .19
L0 = 890e-9

Ibias = np.linspace(-30, 30, 61)*1e-3

Idc_A = np.array([-0.03325, -0.030875, -0.0285, -0.026125, -0.02375, -0.021375, -0.019, -0.016625, -0.01425, -0.011875, -0.0095, -0.007125, -0.00475, -0.002375, 0, 0.002375, 0.00475, 0.007125, 0.0095, 0.011875, 0.01425, 0.016625, 0.019, 0.021375, 0.02375, 0.026125, 0.0285, 0.030875, 0.03325])
Idc_A = Ibias

# Read file if no data provided
with open("cryostat_sparams.pkl", 'rb') as fh:
	S21_data = pickle.load(fh)

######################### CONFIGURE BASIC SIMULATION ##################

# Initialize system
lks = LKSystem(Pgen, C_, l_phys, freq, q, L0)
lks.configure_loss(sparam_data=S21_data)

# Change options (applies to all simulators)
lks.setopt('start_guess_method', GUESS_ZERO_REFLECTION)
lks.setopt('max_iter', 200)

# Select simulator for system to use
lks.select_simulator(SIMULATOR_ABCD)

# Run simulation
lks.solve(Idc_A)

# Get results
Idc = lks.get_solution(parameter='Ibias_c')
I1_ = lks.get_solution(parameter='IL_w', element=0)
I2_ = lks.get_solution(parameter='IL_w', element=1)
I3_ = lks.get_solution(parameter='IL_w', element=2)

ce1 = lks.calculate(parameter=LKSystem.CPARAM_CONVERSION_EFFIC1)
ce2 = lks.calculate(parameter=LKSystem.CPARAM_CONVERSION_EFFIC2)
tpl = lks.calculate(parameter=LKSystem.CPARAM_TOTAL_LOAD_POWER)
h2pl = lks.calculate(parameter=LKSystem.CPARAM_HARM_LOAD_POWER, param2=2)

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
plt.plot(Idc*1e3, ce1, linestyle='dashed', marker='+', color=(0.4, 0, 0.7), label='Generator Power Based')
plt.plot(Idc*1e3, ce2, linestyle='dashed', marker='o', color=(0, 0.7, 0.5), label='Sum of Load Power Based')
plt.grid()
plt.legend()
plt.xlabel("Bias Current (mA)")
plt.ylabel("Efficiency (%)")
plt.title("Conversion Efficiency")

plt.figure(3)
plt.plot(Idc*1e3, tpl*1e3, linestyle='dashed', marker='+', color=(0.4, 0, 0.7), label='Total Load Power')
plt.plot(Idc*1e3, h2pl*1e3, linestyle='dashed', marker='x', color=(0.7, 0, 0.4), label='2nd Harmonic Load Power')
plt.grid()
plt.legend()
plt.xlabel("Bias Current (mA)")
plt.ylabel("Load (mW)")
plt.title("Load Power")

plt.show()