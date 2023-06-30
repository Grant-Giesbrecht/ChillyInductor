from core import *

############################### CONFIGURE SYSTEM PARAMETERS ##################
Pgen = 0 # dBm
C_ = 121e-12
l_phys = 0.5
freq = 10e9
q = .19
L0 = 1e-6

Ibias = np.linspace(24.2, 26.1, 2)*1e-3

######################### CONFIGURE BASIC SIMULATION ##################

# Initialize system
lks = LKSystem(Pgen, C_, l_phys, freq, q, L0)

# Change options (applies to all simulators)
lks.setopt('start_guess_method', GUESS_ZERO_REFLECTION)
lks.setopt('max_iter', 200)

# Select simulator for system to use
lks.select_simulator(SIMULATOR_ABCD)

# Run simulation - ABCD
lks.solve(Ibias, simulator=SIMULATOR_ABCD)
Ig_ABCD = lks.get_solution(simulator=SIMULATOR_ABCD, parameter='Ig_w')
f_ABCD = lks.get_solution(simulator=SIMULATOR_ABCD, parameter='freq_w')

# Run simulation - P0
lks.solve(Ibias, simulator=SIMULATOR_P0)
Ig_P0 = lks.get_solution(simulator=SIMULATOR_P0, parameter='Ig_w')
f_P0 = lks.get_solution(simulator=SIMULATOR_P0, parameter='freq_w')

# Make plot
plt.figure(1)
plt.plot(f_ABCD/1e9, Ig_ABCD*1e3, linestyle='dashed', marker='o', color=(0.7, 0, 0), label="ABCD-Simulator")
plt.plot(f_P0/1e9, Ig_P0*1e3, linestyle='dashed', marker='o', color=(0, 0, 0.7), label="P0-Simulator")
plt.grid()
plt.xlabel("Bias Current (mA)")
plt.ylabel("Current Amplitude (mA)")

plt.show()