from core import *

############################### CONFIGURE SYSTEM PARAMETERS ##################
Pgen = 0 # dBm
C_ = 121e-12
l_phys = 0.5
freq = 10e9
q = .19
L0 = 890e-9

Ibias = np.linspace(-30, 30, 61)*1e-3

######################### CONFIGURE BASIC SIMULATION ##################

# Initialize system
lks = LKSystem(Pgen, C_, l_phys, freq, q, L0)

# Change options (applies to all simulators)
lks.setopt('start_guess_method', GUESS_ZERO_REFLECTION)
lks.setopt('max_iter', 200)
lks.setopt('convergence_sim', SIMULATOR_ABCD)
lks.setopt('result_sim', SIMULATOR_P0)

# Select simulator for system to use
lks.select_simulator(SIMULATOR_HYBRID)

# Run hybrid simulation
lks.solve(Ibias)

# Get solutions from both simulators
Ig_ABCD = lks.get_solution(simulator=SIMULATOR_ABCD, parameter='Ig_w')
Ig1_ABCD = lks.get_solution(simulator=SIMULATOR_ABCD, parameter='Ig_w', element=1)
f_ABCD = lks.get_solution(simulator=SIMULATOR_ABCD, parameter='freq_w')
Ig_P0 = lks.get_solution(simulator=SIMULATOR_P0, parameter='Ig_w')
Ig1_P0 = lks.get_solution(simulator=SIMULATOR_P0, parameter='Ig_w', element=1)
f_P0 = lks.get_solution(simulator=SIMULATOR_P0, parameter='freq_w')

# Make plot
plt.figure(1)
plt.plot(f_ABCD[0]/1e9, Ig_ABCD[0]*1e3, linestyle='dashed', marker='o', color=(0.7, 0, 0), label="ABCD-Simulator")
plt.plot(f_P0[0]/1e9, Ig_P0[0]*1e3, linestyle='dashed', marker='o', color=(0, 0, 0.7), label="P0-Simulator")
plt.legend()
plt.xlabel("Frequency (GHz)")
plt.ylabel("Current Amplitude (mA)")
plt.title("Freq vs Current at 0 Bias")
plt.grid()

plt.figure(2)
plt.plot(Ibias*1e3, Ig1_ABCD*1e3, linestyle='dashed', marker='+', color=(0.5, 0, 0), label="ABCD-Simulator")
plt.plot(Ibias*1e3, Ig1_P0*1e3, linestyle='dashed', marker='+', color=(0, 0, 0.5), label="P0-Simulator")
plt.grid()
plt.legend()
plt.xlabel("Bias Current (mA)")
plt.ylabel("Current Amplitude (mA)")
plt.title("Current vs Bias at Fundamental")

plt.show()