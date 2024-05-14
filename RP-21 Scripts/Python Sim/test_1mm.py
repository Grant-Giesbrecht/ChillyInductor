from core import *

############################### CONFIGURE SYSTEM PARAMETERS ##################

Pgen_dBm = 0
C_ = 150.2e-12
l_phys = 1e-3
freq = 10e9
q = 0.01
L0 = 900e-9*l_phys

Ibias = np.linspace(-30, 30, 61)*1e-3

######################### CONFIGURE BASIC SIMULATION ##################

# Initialize system
lks = LKSystem(Pgen_dBm, C_, l_phys, freq, q, L0)

# Change options (applies to all simulators)
lks.setopt('start_guess_method', GUESS_USE_LAST)
lks.setopt('max_iter', 200)

# Select simulator for system to use
lks.select_simulator(SIMULATOR_ABCD)

# Run hybrid simulation
lks.solve(Ibias)

# Get power spectrum
Pfund = lks.calculate(parameter=LKSystem.CPARAM_HARM_LOAD_POWER, param2=1)
Ph2 = lks.calculate(parameter=LKSystem.CPARAM_HARM_LOAD_POWER, param2=2)
Ph3 = lks.calculate(parameter=LKSystem.CPARAM_HARM_LOAD_POWER, param2=3)
Ph4 = lks.calculate(parameter=LKSystem.CPARAM_HARM_LOAD_POWER, param2=4)
Ph5 = lks.calculate(parameter=LKSystem.CPARAM_HARM_LOAD_POWER, param2=5)


plt.figure(1)
plt.plot(Ibias*1e3, lin2dB(Pfund*1e3), label='Fundamental', linestyle='dashed', marker='+', color=(0.6, 0, 0))
plt.plot(Ibias*1e3, lin2dB(Ph2*1e3), label='2nd Harmonic', linestyle='dashed', marker='+', color=(0, 0, 0.7))
plt.plot(Ibias*1e3, lin2dB(Ph3*1e3), label='3rd Harmonic', linestyle='dashed', marker='+', color=(0, 0.7, 0))
plt.xlabel("Bias Current (mA)")
plt.ylabel("Load Power (dBm)")
plt.legend()
plt.grid()
plt.title("1mm Test Case")
plt.show()
