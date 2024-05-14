from core import *

# Basic system parameters
Pgen = 0 # dBm
C_ = 121e-12
l_phys = 0.5
freq = 10e9
q = 0.190
L0 = 269e-9
L0 = 500e-9

# Create list of currents
Ibias = np.linspace(1, 33, 34)*1e-3

# Create system object and configure simulation
lks = LKSystem(Pgen, C_, l_phys, freq, q, L0)
lks.opt.start_guess_method = GUESS_USE_LAST
lks.opt.tol_pcnt = 0.1
lks.solve(Ibias, show_plot_on_conv=False)

# Pull Iac values from solution data object
Iac = np.array([x.Iac for x in lks.solution])

# Plot results
plt.plot(Ibias*1e3, Iac*1e3, '-g')
plt.grid()
plt.xlabel("Bias Current (mA)")
plt.ylabel("AC Current Amplitude (mA)")
plt.title("Simulation Summary")

plt.show()







