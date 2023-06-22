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

lks = LKSystem(Pgen, C_, l_phys, freq, q, L0)
lks.opt.start_guess_method = GUESS_ZERO_REFLECTION
lks.opt.max_iter = 200
lks.opt.print_soln_on_converge = True
# lks.configure_loss(sparam_data=S21_data)

# lks.crunch(.01, .03, show_plot_td=False, show_plot_spec=True)


lks.solve(Idc_A, show_plot_on_conv=False)

# Get results
# Iac = np.array([x.Iac for x in lks.solution])
Iac_plot = []
Idc_plot = []
fund = []
I2H = []
I3H = []
for idx, x in enumerate(lks.solution):
	if x.convergence_failure:
		continue
	
	Iac_plot.append(x.Iac)
	Idc_plot.append(Idc_A[idx])
	fund.append(x.spec_IL[1])
	I2H.append(x.spec_IL[2])
	I3H.append(x.spec_IL[3])
Iac_plot = np.array(Iac_plot)
Idc_plot = np.array(Idc_plot)
fund = np.array(fund)
I2H = np.array(I2H)
I3H = np.array(I3H)

plt.figure(1)
plt.plot(Idc_plot*1e3, abs(Iac_plot*1e3), linestyle='dashed', marker='o', color=(0, 0, 0.7))
plt.grid()
plt.xlabel("Bias Current (mA)")
plt.ylabel("Current Amplitude (mA)")

plt.figure(2)
plt.plot(Idc_plot*1e3, abs(fund*1e3), linestyle='dashed', marker='o', color=(0, 0, 0.7), label='Fundamental')
plt.plot(Idc_plot*1e3, abs(I2H*1e3), linestyle='dashed', marker='o', color=(0, 0.7, 0), label='2nd Harm.')
plt.plot(Idc_plot*1e3, abs(I3H*1e3), linestyle='dashed', marker='o', color=(0.7, 0, 0), label='3rd Harm.')
plt.grid()
plt.legend()
plt.xlabel("Bias Current (mA)")
plt.ylabel("Current Amplitude (mA)")
plt.show()

plt.show()