from core import *
import time

# Pgen = -4 # dBm
Pgen = 0 # dBm
C_ = 121e-12
l_phys = 0.5
freq = 10e9
q = 0.190

# Create array of L0 values to try
L0_list = np.linspace(1.0e-9, 2.0e-9, 11)
Pgen_list = np.linspace(-6, 0, 13)
q_list = np.linspace(0.18, 0.2, 5)

mcs = MulticoreSim()
mcs.sweep_setup(L0=L0_list, Pgen=Pgen_list, q=q_list)
mcs.constants(C_=C_, freq=freq, l_phys=l_phys)
mcs.mcopt.num_cores = 6

mcs.solve_all()
mcs.solve_search()



# Create array of bias values
Ibias = np.linspace(1, 33, 34)*1e-3

# cm = CMap('plasma', data=L0_list)
cm = CMap('cividis', data=L0_list)
results = []
for idx, L0 in enumerate(L0_list):
	
	logging.info(f"{Fore.LIGHTBLUE_EX}Beginning simulation of L0 = {L0*1e9} nH{standard_color}")

	lks = LKSystem(Pgen, C_, l_phys, freq, q, L0)
	lks.opt.start_guess_method = GUESS_USE_LAST
	# lks.opt.tol_pcnt = 0.1
	# lks.opt.start_guess_method = GUESS_ZERO_REFLECTION
	lks.solve(Ibias, show_plot_on_conv=False)

	Iac = np.array([x.Iac for x in lks.solution])
	
	results.append( (L0, Iac) )

	plt.plot(Ibias*1e3, Iac*1e3, label=f"{rd(L0*1e9,2)} nH", color=cm(idx))
	
plt.grid()
plt.xlabel("Bias Current (mA)")
plt.ylabel("AC Current Amplitude (mA)")
plt.title("Simulation Summary")
plt.legend()

print(f"L0 values: {L0_list}")

plt.show()