from core import *
import time

# Pgen = -4 # dBm
Pgen = 0 # dBm
C_ = 121e-12
l_phys = 0.5
freq = 10e9
L0 = 1.0e-9
q  = 0.19

# Create array of L0 values to try
C_list = np.linspace(80e-12, 160e-12, 11)

# Create array of bias values
Ibias = np.linspace(1, 33, 34)*1e-3

# cm = CMap('plasma', data=L0_list)
cm = CMap('cividis', data=C_list)
results = []
for idx, C_ in enumerate(C_list):
	
	logging.main(f"{Fore.LIGHTBLUE_EX}Beginning simulation of C = {rd(C_*1e12,2)} pF{standard_color}")

	lks = LKSystem(Pgen, C_, l_phys, freq, q, L0)
	lks.opt.start_guess_method = GUESS_USE_LAST
	# lks.opt.tol_pcnt = 0.1
	# lks.opt.start_guess_method = GUESS_ZERO_REFLECTION
	lks.solve(Ibias, show_plot_on_conv=False)

	Iac = np.array([x.Iac for x in lks.solution])
	
	results.append( (L0, Iac) )

	plt.plot(Ibias*1e3, Iac*1e3, label=f"{rd(C_*1e12,2)} pF", color=cm(idx))
	
plt.grid()
plt.xlabel("Bias Current (mA)")
plt.ylabel("AC Current Amplitude (mA)")
plt.title("Simulation Summary")
plt.legend()

print(f"C_ values: {C_list}")

plt.show()