from core import *
import time

# Pgen = -4 # dBm
Pgen = 0 # dBm
C_ = 121e-12
l_phys = 0.5
freq = 10e9
L0 = 1.0e-9

# Create array of L0 values to try
q_list = np.linspace(0.19, 0.21, 11)

# Create array of bias values
Ibias = np.linspace(1, 33, 34)*1e-3

# cm = CMap('plasma', data=L0_list)
cm = CMap('cividis', data=q_list)
results = []
for idx, q in enumerate(q_list):
	
	logging.main(f"{Fore.LIGHTBLUE_EX}Beginning simulation of q = {q*1e3} mA{standard_color}")

	lks = LKSystem(Pgen, C_, l_phys, freq, q, L0)
	lks.opt.start_guess_method = GUESS_USE_LAST
	# lks.opt.tol_pcnt = 0.1
	# lks.opt.start_guess_method = GUESS_ZERO_REFLECTION
	lks.solve(Ibias, show_plot_on_conv=False)

	Iac = np.array([x.Iac for x in lks.solution])
	
	results.append( (L0, Iac) )

	plt.plot(Ibias*1e3, Iac*1e3, label=f"{rd(q*1e3,2)} mA", color=cm(idx))
	
plt.grid()
plt.xlabel("Bias Current (mA)")
plt.ylabel("AC Current Amplitude (mA)")
plt.title("Simulation Summary")
plt.legend()

print(f"q values: {q}")

plt.show()