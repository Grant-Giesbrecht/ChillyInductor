from core import *

############################### CONFIGURE SYSTEM PARAMETERS ##################
Pgen = 0 # dBm
C_ = 121e-12
l_phys = 0.5
freq = 10e9
q = 1e6
L0 = 1e-6
Ibias = .02

# Read file if no data provided
with open("cryostat_sparams.pkl", 'rb') as fh:
	S21_data = pickle.load(fh)

######################### CONFIGURE BASIC SIMULATION ##################

lks = LKSystem(Pgen, C_, l_phys, freq, q, L0)
lks.opt.start_guess_method = GUESS_ZERO_REFLECTION
lks.opt.max_iter = 200
lks.opt.print_soln_on_converge = True
# lks.configure_loss(sparam_data=S21_data)

Iac_list = np.linspace(0, .01, 101)

error1 = []
error2 = []
for Iac in Iac_list:
	
	# Crunch the numbers
	lks.crunch(Iac, Ibias, show_plot_td=False, show_plot_spec=True)
	
	# Get errors
	e1 = lks.soln.Ig_w[1] - lks.soln.Iac
	# e2 = lks.soln.spec_Ig_check[0] - lks.soln.IL_w[1]
	
	error1.append(e1)
	# error2.append(e2)
	
plt.plot(Iac_list*1e3, np.array(error1)*1e3, linestyle='dashed', marker='+', color=(0.7, 0, 0), label="Error 1")
# plt.plot(Iac_list*1e3, np.array(error2)*1e3, linestyle='dashed', marker='+', color=(0.3, 0, 0.7), label="Error 2")
plt.xlabel("AC Current Guess (mA)")
plt.ylabel("Current Error (mA)")
plt.title(f"Error Evolution at Ibias = {Ibias*1e3} mA")
plt.grid()
plt.legend()

plt.show()

