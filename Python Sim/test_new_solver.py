from core import *

############################### CONFIGURE SYSTEM PARAMETERS ##################
Pgen = 0 # dBm
C_ = 121e-12
l_phys = 0.5
freq = 10e9
q = .19
L0 = 269e-9

Ibias = np.linspace(1, 60, 61)*1e-3

freq_tickle = 50e3
Iac_tickle = 10e-3

# Read file if no data provided
with open("cryostat_sparams.pkl", 'rb') as fh:
	S21_data = pickle.load(fh)

######################### CONFIGURE BASIC SIMULATION ##################

lks = LKSystem(Pgen, C_, l_phys, freq, q, L0)
lks.opt.start_guess_method = GUESS_USE_LAST
# lks.opt.tol_pcnt = 0.1

lks.crunch(.01, .03, show_plot_td=False, show_plot_spec=True)
lks.configure_loss(sparam_data=S21_data)