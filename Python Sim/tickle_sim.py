from core import *

############################### CONFIGURE SYSTEM PARAMETERS ##################
Pgen = 0 # dBm
C_ = 121e-12
l_phys = 0.5
freq = 10e9
q = 0.190
L0 = 269e-9

Ibias = np.linspace(1, 60, 61)*1e-3

freq_tickle = 50e3
Iac_tickle = 10e-3

######################### CONFIGURE BASIC SIMULATION ##################

lks = LKSystem(Pgen, C_, l_phys, freq, q, L0)
lks.opt.start_guess_method = GUESS_USE_LAST
# lks.opt.tol_pcnt = 0.1

lks.solve(Ibias, show_plot_on_conv=False)

Iac = np.array([x.Iac for x in lks.solution])

###################### CONFIGURE SIMULATION WITH TICKLE ##################

lkst = LKSystem(Pgen, C_, l_phys, freq, q, L0)
lkst.opt.start_guess_method = GUESS_USE_LAST
# lkst.opt.tol_pcnt = 10
lkst.configure_tickle(0.001, freq_tickle, 2)
lkst.configure_time_domain(1, 3, 6)
lkst.opt.use_Lk_expansion = False
lkst.opt.remove_td = True

lkst.solve(Ibias, show_plot_on_conv=False)

Iact = np.array([x.Iac for x in lkst.solution])

###################### PLOT ALL ##################

plt.plot(Ibias*1e3, Iac*1e3, color=(0.7, 0, 0), linestyle='dashed', label='No Tickle')
plt.plot(Ibias*1e3, Iact*1e3, color=(0, 0.6, 0), linestyle='dashdot', label=f'Tickle: {int(freq_tickle//1000)} KHz, {int(Iac_tickle*1e3)} mA')
plt.grid()
plt.xlabel("Bias Current (mA)")
plt.ylabel("AC Current Amplitude (mA)")
plt.title("Impact of Tickle on Simulation")
plt.legend()

plt.show()