from core import *

Pgen = -4 # dBm
C_ = 121e-12
l_phys = 0.5
freq = 10e9
q = 0.190
L0 = 30e-9

# lks.configure_time_domain()
# lks.convergence_settings()

# lks.solve(Ibias=[-10, 0, 10])
# lks.solution

# lks.crunch(.01, .03, show_plot_td=True, show_plot_spec=True)

Ibias = [.01, .02, .03]

lks = LKSystem(Pgen, C_, l_phys, freq, q, L0)
lks.opt.start_guess_method = GUESS_USE_LAST
lks.solve(Ibias, show_plot_on_conv=True)
