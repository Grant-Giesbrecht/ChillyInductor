from core import *

Pgen = -4 # dBm
C_ = 121e-12
l_phys = 0.5
freq = 10e9
q = 0.190
L0 = 30e-9

lks = LKSystem(Pgen, C_, l_phys, freq, q, L0)
# lks.configure_time_domain()
# lks.convergence_settings()

# lks.solve(Ibias=[-10, 0, 10])
# lks.solution




lks.check_solution(.01, .03, show_plot=True)
lks.get_spec_components(show_plot=False)