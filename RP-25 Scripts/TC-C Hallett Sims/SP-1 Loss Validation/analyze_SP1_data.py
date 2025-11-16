from hallett.nltsim.core import *
from hallett.nltsim.analysis import *

srg = load_sim_results("sim_result_test.nltsim")
srg.print_summary()

hp = srg.aux_results['Res0']

#====== Plot results ==========

fig1 = plt.figure(1)
gs1 = fig1.add_gridspec(1, 1)
ax1a = fig1.add_subplot(gs1[0, 0])

c_fund = 'tab:blue'
c_2h = 'tab:orange'
c_3h = 'tab:green'

plt.plot(hp.x_values, hp.f0, marker='x', label='Fundamental, Explicit', color=c_fund, linestyle='--')
plt.plot(hp.x_values, hp.h2, marker='x', label='2nd harmonic, Explicit', color=c_2h, linestyle='--')
plt.plot(hp.x_values, hp.h3, marker='x', label='3rd harmonic, Explicit', color=c_3h, linestyle='--')

plt.xlabel("Bias Voltage (V)")
plt.ylabel("Power at load (dBm)")
plt.title(f"FDTD Simulation, Explicit vs Implicit Updates")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()