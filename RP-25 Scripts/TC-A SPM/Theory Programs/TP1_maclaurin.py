import numpy as np
import matplotlib.pyplot as plt

A = 130e-18
B = 65e-18
# x = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
x = np.logspace(-6, 1, 161)
v0 = 3e8

coef_0 = v0*np.sqrt(A)
coef_1 = v0*B/2/np.sqrt(A)
coef_2 = v0*B**2/8/A**1.5
coef_3 = v0*B**3/16/A**2.5

n_elec = v0*np.sqrt(A+B*x)
n_elec_approx1 = coef_0 + coef_1*x # - coef_2*x**2 #+ coef_3*x**3
n_elec_approx2 = coef_0 + coef_1*x - coef_2*x**2 #+ coef_3*x**3
n_elec_approx3 = coef_0 + coef_1*x - coef_2*x**2 + coef_3*x**3

print(f"n_elec = {coef_0} + {coef_1}*x - {coef_2}*x^2 + {coef_3}*x^3")

fig1 = plt.figure(1)
gs1 = fig1.add_gridspec(1, 1)
ax1a = fig1.add_subplot(gs1[0, 0])

ax1a.semilogx(x, n_elec_approx3, label=f"Approx.")
ax1a.semilogx(x, n_elec, label=f"Exact")
ax1a.set_xlabel("Power (W)")
ax1a.set_ylabel("$n_{elec}$")
ax1a.legend()
ax1a.grid(True)


fig2 = plt.figure(2, figsize=(16, 6))
gs = fig2.add_gridspec(3, 4)
ax2a = fig2.add_subplot(gs[0, 0])
# ax2b = fig2.add_subplot(gs[0, 1])
ax2c = fig2.add_subplot(gs[1, 0])
# ax2d = fig2.add_subplot(gs[1, 1])
ax2e = fig2.add_subplot(gs[2, 0])
ax2f = fig2.add_subplot(gs[:, 1])
ax2g = fig2.add_subplot(gs[:, 2])
ax2h = fig2.add_subplot(gs[:, 3])

ax2a.semilogx(x, n_elec_approx1, label=f"Approx. (1st order)")
ax2a.semilogx(x, n_elec, label=f"Exact")
ax2a.set_xlabel("Power (W)")
ax2a.set_ylabel("$n_{elec}$")
ax2a.legend()
ax2a.grid(True)


# ax2b.semilogx(x, np.abs(n_elec_approx1-n_elec), label=f"Approx. (1st order)")
# ax2b.set_xlabel("Power (W)")
# ax2b.set_ylabel("Error (1st order)")
# ax2b.grid(True)

ax2c.semilogx(x, n_elec_approx2, label=f"Approx. (2nd order)")
ax2c.semilogx(x, n_elec, label=f"Exact")
ax2c.set_xlabel("Power (W)")
ax2c.set_ylabel("$n_{elec}$")
ax2c.legend()
ax2c.grid(True)


# ax2d.semilogx(x, np.abs(n_elec_approx2-n_elec), label=f"Approx. (2nd order)")
# ax2d.set_xlabel("Power (W)")
# ax2d.set_ylabel("Error (2nd order)")
# ax2d.grid(True)

ax2e.semilogx(x, n_elec_approx3, label=f"Approx. (3rd order)")
ax2e.semilogx(x, n_elec, label=f"Exact")
ax2e.set_xlabel("Power (W)")
ax2e.set_ylabel("$n_{elec}$")
ax2e.legend()
ax2e.grid(True)


# ax2f.semilogx(x, np.abs(n_elec_approx3-n_elec), label=f"Approx. (3rd order)")
# ax2f.set_xlabel("Power (W)")
# ax2f.set_ylabel("Error (3rd order)")
# ax2f.grid(True)

ax2f.semilogx(x, np.abs(n_elec_approx1-n_elec), label=f"Approx. (1st order)")
ax2f.semilogx(x, np.abs(n_elec_approx2-n_elec), label=f"Approx. (2nd order)")
ax2f.semilogx(x, np.abs(n_elec_approx3-n_elec), label=f"Approx. (3rd order)")
ax2f.set_xlabel("Power (W)")
ax2f.set_ylabel("Error (Absolute)")
ax2f.grid(True)
ax2f.set_xlim([.1, 10])
ax2f.set_ylim([0, 3])
ax2f.legend()

ax2g.semilogx(x, np.abs(n_elec_approx1-n_elec)/n_elec*100, label=f"Approx. (1st order)")
ax2g.semilogx(x, np.abs(n_elec_approx2-n_elec)/n_elec*100, label=f"Approx. (2nd order)")
ax2g.semilogx(x, np.abs(n_elec_approx3-n_elec)/n_elec*100, label=f"Approx. (3rd order)")
ax2g.set_xlabel("Power (W)")
ax2g.set_ylabel("Error (Percent)")
ax2g.grid(True)
ax2g.set_xlim([.1, 10])
# ax2g.set_ylim([0, 3])
ax2g.legend()

ax2h.semilogx(x, np.abs(n_elec_approx1-n_elec)/n_elec*100, label=f"Approx. (1st order)")
ax2h.semilogx(x, np.abs(n_elec_approx2-n_elec)/n_elec*100, label=f"Approx. (2nd order)")
ax2h.semilogx(x, np.abs(n_elec_approx3-n_elec)/n_elec*100, label=f"Approx. (3rd order)")
ax2h.set_xlabel("Power (W)")
ax2h.set_ylabel("Error (Percent)")
ax2h.grid(True)
ax2h.set_xlim([.1, 10])
ax2h.set_ylim([0, 10])
ax2h.legend()

ax2a.set_ylim([0, 10])
ax2c.set_ylim([0, 10])
ax2e.set_ylim([0, 10])

fig2.tight_layout()

plt.show()