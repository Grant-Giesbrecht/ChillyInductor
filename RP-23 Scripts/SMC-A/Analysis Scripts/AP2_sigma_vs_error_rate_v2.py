''' Manually transfered the error rates from MATLAB figures to lists seen at the top of this
file. Purpose is to view the trends of the error rate as sigma changes.
'''

import matplotlib.pyplot as plt

sigmas_doub_ns = [10, 20, 30, 37.5, 100]
error_I_doub = [4.1e-2, 1.3e-2, 1.14e-2, 1.23e-2, None]
error_Q_doub = [3.52e-2, 1.15e-2, 1.24e-2, 1.46e-2, 1.5e-2]
pi_cf_doub = [1.0385, 1.0425, 1.037, 1.025, 1.048]
hp_cf_doub = [0.965, 1, 0.9975, 0.9935, 0.9975]

sigmas_dchirp_ns = [20, 20]
error_I_dchirp = [2.14e-2, 3.6e-2]
error_Q_dchirp = [1.01e-3, 2.34e-5]
pi_cf_dchirp = [1.019, 1.019]
hp_cf_dchirp = [0.9975, 0.9975]

sigmas_trad_ns = [10, 20, 30, 37.5, 100]
error_I_trad = [5.43e-3, 7.06e-3, 6.23e-3, 7.58e-3, 6.55e-3]
error_Q_trad = [5.69e-3, 7.39e-3, 6.29e-3, 7.41e-3, 6.16e-3]

# Prepare figure 1
fig1 = plt.figure(1)
gs1 = fig1.add_gridspec(1, 1)
ax1a = fig1.add_subplot(gs1[0, 0])

# Prepare figure 2
fig2 = plt.figure(2)
gs2 = fig2.add_gridspec(1, 1)
ax2a = fig2.add_subplot(gs2[0, 0])

# Plot figure 1
ax1a.plot(sigmas_doub_ns, error_I_doub, linestyle=':', marker='.', color=(0, 0.3, 0.7), label="I, Doubler")
ax1a.plot(sigmas_doub_ns, error_Q_doub, linestyle=':', marker='.', color=(0.7, 0, 0.3), label="Q, Doubler")

ax1a.plot(sigmas_trad_ns, error_I_trad, linestyle=':', marker='s', color=(0, 0.1, 0.5), label="I, Traditional")
ax1a.plot(sigmas_trad_ns, error_Q_trad, linestyle=':', marker='s', color=(0.5, 0, 0.1), label="Q, Traditional")

ax1a.scatter(sigmas_dchirp_ns, error_I_dchirp, marker='o', color=(0, 0, 0.7), label="I, DPD")
ax1a.scatter(sigmas_dchirp_ns, error_Q_dchirp, marker='o', color=(0.7, 0, 0), label="Q, DPD")


ax1a.grid(True)
ax1a.set_xlabel("Pre-Doubled $\\sigma$ (ns)")
ax1a.set_ylabel(f"Error per Gate")
ax1a.set_title("Error Rate Comparison")
ax1a.legend()

# Plot figure 2
ax2a.plot(sigmas_doub_ns, pi_cf_doub, linestyle=':', marker='.', color=(0, 0.6, 0.4), label="$\pi$ Correction")
ax2a.plot(sigmas_doub_ns, hp_cf_doub, linestyle=':', marker='.', color=(0.6, 0.4, 0.6), label="$\pi$/2 Correction")

ax2a.grid(True)
ax2a.set_xlabel("Pre-Doubled $\\sigma$ (ns)")
ax2a.set_ylabel(f"Correction Factor")
ax2a.set_title("Doubler Drive Conditions")
ax2a.legend()

plt.show()