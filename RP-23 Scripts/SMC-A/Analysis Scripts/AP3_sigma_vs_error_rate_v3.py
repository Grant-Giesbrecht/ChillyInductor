''' 
Uses updated timing system w/ 4 sigma on trad, and 2sqrt(2) on doubler.
'''

import matplotlib.pyplot as plt
import numpy as np

#------------------------------------------------------------------
#--- Create Class for organizing data points

class AP3Datapoint:
	
	def __init__(self, doubler_err:float=None, trad_err:float=None, trad_pi_cf:float=None, 
			  trad_halfpi_cf:float=None, doubler_pi_cf:float=None, doubler_halfpi_cf:float=None,
			  trad_sigma:float=None, doubler_sigma:float=None, trad_nsigtrunc:float=4, doubler_nsigtrunc:float=2.8284):
		
		self.doubler_err = doubler_err
		self.trad_err = trad_err
		
		self.trad_pi_cf = trad_pi_cf
		self.trad_halfpi_cf = trad_halfpi_cf
		self.doubler_pi_cf = doubler_pi_cf
		self.doubler_halfpi_cf = doubler_halfpi_cf
		
		self.trad_sigma = trad_sigma
		self.doubler_sigma = doubler_sigma
		if (self.doubler_sigma is None) and (self.trad_sigma is not None):
			self.doubler_sigma = self.trad_sigma*np.sqrt(2)
		
		self.trad_nsigtrunc = trad_nsigtrunc
		self.doubler_nsigtrunc = doubler_nsigtrunc

##------------------------------------------------------------------
#---- Manually enter data

data_points = []
data_points.append(AP3Datapoint( trad_sigma=25e-9, doubler_err = 1.16e-2, trad_err=9e-3,
								doubler_pi_cf=1.03, doubler_halfpi_cf=0.995,
								 trad_pi_cf=1.0125, trad_halfpi_cf=1 ))

data_points.append(AP3Datapoint( trad_sigma=15e-9, doubler_err = 1.11e-2, trad_err=7.4e-3,
								doubler_pi_cf=1.015, doubler_halfpi_cf=1.003,
								 trad_pi_cf=1.00675, trad_halfpi_cf=1.0063 ))

data_points.append(AP3Datapoint( trad_sigma=20e-9, doubler_err = 9.94e-3, trad_err=8.2e-3,
								doubler_pi_cf=1.02, doubler_halfpi_cf=1,
								 trad_pi_cf=1.0065, trad_halfpi_cf=1.0074 ))

data_points.append(AP3Datapoint( trad_sigma=10e-9, doubler_err = 1.51e-2, trad_err=6.31e-3,
								doubler_pi_cf=1.0375, doubler_halfpi_cf=0.99,
								 trad_pi_cf=1.01, trad_halfpi_cf=1.0066 ))

data_points.append(AP3Datapoint( trad_sigma=30e-9, doubler_err = 9.35e-3, trad_err=9.38e-3,
								doubler_pi_cf=1.03, doubler_halfpi_cf=0.995,
								 trad_pi_cf=1.10125, trad_halfpi_cf=1.00674 ))

data_points.append(AP3Datapoint( trad_sigma=50e-9, doubler_err = 1.27e-2, trad_err=1.32e-2,
								doubler_pi_cf=1.0425, doubler_halfpi_cf=0.995,
								 trad_pi_cf=1.011, trad_halfpi_cf=1.005 ))

data_points.append(AP3Datapoint( trad_sigma=200e-9, doubler_err = 5.59e-2, trad_err=4.92e-2,
								doubler_pi_cf=1.045, doubler_halfpi_cf=1,
								 trad_pi_cf=0.9763, trad_halfpi_cf=1 ))

data_points.append(AP3Datapoint( trad_sigma=40e-9, doubler_err = 1.16e-2, trad_err=1.2e-2,
								doubler_pi_cf=1.04, doubler_halfpi_cf=0.9925,
								 trad_pi_cf=1.0125, trad_halfpi_cf=1.0063 ))

data_points.append(AP3Datapoint( trad_sigma=100e-9, doubler_err = 2.45e-2, trad_err=2.7e-2,
								doubler_pi_cf=None, doubler_halfpi_cf=None,
								 trad_pi_cf=1.003, trad_halfpi_cf=1.0046 ))

# data_points.append(AP3Datapoint( trad_sigma=None, doubler_err = None, trad_err=None,
# 								doubler_pi_cf=None, doubler_halfpi_cf=None,
# 								 trad_pi_cf=None, trad_halfpi_cf=None ))

# Sort objects for ascending sigma
data_points.sort(key=lambda x: x.trad_sigma)

trad_sigma = []
trad_err = []
doubler_err = []
for dp in data_points:
	trad_err.append(dp.trad_err)
	doubler_err.append(dp.doubler_err)
	trad_sigma.append(dp.trad_sigma*1e9)

##------------------------------------------------------------------
#---- Plot results

# Prepare figure 1
fig1 = plt.figure(1)
gs1 = fig1.add_gridspec(1, 1)
ax1a = fig1.add_subplot(gs1[0, 0])

# # Prepare figure 2
# fig2 = plt.figure(2)
# gs2 = fig2.add_gridspec(1, 1)
# ax2a = fig2.add_subplot(gs2[0, 0])

# # Prepare figure 1
# fig3 = plt.figure(3)
# gs3 = fig3.add_gridspec(1, 1)
# ax3a = fig3.add_subplot(gs3[0, 0])

# Plot figure 1
# ax1a.semilogy(trad_sigma, doubler_err, linestyle=':', marker='+', color=(0, 0.3, 0.7), label="Doubler")
# ax1a.semilogy(trad_sigma, trad_err, linestyle=':', marker='x', color=(0.7, 0, 0.3), label="Traditional")

ax1a.plot(trad_sigma, doubler_err, linestyle=':', marker='+', color=(0, 0.3, 0.7), label="Doubler")
ax1a.plot(trad_sigma, trad_err, linestyle=':', marker='x', color=(0.7, 0, 0.3), label="Traditional")

# ax1a.plot(sigmas_trad_ns, error_I_trad, linestyle=':', marker='s', color=(0, 0.1, 0.5), label="I, Traditional")
# ax1a.plot(sigmas_trad_ns, error_Q_trad, linestyle=':', marker='s', color=(0.5, 0, 0.1), label="Q, Traditional")

# ax1a.scatter(sigmas_dchirp_ns, error_I_dchirp, marker='o', color=(0, 0, 0.7), label="I, DPD")
# ax1a.scatter(sigmas_dchirp_ns, error_Q_dchirp, marker='o', color=(0.7, 0, 0), label="Q, DPD")


ax1a.grid(True)
ax1a.set_xlabel("$\\sigma$ (ns)")
ax1a.set_ylabel(f"Error per Gate")
ax1a.set_title("Error Rate Comparison")
ax1a.legend()

# # Plot figure 2
# ax2a.plot(sigmas_doub_ns, pi_cf_doub, linestyle=':', marker='.', color=(0, 0.6, 0.4), label="$\\pi$ Correction")
# ax2a.plot(sigmas_doub_ns, hp_cf_doub, linestyle=':', marker='.', color=(0.6, 0.4, 0.6), label="$\\pi$/2 Correction")

# ax2a.grid(True)
# ax2a.set_xlabel("Pre-Doubled $\\sigma$ (ns)")
# ax2a.set_ylabel(f"Correction Factor")
# ax2a.set_title("Doubler Drive Conditions")
# ax2a.legend()

# # Plot figure 3
# # ax3a.plot(sigmas_doub_ns, error_Q_doub, linestyle=':', marker='o', color=(33/255, 91/255, 114/255), label="Doubler")
# # ax3a.plot(sigmas_trad_ns, error_Q_trad, linestyle=':', marker='s', color=(202/255, 158/255, 200/255), label="Traditional")

# # ax3a.plot(sigmas_doub_ns, error_Q_doub, linestyle=':', marker='o', color=(33/255, 170/255, 190/255), label="Doubler")
# # ax3a.plot(sigmas_trad_ns, error_Q_trad, linestyle=':', marker='s', color=(110/255, 30/255, 110/255), label="Traditional")

# ax3a.plot(sigmas_doub_ns, error_sel_doub, linestyle=':', marker='o', color=(35/255, 142/255, 169/255), label="Doubler")
# ax3a.plot(sigmas_trad_ns, error_sel_trad, linestyle=':', marker='s', color=(102/255, 35/255, 170/255), label="Traditional")

# ax3a.grid(True)
# ax3a.set_xlabel("Pre-Doubled $\\sigma$ (ns)")
# ax3a.set_ylabel(f"Error per Gate")
# ax3a.set_title("Error Rate Comparison")
# ax3a.legend()

# plt.savefig("AP2_Error_Rate_Summary.svg")
# plt.savefig("AP2_Error_Rate_Summary.pdf")
# plt.savefig("AP2_Error_Rate_Summary.ps")
# plt.savefig("AP2_Error_Rate_Summary.eps")

plt.show()