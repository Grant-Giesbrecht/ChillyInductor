''' 
Built off of RP-23, SMC-A, AP-3a. RP23A:RP23D_AP1_a plotted error rate versus sigma for the data pub in
ARFTG paper, it included a direct drive and subharmonic drive for the original chip.

This expands it to include the data from RP-23 SMC-D, the med. length trace and improved bias
test. Goal is to build a sense of how they compare to each other. 
'''

import matplotlib as mpl
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from ganymede import extract_visible_xy
import json
import mplcursors

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pub', help="Publication version.", action='store_true')
parser.add_argument('-s', '--save', help="Save figure to PDF.", action='store_true')
parser.add_argument('-m', '--midas', help="Save figure data to MIDAS text files.", action='store_true')
args = parser.parse_args()

#------------------------------------------------------------------
#--- Create Class for organizing data points

class RP23D_AP1_Datapoint:
	
	def __init__(self, doubler_err:float=None, trad_err:float=None, bypass_err:float=None, bypass_nT2_err:float=None, trad_pi_cf:float=None, subharmonic_med_err:float=None, direct_err:float=None,
			  trad_halfpi_cf:float=None, doubler_pi_cf:float=None, doubler_halfpi_cf:float=None,
			  trad_sigma:float=None, doubler_sigma:float=None, trad_nsigtrunc:float=4, doubler_nsigtrunc:float=2.8284):
		
		self.doubler_err = doubler_err
		self.trad_err = trad_err
		
		self.trad_pi_cf = trad_pi_cf
		self.trad_halfpi_cf = trad_halfpi_cf
		self.doubler_pi_cf = doubler_pi_cf
		self.doubler_halfpi_cf = doubler_halfpi_cf
		
		self.direct_err = direct_err
		self.bypass_err = bypass_err
		self.bypass_nT2_err = bypass_nT2_err
		self.subharmonic_med_err = subharmonic_med_err
		
		self.trad_sigma = trad_sigma
		self.doubler_sigma = doubler_sigma
		if (self.doubler_sigma is None) and (self.trad_sigma is not None):
			self.doubler_sigma = self.trad_sigma*np.sqrt(2)
		
		self.trad_nsigtrunc = trad_nsigtrunc
		self.doubler_nsigtrunc = doubler_nsigtrunc

##------------------------------------------------------------------
#---- Manually enter data

data_points = []
data_points.append(RP23D_AP1_Datapoint( trad_sigma=25e-9, doubler_err = 1.16e-2, trad_err=9e-3,
								doubler_pi_cf=1.03, doubler_halfpi_cf=0.995,
								 trad_pi_cf=1.0125, trad_halfpi_cf=1, direct_err=6.43E-03))

data_points.append(RP23D_AP1_Datapoint( trad_sigma=15e-9, doubler_err = 1.11e-2, trad_err=7.4e-3,
								doubler_pi_cf=1.015, doubler_halfpi_cf=1.003,
								 trad_pi_cf=1.00675, trad_halfpi_cf=1.0063, direct_err=4.4e-03))

data_points.append(RP23D_AP1_Datapoint( trad_sigma=20e-9, doubler_err = 9.94e-3, trad_err=8.2e-3,
								doubler_pi_cf=1.02, doubler_halfpi_cf=1,
								 trad_pi_cf=1.0065, trad_halfpi_cf=1.0074, direct_err=5.15E-03))

data_points.append(RP23D_AP1_Datapoint( trad_sigma=10e-9, doubler_err = 1.51e-2, trad_err=6.31e-3,
								doubler_pi_cf=1.0375, doubler_halfpi_cf=0.99,
								 trad_pi_cf=1.01, trad_halfpi_cf=1.0066, direct_err=2.51e-03))

data_points.append(RP23D_AP1_Datapoint( trad_sigma=30e-9, doubler_err = 9.35e-3, trad_err=9.38e-3,
								doubler_pi_cf=1.03, doubler_halfpi_cf=0.995,
								 trad_pi_cf=1.10125, trad_halfpi_cf=1.00674, direct_err=7.62e-3))

data_points.append(RP23D_AP1_Datapoint( trad_sigma=50e-9, doubler_err = 1.27e-2, trad_err=1.32e-2,
								doubler_pi_cf=1.0425, doubler_halfpi_cf=0.995,
								 trad_pi_cf=1.011, trad_halfpi_cf=1.005, direct_err=1.16e-2 ))

data_points.append(RP23D_AP1_Datapoint( trad_sigma=200e-9, doubler_err = 5.59e-2, trad_err=4.92e-2,
								doubler_pi_cf=1.045, doubler_halfpi_cf=1,
								 trad_pi_cf=0.9763, trad_halfpi_cf=1, direct_err=4.17e-2 ))

data_points.append(RP23D_AP1_Datapoint( trad_sigma=40e-9, doubler_err = 1.16e-2, trad_err=1.2e-2,
								doubler_pi_cf=1.04, doubler_halfpi_cf=0.9925,
								 trad_pi_cf=1.0125, trad_halfpi_cf=1.0063, direct_err=9.24e-3))

data_points.append(RP23D_AP1_Datapoint( trad_sigma=100e-9, doubler_err = 2.45e-2, trad_err=2.7e-2,
								doubler_pi_cf=None, doubler_halfpi_cf=None,
								 trad_pi_cf=1.003, trad_halfpi_cf=1.0046, direct_err=2.50e-2 ))

if args.pub:
	mpl.rcParams['font.family'] = 'sans-serif'
	mpl.rcParams['font.sans-serif'] = ['Arial']
	mpl.rcParams['font.size'] = 12

# data_points.append(RP23D_AP1_Datapoint( trad_sigma=None, doubler_err = None, trad_err=None,
# 								doubler_pi_cf=None, doubler_halfpi_cf=None,
# 								 trad_pi_cf=None, trad_halfpi_cf=None ))

# Sort objects for ascending sigma
data_points.sort(key=lambda x: x.trad_sigma)

trad_sigma = []
trad_err = []
doubler_err = []
direct_err = []
subharm_err = []
for dp in data_points:
	trad_err.append(dp.trad_err)
	doubler_err.append(dp.doubler_err)
	trad_sigma.append(dp.trad_sigma*1e9)
	direct_err.append(dp.direct_err)
	# subharm_err.append(dp.subharmonic_med_err)
##------------------------------------------------------------------
#---- Plot results

subharm_sigma_presquare = np.array([10, 12, 15, 20, 25, 30, 35.35, 40, 70.7, 141, 282])
subharm_sigma = subharm_sigma_presquare/np.sqrt(2)
subharm_err = np.array([8.19E-03, 5.62e-3, 3.32E-03, 4.64E-03, 5.02E-03, 5.64E-03, 6.72e-3 , 8.04E-03, 1.37e-2, 2.78e-2, 5.1e-2])

subharm_max_sigma = [10, 15, 20, 25, 30, 35, 40, 50, 100]
subharm_max_err = [3.614e-3, 4.387e-3, 5.46e-3, 6.987e-3, 8.391e-3, 9.5975e-3, 1.097e-2, 1.405e-2, 3.337e-2]

trad_max_sigma = [10, 15, 20, 25]
trad_max_err = [3.08e-3, 3.7913e-3, 5.0167e-3, 6.637e-3]

# color_trad = (0.7, 0, 0.3)
# color_doub = (0, 0.3, 0.7)

color_trad = (0.3, 0.3, 0.3)
# color_trad = (0.45, 0.45, 0.45)
# color_trad = (0, 179/255, 146/255)
color_doub = (0, 119/255, 179/255) # From TQE template section header color
color_direct = (179/255, 119/255, 0) # From TQE template section header color
color_subh = (119/255, 179/255, 0) # From TQE template section header color
color_subhmax = (0, 179/255, 0) # From TQE template section header color
color_tradhmax = (0.7, 0, 0) # From TQE template section header color


# Prepare figure 1
fig1 = plt.figure(1)
gs1 = fig1.add_gridspec(1, 1)
ax1a = fig1.add_subplot(gs1[0, 0])

ax1a.plot(trad_sigma, doubler_err, linestyle=':', marker='o', color=color_doub, label="Subharmonic Drive, Trace-3", markersize=7)
ax1a.plot(trad_sigma, trad_err, linestyle='--', marker='+', color=color_trad, label="Direct Drive, Trace-3", markersize=10, markeredgewidth=2)
ax1a.plot(trad_sigma, direct_err, linestyle='-.', marker='x', color=color_direct, label="Direct Drive, Trace-2", markersize=10, markeredgewidth=2)
ax1a.plot(subharm_sigma, subharm_err, linestyle=':', marker='v', color=color_subh, label="Subharm, Trace-2", markersize=6, markeredgewidth=2)
ax1a.plot(subharm_max_sigma, subharm_max_err, linestyle=':', marker='*', color=color_subhmax, label="Subharmonic Drive, Trace-2, Long Sigma", markersize=10, markeredgewidth=2)
ax1a.plot(trad_max_sigma, trad_max_err, linestyle=':', marker='+', color=color_tradhmax, label="Traditional Drive, Trace-2, Long Sigma", markersize=10, markeredgewidth=2)

ax1a.grid(True)
ax1a.set_xlabel("Pulse Width Parameter ($\\sigma$) (ns)")

ax1a.set_ylabel(f"Error per Gate")
ax1a.legend()

if args.save:
	fig1.savefig(os.path.join("figures", "RP23D_AP1_a_fig1.pdf"))

def fig_to_midas_dict(fig):
	
	def format_dict(dd):
		return {"pulse_width_sigma_ns":list(dd['x']), "error_per_gate":list(dd['y']), "label":dd['label']}
	
	data_all = extract_visible_xy(fig)
	trad_data = data_all[1]
	doub_data = data_all[0]
	
	return {"subharmonic_trace":format_dict(doub_data), "direct_drive_trace": format_dict(trad_data)}

def fig_to_midas_json(fig, filename):
	
	midas_dict = fig_to_midas_dict(fig)
	
	with open(filename, "w") as json_file:
		json.dump(midas_dict, json_file)

if args.midas:
	print(f"Saving midas data")
	fig_to_midas_json(fig1, os.path.join(".", "midas_data", "Fig4_data.json"))

mplcursors.cursor(multiple=True)


standard_sigmas = np.array([10, 15, 20, 25, 30, 40])

mod_sigmas = standard_sigmas * 2 * 2 *np.sqrt(2)
print(f"Standard sigmas -> 2*2*root(2)")
print(mod_sigmas)
print(mod_sigmas*np.sqrt(2)/8)



plt.show()