''' 
Uses updated timing system w/ 4 sigma on trad, and 2sqrt(2) on doubler.
'''

import matplotlib as mpl
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from ganymede import extract_visible_xy
import json

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pub', help="Publication version.", action='store_true')
parser.add_argument('-s', '--save', help="Save figure to PDF.", action='store_true')
parser.add_argument('-m', '--midas', help="Save figure data to MIDAS text files.", action='store_true')
args = parser.parse_args()

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

if args.pub:
	mpl.rcParams['font.family'] = 'sans-serif'
	mpl.rcParams['font.sans-serif'] = ['Arial']
	mpl.rcParams['font.size'] = 12

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

# color_trad = (0.7, 0, 0.3)
# color_doub = (0, 0.3, 0.7)

color_trad = (0.3, 0.3, 0.3)
# color_trad = (0.45, 0.45, 0.45)
# color_trad = (0, 179/255, 146/255)
color_doub = (0, 119/255, 179/255) # From TQE template section header color



# Prepare figure 1
fig1 = plt.figure(1)
gs1 = fig1.add_gridspec(1, 1)
ax1a = fig1.add_subplot(gs1[0, 0])

ax1a.plot(trad_sigma, doubler_err, linestyle=':', marker='o', color=color_doub, label="Subharmonic Drive", markersize=7)
ax1a.plot(trad_sigma, trad_err, linestyle='--', marker='+', color=color_trad, label="Direct Drive", markersize=10, markeredgewidth=2)

ax1a.grid(True)
ax1a.set_xlabel("Pulse Width Parameter ($\\sigma$) (ns)")

ax1a.set_ylabel(f"Error per Gate")
ax1a.legend()

if args.save:
	fig1.savefig(os.path.join("figures", "AP3a_fig1.pdf"))

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


plt.show()