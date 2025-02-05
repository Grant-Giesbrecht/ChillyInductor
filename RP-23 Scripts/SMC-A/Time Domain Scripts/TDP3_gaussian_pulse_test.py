import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from colorama import Fore, Style

pi = 3.1415926535

dfA1 = pd.read_csv("./C1NOBIAS-F4,8GHZ_3dBm00000.txt", skiprows=4, encoding='utf-8')
dfB1 = pd.read_csv("./C10,1VBIAS-F2,4GHZ_3dBm00000.txt", skiprows=4, encoding='utf-8')
dfB2 = pd.read_csv("./C20,1VBIAS-F2,4GHZ_3dBm00000.txt", skiprows=4, encoding='utf-8')

dfC1 = pd.read_csv("./C1NOBIAS-F4,8GHZ_-4dBm00000.txt", skiprows=4, encoding='utf-8')
dfD1 = pd.read_csv("./C10,275VBIAS-F2,4GHZ_-4dBm00000.txt", skiprows=4, encoding='utf-8')
dfD2 = pd.read_csv("./C20,275VBIAS-F2,4GHZ_-4dBm00000.txt", skiprows=4, encoding='utf-8')

red = (0.6, 0, 0)
blue = (0, 0, 0.7)

time_ns = dfC1['Time']*1e9
ampl_mV = dfC1['Ampl']*1e3

time_ns = dfB1['Time']*1e9
ampl_mV = dfB1['Ampl']*1e3

# time_ns = dfB1['Time']*1e9
# ampl_mV = dfB1['Ampl']*1e3 + dfB2['Ampl']*1e3
# ampl_mV = np.sqrt(np.abs(dfB1['Ampl']*1e3))*np.sign(dfB1['Ampl'])

def guassian_sine (x, ampl, b, sigma, omega, phi):
	return np.sin(omega*x-phi)*ampl*np.exp( -(x-b)**2/(2*sigma**2) )

def plot_test(ampl, b, sigma, omega, phi):
	
	soln_y = guassian_sine(time_ns, ampl, b, sigma, omega, phi)
	
	fig4 = plt.figure()
	gs = fig4.add_gridspec(3, 2)
	
	ax0 = fig4.add_subplot(gs[0, 0])
	ax1 = fig4.add_subplot(gs[1, 0])
	ax2 = fig4.add_subplot(gs[0, 1])
	ax3 = fig4.add_subplot(gs[1, 1])
	ax4 = fig4.add_subplot(gs[2, :])
	
	ax0.plot(time_ns, ampl_mV, linestyle=':', marker='.', color=red, alpha=0.2)
	ax1.plot(time_ns, soln_y, linestyle=':', marker='.', color=blue, alpha=0.2)
	
	ax2.plot(time_ns, ampl_mV, linestyle=':', marker='.', color=red, alpha=0.2)
	ax2.set_xlim([25, 27])
	ax3.plot(time_ns, soln_y, linestyle=':', marker='.', color=blue, alpha=0.2)
	ax3.set_xlim([25, 27])
	
	ax4.plot(time_ns, ampl_mV, linestyle=':', marker='.', color=red, alpha=0.2)
	ax4.plot(time_ns, soln_y, linestyle=':', marker='.', color=blue, alpha=0.2)
	
	plt.tight_layout()
	
	plt.show()

# Initial guess
freq = 4.825
p0=[125, 30, 12, 2*3.14159*freq, 0]

# Set bounds
lower = (10, 15, 5, 2*pi*2, -pi)
upper = (150, 45, 35, 2*pi*8, pi)
bounds = (lower, upper)

# Perform fit
param, param_cov = curve_fit(guassian_sine, time_ns, ampl_mV, p0=p0, bounds=bounds)
param_err = np.sqrt(np.diag(param_cov))

pp0, pp1, pp2, pp3, pp4 = param[0], param[1], param[2], param[3], param[4]
pe0, pe1, pe2, pe3, pe4 = param_err[0], param_err[1], param_err[2], param_err[3], param_err[4]
print(f"Fit Results:")
print(f"    Amplitude (mV): {Fore.YELLOW}{pp0}{Style.RESET_ALL} +/- {Fore.CYAN}{pe0}{Style.RESET_ALL}")
print(f"    Time shift (ns): {Fore.YELLOW}{pp1}{Style.RESET_ALL} +/- {Fore.CYAN}{pe1}{Style.RESET_ALL}")
print(f"    sigma (ns): {Fore.YELLOW}{pp2}{Style.RESET_ALL} +/- {Fore.CYAN}{pe2}{Style.RESET_ALL}")
print(f"    Omega (rad/s): {Fore.YELLOW}{pp3}{Style.RESET_ALL} +/- {Fore.CYAN}{pe3}{Style.RESET_ALL}")
print(f"        Freq (GHz): {Fore.YELLOW}{pp3/2/pi}{Style.RESET_ALL} +/- {Fore.CYAN}{pe3/2/pi}{Style.RESET_ALL}")
print(f"    Phi (rad): {Fore.YELLOW}{pp4}{Style.RESET_ALL} +/- {Fore.CYAN}{pe4}{Style.RESET_ALL}")

plot_test(param[0], param[1], param[2], param[3], param[4])