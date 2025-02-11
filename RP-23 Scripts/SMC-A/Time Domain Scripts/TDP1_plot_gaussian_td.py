import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

DATADIR = "G:\ARC0 PhD Data\RP-23 Qubit Readout\Data\SMC-A\Time Domain Measurements"

#===========================================================================

dfA1 = pd.read_csv(f"{DATADIR}/C1NOBIAS-F4,8GHZ_3dBm00000.txt", skiprows=4, encoding='utf-8')
dfB1 = pd.read_csv(f"{DATADIR}/C10,1VBIAS-F2,4GHZ_3dBm00000.txt", skiprows=4, encoding='utf-8')

CaCOLOR = (0.6, 0, 0)
CbCOLOR = (0, 0, 0.7)
CcCOLOR = (0, 0.6, 0)

fig, axes = plt.subplots(2, 1)

axes[0].plot(dfA1['Time']*1e9, dfA1['Ampl']*1e3, linestyle=':', marker='.', color=CaCOLOR)
axes[1].plot(dfB1['Time']*1e9, dfB1['Ampl']*1e3, linestyle=':', marker='.', color=CbCOLOR)

for i in range(2):
	axes[i].grid(True)
	axes[i].set_xlabel("Time (ns)")
	axes[i].set_ylabel("Voltage (mV)")
	# axes[i].set_xlim([25, 27])
	# axes[i].set_ylim([-250, 250])

fig.suptitle("Prf = 3 dBm")
axes[0].set_title("No bias, Drive at 4,8 GHz")
axes[1].set_title("0.1V bias, Drive at 2,4 GHz")

plt.tight_layout()

#===========================================================================

fig2 = plt.figure(2)
ax2 = fig2.gca()

ax2.plot(dfA1['Time']*1e9, dfA1['Ampl']*1e3, linestyle=':', marker='.', color=CaCOLOR, alpha=0.2)
ax2.plot(dfB1['Time']*1e9, dfB1['Ampl']*1e3, linestyle=':', marker='.', color=CbCOLOR, alpha=0.2)

# for i in range(2):
ax2.grid(True)
ax2.set_xlabel("Time (ns)")
ax2.set_ylabel("Voltage (mV)")
# ax2.set_xlim([25, 27])
# ax2.set_ylim([-250, 250])

fig.suptitle("Prf = 3 dBm")
ax2.set_title("No bias, Drive at 4,8 GHz")
ax2.set_title("0.1V bias, Drive at 2,4 GHz")

plt.tight_layout()

#===========================================================================

dfC1 = pd.read_csv(f"{DATADIR}/C1NOBIAS-F4,8GHZ_-4dBm00000.txt", skiprows=4, encoding='utf-8')
dfD1 = pd.read_csv(f"{DATADIR}/C10,275VBIAS-F2,4GHZ_-4dBm00000.txt", skiprows=4, encoding='utf-8')
dfD2 = pd.read_csv(f"{DATADIR}/C20,275VBIAS-F2,4GHZ_-4dBm00000.txt", skiprows=4, encoding='utf-8')

fig2 = plt.figure(3)
ax2 = fig2.gca()

ax2.plot(dfC1['Time']*1e9, dfC1['Ampl']*1e3, linestyle=':', marker='.', color=CaCOLOR, alpha=0.2)
ax2.plot(dfD1['Time']*1e9, dfD1['Ampl']*1e3, linestyle=':', marker='.', color=CbCOLOR, alpha=0.2)
ax2.plot(dfD1['Time']*1e9, (dfD1['Ampl']+dfD2['Ampl'])*1e3, linestyle=':', marker='.', color=CcCOLOR, alpha=0.2)

# for i in range(2):
ax2.grid(True)
ax2.set_xlabel("Time (ns)")
ax2.set_ylabel("Voltage (mV)")
# ax2.set_xlim([25, 27])
# ax2.set_ylim([-250, 250])

fig.suptitle("Prf = 3 dBm")
ax2.set_title("No bias, Drive at 4,8 GHz")
ax2.set_title("0.1V bias, Drive at 2,4 GHz")

plt.tight_layout()



# #===========================================================================

def guassian_sine (x, ampl, b, sigma, omega, phi):
	
	return np.sin(omega*x-phi)*ampl*np.exp( -(x-b)**2/(2*sigma**2) )

param, param_cov = curve_fit(guassian_sine, dfC1['Time']*1e9, dfC1['Ampl']*1e3, p0=[0.125, 25, 12, 2*3*7.2, 0])

soln_y = guassian_sine(dfC1['Time']*1e9, param[0], param[1], param[2], param[3], param[4])

# fig4 = plt.figure(4)
# ax4 = fig4.gca()

# ax4.plot(dfC1['Time']*1e9, dfC1['Ampl']*1e3, linestyle=':', marker='.', color=CaCOLOR, alpha=0.2)
# ax4.plot(dfC1['Time']*1e9, soln_y*1e3, linestyle=':', marker='.', color=CbCOLOR, alpha=0.2)

# plt.show()

def plot_test(ampl, b, sigma, omega, phi):
	
	soln_y = guassian_sine(dfC1['Time']*1e9, ampl, b, sigma, omega, phi)

	fig4, axes4 = plt.subplots(2, 1)

	axes4[0].plot(dfC1['Time']*1e9, dfC1['Ampl']*1e3, linestyle=':', marker='.', color=CaCOLOR, alpha=0.2)
	axes4[1].plot(dfC1['Time']*1e9, soln_y*1e3, linestyle=':', marker='.', color=CbCOLOR, alpha=0.2)

	plt.show()

plot_test(param[0], param[1], param[2], param[3], param[4])