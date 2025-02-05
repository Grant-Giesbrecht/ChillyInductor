import matplotlib.pyplot as plt
import pandas as pd

#===========================================================================

dfA1 = pd.read_csv("/Volumes/M4 PHD/C1NOBIAS-F24GHZ_6dBm00000.txt", skiprows=4, encoding='utf-8')
dfA4 = pd.read_csv("/Volumes/M4 PHD/C4NOBIAS-F24GHZ_6dBm00000.txt", skiprows=4, encoding='utf-8')

C1COLOR = (0.7, 0.65, 0)
C4COLOR = (0.2, 0.7, 0.1)

fig, axes = plt.subplots(2, 1)

axes[0].plot(dfA1['Time']*1e9, dfA1['Ampl']*1e3, linestyle=':', marker='.', color=C1COLOR)
axes[1].plot(dfA4['Time']*1e9, dfA4['Ampl']*1e3, linestyle=':', marker='.', color=C4COLOR)

for i in range(2):
	axes[i].grid(True)
	axes[i].set_xlabel("Time (ns)")
	axes[i].set_ylabel("Voltage (mV)")
fig.suptitle("No Bias, Input = 2.4 GHz @ 6 dBm")

plt.tight_layout()

#===========================================================================

dfB1 = pd.read_csv("/Volumes/M4 PHD/C10,1VBIAS-F24GHZ_6dBm00000.txt", skiprows=4, encoding='utf-8')
dfB4 = pd.read_csv("/Volumes/M4 PHD/C40,1VBIAS-F24GHZ_6dBm00000.txt", skiprows=4, encoding='utf-8')

C1COLOR = (0.7, 0.65, 0)
C4COLOR = (0.2, 0.7, 0.1)

fig, axes = plt.subplots(2, 1)

axes[0].plot(dfB1['Time']*1e9, dfB1['Ampl']*1e3, linestyle=':', marker='.', color=C1COLOR)
axes[1].plot(dfB4['Time']*1e9, dfB4['Ampl']*1e3, linestyle=':', marker='.', color=C4COLOR)

for i in range(2):
	axes[i].grid(True)
	axes[i].set_xlabel("Time (ns)")
	axes[i].set_ylabel("Voltage (mV)")
fig.suptitle("0,1V Bias, Input = 2.4 GHz @ 6 dBm")

plt.tight_layout()

#===========================================================================

dfC1 = pd.read_csv("/Volumes/M4 PHD/C10,1VBIAS-F48GHZ_2dBm00000.txt", skiprows=4, encoding='utf-8')
dfC4 = pd.read_csv("/Volumes/M4 PHD/C40,1VBIAS-F48GHZ_2dBm00000.txt", skiprows=4, encoding='utf-8')

C1COLOR = (0.7, 0.65, 0)
C4COLOR = (0.2, 0.7, 0.1)

fig, axes = plt.subplots(2, 1)

axes[0].plot(dfC1['Time']*1e9, dfC1['Ampl']*1e3, linestyle=':', marker='.', color=C1COLOR)
axes[1].plot(dfC4['Time']*1e9, dfC4['Ampl']*1e3, linestyle=':', marker='.', color=C4COLOR)

for i in range(2):
	axes[i].grid(True)
	axes[i].set_xlabel("Time (ns)")
	axes[i].set_ylabel("Voltage (mV)")
fig.suptitle("0,1V Bias, Input = 4.8 GHz @ 2 dBm")

plt.tight_layout()

#===========================================================================

dfD1 = pd.read_csv("/Volumes/M4 PHD/C1NOBIAS-F48GHZ_2dBm00000.txt", skiprows=4, encoding='utf-8')
dfD4 = pd.read_csv("/Volumes/M4 PHD/C4NOBIAS-F48GHZ_2dBm00000.txt", skiprows=4, encoding='utf-8')

C1COLOR = (0.7, 0.65, 0)
C4COLOR = (0.2, 0.7, 0.1)

fig, axes = plt.subplots(2, 1)

axes[0].plot(dfD1['Time']*1e9, dfD1['Ampl']*1e3, linestyle=':', marker='.', color=C1COLOR)
axes[1].plot(dfD4['Time']*1e9, dfD4['Ampl']*1e3, linestyle=':', marker='.', color=C4COLOR)

for i in range(2):
	axes[i].grid(True)
	axes[i].set_xlabel("Time (ns)")
	axes[i].set_ylabel("Voltage (mV)")
fig.suptitle("No Bias, Input = 4.8 GHz @ 2 dBm")

plt.tight_layout()

plt.show()