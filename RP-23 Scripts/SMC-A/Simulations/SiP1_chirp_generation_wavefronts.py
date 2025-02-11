import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from colorama import Fore, Style
from scipy.signal import hilbert
from scipy.signal import butter, lfilter, freqz
import os
import datetime

from pylogfile.base import markdown

#=========================== Create initial conditions ===========================
f0_GHz = 4.825

ampl = 180
b = 30
sigma = 8
omega = np.pi*2*f0_GHz
phi = 0

tmax = 65

# Create time points when wavefronts occur
T_ns = 1/f0_GHz
wavefronts = np.arange(0, tmax-T_ns/4+T_ns, T_ns)
wavefronts = wavefronts + T_ns/4

# Create powers
powers = np.exp( -(wavefronts-b)**2/(2*sigma**2) )

#=========================== Apply delay ===========================

# Set some coefficients
length = 1
L0 = 1
I_star = 1
C0 = 1

L_prime = L0*(1 + powers**2/I_star**2) # Calcualte total inductnace
v_phase = 1/np.sqrt(L_prime*C0) # Calculate phase velocity
t_prop = length/v_phase # Calculate propagation time


wavefronts_f = wavefronts + t_prop # Calculate propagated signal

# Calculate resulting frequencies
freqs_0 = 1/np.diff(wavefronts)
times_0 = wavefronts[:-1] + T_ns/2

delta_t_f = np.diff(wavefronts_f)
freqs_f = 1/delta_t_f
times_f = wavefronts_f[:-1] + delta_t_f/2

#=========================== Plot results ===========================

fig1 = plt.figure(figsize=(8,6))
gs = fig1.add_gridspec(3, 1)

ax0 = fig1.add_subplot(gs[0:2, 0])
ax1 = fig1.add_subplot(gs[2, 0])

ax0.set_title("Transformed Pulse: Frequency")
ax0.plot(times_f, freqs_f)
ax0.set_xlabel("Time (ns)")
ax0.set_ylabel("Frequency (GHz)")
ax0.grid(True)
# ax0.set_ylim([4, 5.5])

for wav, env in zip(wavefronts_f, powers):
	env0 = np.abs(env)	
	ax1.plot([wav, wav], [-env0, env], linestyle='-', color=(0.6, 0, 0), alpha=0.2)

ax1.grid(True)
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Amplitude (1)")
ax1.set_title("Transformed Pulse: Time Domain")

plt.tight_layout()




#======================== Print conditions summary ==========================

tab = "    "
print(markdown(f">:aSimulation Conditions Summary:<"))
print(markdown(f"{tab}>:aStimulus:<"))
print(markdown(f"{tab}{tab}ampl: >{ampl}<"))
print(markdown(f"{tab}{tab}b: >{b}<"))
print(markdown(f"{tab}{tab}sigma: >{sigma}<"))
print(markdown(f"{tab}{tab}f0: >{f0_GHz}<"))
print(markdown(f"{tab}{tab}omega: >{omega}<"))
print(markdown(f"{tab}{tab}phi: >{phi}<"))
print(markdown(f"{tab}{tab}tmax: >{tmax}<"))
print(markdown(f"{tab}>:aNonlinearity:< "))
print(markdown(f"{tab}{tab}length: >{length}<"))
print(markdown(f"{tab}{tab}L0: >{L0}<"))
print(markdown(f"{tab}{tab}I_star: >{I_star}<"))
print(markdown(f"{tab}{tab}C0: >{C0}<"))
print(markdown(f">:aSimulation Name:< >:q{os.path.basename(__file__)}<"))
print(markdown(f"{tab}Finish date: >:q{datetime.datetime.now()}<"))

# print(markdown(f"{tab}{tab}: >{}<"))
# print(markdown(f"{tab}{tab}: >{}<"))
# print(markdown(f"{tab}{tab}: >{}<"))
# print(markdown(f"{tab}{tab}: >{}<"))
# print(markdown(f"\t"))
# print(markdown(f"\t"))
# print(markdown(f"\t"))
# print(markdown(f"\t"))
# print(markdown(f"\t"))
# print(markdown(f"\t"))
# print(markdown(f"\t"))


plt.show()
