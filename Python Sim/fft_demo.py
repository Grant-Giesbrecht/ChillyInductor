from scipy.fft import fft, fftfreq
from scipy import signal

import numpy as np
from colorama import Fore, Style
import matplotlib.pyplot as plt


############################### USER OPTIONS ############################

# Select Frequencies
f1 = 5e3
f2 = 8e3

# Select sampling options
num_T = 1000 # Picks frequency resolution
max_harm = 1
min_pts = 10 # Picks max frequency

enh_fact = 3

####################### CREATE VARS FROM USER OPTIONS ####################

# sampling parameters
t_max = num_T/f2
num_pts = min_pts*max_harm*num_T

# Create time array
t = np.linspace(0.0, t_max, num_pts)
dt = t[1]-t[0]

######################## CREATE WAVEFORM ################################

y = np.sin(f1*2*np.pi*t) + 0.5*np.sin(f2*2.0*np.pi*t)

################# PROCESS WAVEFORM & EXECUTE FFT #######################

# Run FFT
spec_raw = fft(y)[:num_pts//2]

# Fix magnitude to compensate for number of points
spec = 2.0/num_pts*np.abs(spec_raw)

# Calculate X axis
spec_freqs = fftfreq(num_pts,dt)[:num_pts//2]
spec_freqs_KHz = spec_freqs/1e3

############# RESAMPLE WAVEFORM WITH UPCONVERSION ###################

f0 = 3.5e3
f1 = 6.5e3

freq_rs = []
spec_rs = []
for idx, f in enumerate(spec_freqs):
		
	if f >= f0 and f <= f1:
		freq_rs.append(f)
		spec_rs.append(spec[idx])
		

(spec_enh, spec_freqs_enh) = signal.resample(spec_rs, num_pts*enh_fact, freq_rs)

############################# INTEGRATE POWER #######################

f0 = 4.9e3
f1 = 5.1e3

n_pts = 0
Ptot = 0
Ptot_fs = 0
for idx, f in enumerate(spec_freqs):
	if f >= 1e3 and f <= 10e3:
		Ptot_fs += spec[idx]
		
	if f >= f0 and f <= f1:
		Ptot += spec[idx]
		n_pts += 1
		

##################### CALCULATE AND PRINT STATS ##########################

# freq_step = round((spec_freqs[2]-spec_freqs[1])/1e1)*1e2
freq_step = spec_freqs[2]-spec_freqs[1]

c1 = Fore.BLUE
c2 = Fore.YELLOW
c3 = Fore.GREEN
rst = Style.RESET_ALL
print(f"{c3}-------------------------------------{rst}")
print(f"{c1}No. Periods:{c2} {num_T}{rst}")
print(f"{c1}No. Harmonics:{c2} {max_harm}{rst}")
print(f"{c1}Min. Pnts:{c2} {min_pts}{rst}")
print(f"{c3}-------------------------------------{rst}")
print(f"{c1}t max:{c2} {t_max*1e3} ms{rst}")
print(f"{c1}No. Points:{c2} {num_pts}{rst}")
print(f"{c1}dt:{c2} {round(dt*1e8)/1e2} us{rst}")
print(f"{c3}-------------------------------------{rst}")
print(f"{c1}Max Display Freq:{c2} {round(np.max(spec_freqs_KHz)/1e1)*1e2} KHz{rst}")
print(f"{c1}Display Freq Step:{c2} {round(freq_step)} Hz{rst}")
print(f"{c3}------------- 5 KHz Peak -------------{rst}")
print(f"{c1}Integration Total:{c2} {round(Ptot*100)/100}{rst}")
print(f"{c1}Number of matching points:{c2} {n_pts} {rst}")
print(f"{c3}------------- Full Span -------------{rst}")
print(f"{c1}Integration Total:{c2} {round(Ptot_fs*100)/100}{rst}")

######################## PLOT RESULTS ###############################

plt.plot(spec_freqs_KHz, spec, '-b')
plt.plot(spec_freqs_enh/1e3, spec_enh, '-r')
plt.legend('FFT')
plt.grid()
plt.show()