from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt

############################### USER OPTIONS ############################

# Select Frequencies
f1 = 5e3
f2 = 8e3

# Select sampling options
num_T = 120 # Picks frequency resolution
max_harm = 1
min_pts = 20 # Picks max frequency

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

##################### CALCULATE AND PRINT STATS ##########################

# freq_step = round((spec_freqs[2]-spec_freqs[1])/1e1)*1e2
freq_step = spec_freqs[2]-spec_freqs[1]

print(f"No. Periods: {num_T}")
print(f"No. Harmonics: {max_harm}")
print(f"Min. Pnts: {min_pts}")
print(f"-------------------------------------")
print(f"t max: {t_max*1e3} ms")
print(f"No. Points: {num_pts}")
print(f"dt: {round(dt*1e8)/1e2} us")
print(f"-------------------------------------")
print(f"Max Display Freq: {round(np.max(spec_freqs)/1e1)*1e2} KHz")
print(f"Display Freq Step: {freq_step} KHz")

######################## PLOT RESULTS ###############################

plt.plot(spec_freqs_KHz, spec, '-b')
plt.legend('FFT')
plt.grid()
plt.show()