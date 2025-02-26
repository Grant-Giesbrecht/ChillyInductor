import numpy as np
import matplotlib.pyplot as plt


def find_zero_crossings(x, y):
	''' Finds zero crossings of x, then uses y-data to interpolate between the points.'''
	
	signs = np.sign(y)
	sign_changes = np.diff(signs)
	zc_indices = np.where(sign_changes != 0)[0]
	
	# Trim end points - weird things can occur if the waveform starts or ends at zero
	zc_indices = zc_indices[1:-1]
	
	# Interpolate each zero-crossing
	cross_times = []
	for zci in zc_indices:
		dx = x[zci+1] - x[zci]
		dy = y[zci+1] - y[zci]
		frac = np.abs(y[zci]/dy)
		cross_times.append(x[zci]+dx*frac)
	
	return cross_times

t = np.linspace(0, 10, 101)
y = np.sin(2*np.pi*0.5*t)

tzc = find_zero_crossings(t, y)

periods = np.diff(tzc)
freqs = 1/periods

t_freqs = tzc[:-1] + periods/2

fig1 = plt.figure(1)
gs = fig1.add_gridspec(2, 1)
ax1a = fig1.add_subplot(gs[0, 0])
ax1b = fig1.add_subplot(gs[1, 0])

ax1a.plot(t, y, linestyle=':', marker='.', color=(0, 0, 0.65))
ax1a.set_title("Full Waveform")
ax1a.set_xlabel("Time")
ax1a.set_ylabel("Amplitude")
ax1a.grid(True)
ax1a.set_xlim([0, 10])

ax1b.plot(t_freqs, freqs, linestyle=':', marker='+', color=(0, 0.6, 0))
ax1b.set_title("Full Waveform")
ax1b.set_xlabel("Time")
ax1b.set_ylabel("Frequency")
ax1b.grid(True)
ax1b.set_xlim([0, 10])

plt.show()