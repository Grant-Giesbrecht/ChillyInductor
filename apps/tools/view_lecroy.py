#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from colorama import Fore, Style
from scipy.signal import hilbert
from scipy.signal import butter, lfilter, freqz
import time
import os
import sys
import argparse
import mplcursors

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('-f', '--fft', help='show fft of data', action='store_true')
parser.add_argument('--trimdc', help='Removes DC from FFT data', action='store_true')
parser.add_argument('--f1', help='Plot F1 file over FFT', action='store_true')
parser.add_argument('-z', '--zerocross', help='Zero-crossing analysis', action='store_true')
parser.add_argument('--zcmin', help='Zero-crossing analysis plot, minimum Y value', type=float)
parser.add_argument('--zcmax', help='Zero-crossing analysis plot, maximum Y value', type=float)
args = parser.parse_args()


def run_fft_old(t_si:list, v_si:list):
	''' Original FFT code used for view_lecroy. Replaced with run_fft_windowed on 3-Dec-2025.
	'''
	
	R = 50
	sample_period = (t_si[-1]-t_si[0])/(len(t_si)-1)
	sample_rate = 1/sample_period
	spectrum_double = np.fft.fft(v_si)
	freq_double = np.fft.fftfreq(len(spectrum_double), d=1/sample_rate)
	
	# Get single-ended results
	freq = freq_double[:len(freq_double)//2]
	spectrum_W = (np.abs(spectrum_double[:len(freq_double)//2])**2) / (R * sample_rate) #len(v_si))
	spectrum = 10 * np.log10(spectrum_W*1e3)
	
	return freq, spectrum

def run_fft_windowed(t_si:list, v_si:list):
	''' Original FFT code used for view_lecroy. Replaced with run_fft on 3-Dec-2025.
	'''
	
	print(f"Using windowed FFT.")
	
	R = 50
	
	# Sampling info
	dt = t_si[1] - t_si[0]
	fs = 1.0 / dt
	N = len(t_si)
	
	# Windowing (use Nuttall if you want the cleanest skirts)
	window = np.hanning(N)
	v_win = v_si * window
	
	# Coherent gain of window (needed for proper amplitude)
	U = np.sum(window**2) / N   # window power loss factor
	
	# FFT
	V = np.fft.rfft(v_win)
	freq = np.fft.rfftfreq(N, d=dt)
	
	# Power spectral density (single-sided)
	# PSD_W_per_Hz
	PSD = (np.abs(V)**2) / (R * fs * N * U)
	
	# Convert to dBm/Hz or dBm-bin
	PSD_dBm_per_Hz = 10*np.log10(PSD * 1e3)
	
	return freq, PSD_dBm_per_Hz


# def find_closest_index(lst, X):
# 	closest_index = min(range(len(lst)), key=lambda i: abs(lst[i] - X))
# 	return closest_index

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

# Read file
try:
	df = pd.read_csv(args.filename, skiprows=4, encoding='utf-8')
except:
	print(f"Failed to find file {args.filename}. Aborting.")
	sys.exit()

# Read file
df_f1 = None
if args.f1:
	try:
		filename = args.filename
		f1_fn = filename[2:]
		f1_fn = f"F1{f1_fn}"
		df_f1 = pd.read_csv(f1_fn, skiprows=4, encoding='utf-8')
	except:
		print(f"Failed to find file {f1_fn}. Aborting.")
		sys.exit()

# Create local variables
t = np.array(df['Time']*1e9)
v = np.array(df['Ampl']*1e3)

t_si = np.array(df['Time'])
v_si = np.array(df['Ampl'])

# Create figure
fig1 = plt.figure(1, figsize=(8, 8))

if args.fft and (df_f1 is not None):
	gs = fig1.add_gridspec(4, 1)
	ax1a = fig1.add_subplot(gs[0:2, 0])
	ax1b = fig1.add_subplot(gs[2, 0])
	ax1c = fig1.add_subplot(gs[3, 0])
	
	freq, spectrum = run_fft_windowed(t_si, v_si)
	
	df1_freq = np.array(df_f1['Time']/1e9)
	df1_spec = np.array(df_f1['Ampl'])
	
	ax1b.plot(freq/1e9, spectrum, linestyle=':', marker='.', color=(0, 0.6, 0), label='FFT')
	ax1b.plot(df1_freq, df1_spec, linestyle=':', marker='.', color=(0.56, 0., 0.56), label='Scope spectrum')
	ax1b.set_xlabel(f"Frequency (GHz)")
	ax1b.set_ylabel(f"Power")
	ax1b.set_title(f"Fourier Transform")
	ax1b.set_xlim([4.8, 4.85])
	ax1b.grid(True)
	
	ax1c.plot(df1_freq, df1_spec, linestyle=':', marker='.', color=(0.56, 0., 0.56), label='Scope spectrum')
	ax1c.set_xlabel(f"Frequency (GHz)")
	ax1c.set_ylabel(f"Power")
	ax1c.set_title(f"Scope Fourier Transform")
	ax1c.set_xlim([4.8, 4.85])
	ax1c.grid(True)
	
	fig1.suptitle(args.filename)
	ax1a.set_title(f"Time Domain Signal")
elif args.fft:
	gs = fig1.add_gridspec(4, 1)
	ax1a = fig1.add_subplot(gs[0:2, 0])
	ax1b = fig1.add_subplot(gs[2, 0])
	ax1c = fig1.add_subplot(gs[3, 0])
	
	freq, spectrum = run_fft_windowed(t_si, v_si)
	
	print(spectrum)
	
	# Trim DC if requested
	if args.trimdc:
		try:
			# Trim the first 10 points
			spectrum = spectrum[10:]
			freq = freq[10:]
		except Exception as e:
			print(f"Failed to trim spectrum. ({e})")
			
	
	ax1b.plot(freq/1e9, spectrum, linestyle=':', marker='.', color=(0, 0.6, 0), label='FFT')
	ax1b.set_xlabel(f"Frequency (GHz)")
	ax1b.set_ylabel(f"Power (dBm/Hz)")
	ax1b.set_title(f"Fourier Transform")
	ax1b.grid(True)
	
	ax1c.plot(freq/1e9, spectrum, linestyle=':', marker='.', color=(0, 0.6, 0), label='FFT')
	ax1c.set_xlabel(f"Frequency (GHz)")
	ax1c.set_ylabel(f"Power (dBm/Hz)")
	ax1c.set_title(f"Fourier Transform")
	ax1c.set_xlim([4.8, 4.82])
	ax1c.grid(True)
	
	fig1.suptitle(os.path.basename(args.filename))
	ax1a.set_title(f"Time Domain Signal")
else:
	gs = fig1.add_gridspec(1, 1)
	ax1a = fig1.add_subplot(gs[0, 0])
	ax1a.set_title(f"{args.filename}")

ax1a.plot(t, v, linestyle=':', marker='.', color=(0, 0, 0.65))
ax1a.set_xlabel("Time (ns))")
ax1a.set_ylabel("Voltage (mV)")
ax1a.grid(True)



if args.zerocross:
	tzc = find_zero_crossings(t_si, v_si)
	N_avg = 1	
	# Select every other zero crossing to see full periods and become insensitive to y-offsets.
	tzc_fullperiod = tzc[::2*N_avg]
	periods = np.diff(tzc_fullperiod)
	freqs = (1/periods)*N_avg

	t_freqs = tzc_fullperiod[:-1] + periods/2
	
	fig2 = plt.figure(2)
	gs2 = fig2.add_gridspec(2, 1)
	ax2a = fig2.add_subplot(gs2[0, 0])
	ax2b = fig2.add_subplot(gs2[1, 0])
	
	ax2a.plot(t, v, linestyle=':', marker='.', color=(0, 0, 0.65))
	ax2a.grid(True)
	ax2a.set_xlabel("Time (ns)")
	ax2a.set_ylabel("Frequency (GHz)")
	ax2a.set_title("Zero Crossing Analysis")
	ax2a.set_xlim([np.min(t), np.max(t)])
	
	ax2b.plot(t_freqs*1e9, freqs/1e9, linestyle=':', marker='.', color=(0, 0.65, 0))
	ax2b.grid(True)
	ax2b.set_xlabel("Time (ns)")
	ax2b.set_ylabel("Frequency (GHz)")
	ax2b.set_title("Zero Crossing Analysis")
	ax2b.set_xlim([np.min(t), np.max(t)])
	
	if args.zcmin is not None:
		lim = ax2b.get_ylim()
		ax2b.set_ylim([args.zcmin, lim[1]])
	if args.zcmax is not None:
		lim = ax2b.get_ylim()
		ax2b.set_ylim([lim[0], args.zcmax])
	
	fig2.tight_layout()
	fig2.suptitle(f"File: {os.path.basename(args.filename)}")


mplcursors.cursor(multiple=True)

fig1.tight_layout()

plt.show()