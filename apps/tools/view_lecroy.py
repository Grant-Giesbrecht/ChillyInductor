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
args = parser.parse_args()

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
	
	R = 50
	sample_period = (t_si[-1]-t_si[0])/(len(t_si)-1)
	sample_rate = 1/sample_period
	spectrum_double = np.fft.fft(v_si)
	freq_double = np.fft.fftfreq(len(spectrum_double), d=1/sample_rate)
	
	# Get single-ended results
	freq = freq_double[:len(freq_double)//2]
	spectrum_W = (np.abs(spectrum_double[:len(freq_double)//2])**2) / (R * sample_rate) #len(v_si))
	spectrum = 10 * np.log10(spectrum_W*1e3)
	
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
	
	sample_period = (t_si[-1]-t_si[0])/(len(t_si)-1)
	sample_rate = 1/sample_period
	spectrum_double = np.fft.fft(v_si)
	freq_double = np.fft.fftfreq(len(spectrum_double), d=1/sample_rate)
	
	print(f"Sample rate: {sample_rate}")
	
	# Get single-ended results
	R = 50
	freq = freq_double[:len(freq_double)//2]
	spectrum_W = (np.abs(spectrum_double[:len(freq_double)//2])**2) / (R * sample_rate) #len(v_si))
	spectrum = 10 * np.log10(spectrum_W*1e3)
	
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

mplcursors.cursor(multiple=True)

fig1.tight_layout()

plt.show()