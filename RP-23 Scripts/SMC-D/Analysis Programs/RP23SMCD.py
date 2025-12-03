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
from dataclasses import dataclass, field

@dataclass
class LoadParameters:
	
	filename:str
	
	t_offset:float = field(default=0)
	v_offset:float = field(default=0)
	
	# Time to trim to (after t_offset is applied)
	t_start:float = field(default=None)
	t_end:float = field(default=None)

@dataclass
class AnalysisResult:
	
	params:LoadParameters
	
	t_si:np.ndarray
	v_si:np.ndarray
	
	freqs:np.ndarray
	spectrum:np.ndarray
	
	axes:list

def get_time_series(params:LoadParameters):
	
	# Read file
	try:
		df = pd.read_csv(params.filename, skiprows=4, encoding='utf-8')
	except:
		print(f"Failed to find file {params.filename}. Aborting.")
		sys.exit()
	
	# Get arrays
	t_si = np.array(df['Time']) + params.t_offset
	v_si = np.array(df['Ampl']) + params.v_offset
	
	# Trim to start
	if params.t_start is not None:
		indices = np.where(t_si >= params.t_start)[0]
		idx = indices[0] if indices.size else None
		
		if idx is None:
			print(f"Warning: Ignoring t_start parameter, as it would return 0 points.")
		else:
			print(f"idx start = {idx}")
			t_si = t_si[idx:]
			v_si = v_si[idx:]
	
	# Trim to end
	if params.t_end is not None:
		indices = np.where(t_si > params.t_end)[0]
		idx = indices[0] if indices.size else None
		
		if idx is None:
			print(f"Warning: Ignoring t_end parameter, as it would return 0 points.")
		else:
			print(f"idx end = {idx}")
			t_si = t_si[:idx]
			v_si = v_si[:idx]
	
	return t_si, v_si

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

def plot_time_series(t_si, v_si, fignum:int=1, ax=None):
	
	if ax is None:
		fig1 = plt.figure(fignum)
		gs1 = fig1.add_gridspec(1, 1)
		ax = fig1.add_subplot(gs1[0, 0])
	
	ax.plot(t_si*1e9, v_si*1e3, linestyle=':', marker='.')
	ax.set_xlabel("Time (ns)")
	ax.set_ylabel("Voltage (mV)")
	ax.grid(True)
	
	return ax

def plot_spectrum(freq, spec, fignum:int=1, ax=None):
	
	if ax is None:
		fig1 = plt.figure(fignum)
		gs1 = fig1.add_gridspec(1, 1)
		ax = fig1.add_subplot(gs1[0, 0])
	
	ax.plot(freq/1e9, spec, linestyle=':', marker='.')
	ax.set_xlabel("Frequency (GHz)")
	ax.set_ylabel("PSD (dBm)")
	ax.grid(True)
	
	return ax

def hybrid_plot(t_si, v_si, freq, spectrum, fignum:int=1):
	
	fig1 = plt.figure(fignum)
	gs1 = fig1.add_gridspec(2, 1)
	axa = fig1.add_subplot(gs1[0, 0])
	axb = fig1.add_subplot(gs1[1, 0])
	
	plot_time_series(t_si, v_si, ax=axa)
	plot_spectrum(freq, spectrum, ax=axb)
	
	return axa, axb

def full_analysis(params:LoadParameters, fignum:int=1, title:str=None):
	
	if title is None:
		title = params.filename
	
	# Do analysis
	t_si, v_si = get_time_series(params)
	freq, spec = run_fft_windowed(t_si, v_si)
	axes = hybrid_plot(t_si, v_si, freq, spec, fignum=fignum)
	
	# Set title
	axes[0].set_title(title)
	
	# Zoom axes
	axes[1].set_xlim([0, 3])
	
	# Package result
	result = AnalysisResult(params, t_si, v_si, freq, spec, axes)
	
	return result

def plot_spectrum_overlay(result_list, fignum):
	
	fig1 = plt.figure(fignum)
	gs1 = fig1.add_gridspec(1, 1)
	axa = fig1.add_subplot(gs1[0, 0])
	
	for res in result_list:
		plot_spectrum(res.freqs, res.spectrum, ax=axa)