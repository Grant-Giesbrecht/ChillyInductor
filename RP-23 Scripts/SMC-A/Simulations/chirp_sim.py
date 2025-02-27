import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import pandas as pd
import numpy as np
import os
import datetime
import hashlib
import matplotlib.animation as animation
import copy
import time
from matplotlib.animation import FFMpegWriter
import argparse

import pylogfile.base as plf
from pylogfile.base import markdown

def find_first_gte_index(lst, X):
	for index, value in enumerate(lst):
		if value >= X:
			return index
	return -1  # Return -1 if no such value is found

def find_first_lte_index(lst, X):
	for index, value in enumerate(lst):
		if value <= X:
			return index
	return -1  # Return -1 if no such value is found

def find_duplicate_indices(input_list):
	from collections import defaultdict

	index_map = defaultdict(list)
	duplicates = []

	for index, value in enumerate(input_list):
		index_map[value].append(index)

	for indices in index_map.values():
		if len(indices) > 1:
			duplicates.append(indices)

	return duplicates

# Example usage:
input_list = [1, 2, 3, 2, 1, 4, 5, 1]
print(find_duplicate_indices(input_list))

class Waveform:
	''' Represents a waveform, defined by non-uniformly spaced positions. '''
	
	def __init__(self):
		
		self.amplitudes = []
		self.positions = []
		
		self.timestamp = None
	
	def get_envelope(self):
		''' Returns the amplitude of the waveform envelope. Format is a list, with each value referring to self.positions.'''
		
		return self.amplitudes
	
	def propagate(self, vp, dt, sim_area):
		''' Propagate waveform by vp and time '''
		
		# Update positions
		self.positions += vp*dt
		
		# Re-bin waveform positions
		self.positions = np.floor(self.positions/sim_area.bin_size)
		
		# Find duplicates
		dupls = find_duplicate_indices(self.positions)
		for dup in dupls:
			
			#TODO
		
class SimArea:
	
	def __init__(self, sim_length:int, bin_size:float):
		
		self.sim_region = [0, 30]
		self.nonlinear_region = [10, 20]
		
		self.I_star = 1
		self.L0 = 1
		self.C_ = 1
		
		self.v_phase_0 = 1/np.sqrt(self.L0*self.C_)
		self.v_phase_0_list = [self.v_phase_0]*len(sim_length)
		
		self.bin_size = bin_size
	
	def get_phase_velocities(self, wave):
		''' Returns the phase velocity for a given position with a given amplitude. Position must
		be in ascending order. '''
		
		position = wave.get_positions()
		amplitude = wave.get_envelope()
		
		# Initialize modified phase velocities
		L_tot = self.L0*(1 + amplitude**2/self.I_star**2) # Calcualte total
		v_phase_nl= 1/np.sqrt(L_tot*self.C_) # Calculate phase velocity if in nonlinear region
		
		# Find nonlinear region
		idx_start = find_first_gte_index(position, self.nonlinear_region[0])
		idx_end = find_first_gte_index(position, self.nonlinear_region[1])
		
		# Splice together list
		return np.concatenate( (self.v_phase_0_list[:idx_start], self.v_phase_0_list[idx_start:idx_end+1], self.v_phase_0_list[self.idx_end+1:]) )

class ChirpSimulation:
	
	def __init__(self, wave:Waveform, sim_area:SimArea, log:plf.LogPile, dt:float=1e-9):
		
		self.waveform = wave
		self.sim_area = sim_area
		
		self.dt = 1e-9
	
	def next_frame(self):
		
		# Get envelope
		
		# Get phase velocity
		vp = self.sim_area.get_phase_velocities(self.waveform)
		
		# Update waveform
		self.wave.propagate(vp, self.dt)
	
	def get_phase_velocity(self):
		
		env = self.waveform.get_envelope()
		
		
		
	