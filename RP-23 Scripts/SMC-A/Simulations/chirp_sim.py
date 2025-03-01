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
import collections
from colorama import Fore,Style

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

def on_close(event):
	event.canvas.figure.axes[0].has_been_closed = True

# def find_duplicate_indices(input_list):
# 	from collections import defaultdict
# 
# 	index_map = defaultdict(list)
# 	duplicates = []
# 
# 	for index, value in enumerate(input_list):
# 		index_map[value].append(index)
# 
# 	for indices in index_map.values():
# 		if len(indices) > 1:
# 			duplicates.append(indices)

# 	return duplicates

def find_duplicate_indices(lst):
	duplicates = {}
	counter = collections.Counter(lst)
	
	for index, value in enumerate(lst):
		if counter[value] > 1:
			if value not in duplicates:
				duplicates[value] = []
			duplicates[value].append(index)
	return duplicates

class Waveform:
	''' Represents a waveform, defined by non-uniformly spaced positions. '''
	
	def __init__(self, positions:list, amplitudes:list, log:plf.LogPile):
		
		self.log = log
		self.amplitudes = np.array(amplitudes)
		self.positions = np.array(positions)
		
		self.timestamp = None
	
	def get_positions(self):
		
		return self.positions
	
	def get_envelope(self):
		''' Returns the amplitude of the waveform envelope. Format is a list, with each value referring to self.positions.'''
		
		return self.amplitudes
	
	def propagate(self, vp, dt, sim_area):
		''' Propagate waveform by vp and time '''
		
		self.log.lowdebug(f"Propagating waveform.")
		self.log.lowdebug(f"propagate:  vp = {vp}")
		self.log.lowdebug(f"propagate: dt = {dt}")
		# self.log.lowdebug(f"propagate: amplitudes = {self.amplitudes}")
		# self.log.lowdebug(f"propagate: positions = {self.positions}")
		
		# Update positions
		self.positions += vp*dt
		
		# Re-bin waveform positions
		self.positions = np.floor(self.positions/sim_area.bin_size)*sim_area.bin_size
		
		# Find duplicate indices
		dupls_dict = find_duplicate_indices(self.positions)
		# self.log.lowdebug(f"propagate: dupls: {dupls}")
		
		# Get positions that are duplicated
		dup_pos = dupls_dict.keys()
			
		# Create new positions list
		new_positions = np.unique(self.positions)
		# self.log.lowdebug(f"propagate: new_positions: {new_positions}")
		
		# Iterate over new positions
		new_ampl = []
		for npos in new_positions:
			
			# self.log.lowdebug(f"propagate: npos: {npos}")
			# self.log.lowdebug(f"propagate: dup_pos: {dup_pos}")
			
			# New position has mult Ys to sum
			if npos in dup_pos:
				
				# Sum each duplicate point
				y = 0
				for dup in dupls_dict[npos]:
					y += self.amplitudes[dup]
				try:
					new_ampl.append(float(y))
				except Exception as e:
					self.log.error(f"Tried to add incorrect type to amplitude array. y={y}, type(y)={type(y)}", detail=f"{e}")
			
			# New position does not have mult Ys to sum
			else:
				nav = self.amplitudes[np.where(self.positions == npos)[0][0]]
				new_ampl.append(float(nav))
		
		# Update position and amplitude with new values
		self.positions = new_positions
		self.amplitudes = new_ampl
		
class SimArea:
	
	def __init__(self, bin_size:float, log:plf.LogPile):
		
		self.log = log
		
		self.sim_region = [0, 30]
		self.nonlinear_region = [10, 20]
		
		self.I_star = 2
		self.L0 = 1
		self.C_ = 1
		
		self.v_phase_0 = 1/np.sqrt(self.L0*self.C_)
		
		
		self.bin_size = bin_size
	
	def get_phase_velocities(self, wave):
		''' Returns the phase velocity for a given position with a given amplitude. Position must
		be in ascending order. '''
		
		position = wave.get_positions()
		amplitude = wave.get_envelope()
		
		self.log.lowdebug(f"get_phase_velocities: L0={self.L0}")
		self.log.lowdebug(f"get_phase_velocities: amplitude={amplitude}")
		self.log.lowdebug(f"get_phase_velocities: I_star={self.I_star}")
		
		# Initialize modified phase velocities
		L_tot = self.L0*(1 + np.power(amplitude, 2)/np.power(self.I_star, 2)) # Calcualte total
		v_phase_nl= 1/np.sqrt(L_tot*self.C_) # Calculate phase velocity if in nonlinear region
		
		# Make list of Vp0s
		v_phase_0_list = [self.v_phase_0]*len(position)
		
		# Get bounds of sim area
		xmin = np.min(position)
		xmax = np.max(position)
		
		if xmax < self.nonlinear_region[0]: # Sim is entirely before nonlin region (A)
			return np.array(v_phase_0_list)
		elif xmin > self.nonlinear_region[1]: # Sim is entirely after nonlin region (E)
			return np.array(v_phase_0_list)
		elif xmax <= self.nonlinear_region[1] and xmin >= self.nonlinear_region[0]: # Sim is entirely within nonlinear region (C)
			return np.array(v_phase_nl)
		elif xmin < self.nonlinear_region[0] and xmax > self.nonlinear_region[1]: # Sim spans before, during, and after nonlin region (F)
			
			# Find nonlinear region
			idx_start = find_first_gte_index(position, self.nonlinear_region[0])
			idx_end = find_first_gte_index(position, self.nonlinear_region[1])
			
			# TODO: Should repeat the 'find' class, I can make that more efficient
			return np.array(np.concatenate( (v_phase_0_list[:idx_start], v_phase_nl[idx_start:idx_end+1], v_phase_0_list[idx_end+1:]) ))
		elif xmin < self.nonlinear_region[0] and xmax <= self.nonlinear_region[1]: # Sim starts before, and ends during nonlin region (B)
			idx_start = find_first_gte_index(position, self.nonlinear_region[0])
			
			return np.array(np.concatenate( (v_phase_0_list[:idx_start], v_phase_nl[idx_start:]) ))
		elif xmin >= self.nonlinear_region[0] and xmax > self.nonlinear_region[1]: # Sim starts during, ends after nonlin region (D)
			idx_end = find_first_gte_index(position, self.nonlinear_region[1])
			
			return np.array(np.concatenate( (v_phase_nl[:idx_end+1], v_phase_0_list[idx_end+1:]) ))
		else:
			self.log.critical(f"Logical error in `get_phase_velocities`.", detail=f"xmin={xmin}, xmax={xmax}, Nonlinear region={self.nonlinear_region}")
		# if np.max(position) < self.nonlinear_region[0]: # Sim area ends before nonlinear region
		# 	return v_phase_0_list
		# elif np.max(position) > self.nonlinear_region[1]: # Sim area is entirely after nonlinear region
		# 	return v_phase_0_list
		
		
		# if idx_start == -1 and idx_end == -1: # Nonlinear region is not in range
		# 	# Splice together list
		# elif idx_start == -1: # Starts in nonlinear region
		# 	return np.concatenate( (v_phase_nl[:idx_end+1], v_phase_0_list[idx_end+1:]) )
		# else: # Simulation area spans both sides of nonlinear region
			

class ChirpSimulation:
	
	def __init__(self, wave:Waveform, sim_area:SimArea, log:plf.LogPile, dt:float=1e-9, t_start:float=0, t_stop:float=1):
		
		self.log = log
		
		datestr = f"{datetime.datetime.now()}"
		self.hash_id = hashlib.sha1(datestr.encode("utf-8")).hexdigest()
		self.hash_id_short = self.hash_id[-6:]
		
		self.waveform = wave
		self.sim_area = sim_area
		
		self.t_start = t_start
		self.t_stop = t_stop
		self.dt = dt
		self.t_current = t_start
		self.frame_idx = 0
		
		self.limit_frame_rate = False
		self.fps_limit = 60
	
	def reset(self):
		self.frame_idx = 0
		self.t_current = self.t_start
	
	def set_frame_rate(self, fps:int):
		self.limit_frame_rate = True
		self.fps_limit = fps
	
	def next_frame(self):
		
		self.log.debug(f"Calculating frame {self.frame_idx}. t={self.t_current}, num_points={len(self.waveform.positions)}")
		
		# Get phase velocity
		vp = self.sim_area.get_phase_velocities(self.waveform)
		
		# Update time
		self.t_current += self.dt
		
		# Update waveform
		self.waveform.propagate(vp, self.dt, self.sim_area)
		
		self.frame_idx += 1
	
	def run(self, artist=None, fig=None, manual_step:bool=False):
		
		t0 = time.time()
		self.log.info("Beginning simulation.")
		
		# Configure figure if provided
		if (artist is not None) and (fig is not None):
			
			# Set X-bounds manually. This will not autoset on each frame, so animation can miss.
			artist.axes.set_xlim(self.sim_area.sim_region)
			region_artist = artist.axes.fill_between([self.sim_area.nonlinear_region[0], self.sim_area.nonlinear_region[1]], [-100, -100], [100, 100], color=(0.4, 0.4, 0.4), alpha=0.2)
		
		# Reset time
		self.reset()
		
		# Set a bogus last time so no delay on first frame
		tlast = time.time() - 10
		frame_time = 1/self.fps_limit
		
		if manual_step:
			num_run = 0
		else:
			num_run = -1
		
		# Initialize waveform on graph
		artist.set_data(self.waveform.positions, self.waveform.amplitudes)
		fig.canvas.draw()
		fig.canvas.flush_events()
		
		
		# Run until time runs out
		while (self.t_current < self.t_stop):
			
			# Wait for user input if set to manually step
			if num_run == 0:
				usr_input = input(f"{Fore.LIGHTBLUE_EX}Number of frames to progress (default=1, quit to stop):{Style.RESET_ALL} ")
				if (usr_input.upper() == "QUIT") or (usr_input.upper() == "EXIT"):
					self.log.info("Aborting simulation.")
					break
				try:
					num_run = int(usr_input)
					if num_run < 1:
						num_run = 1
				except:
					num_run = 1
				self.log.info(f"Advancing {num_run} frame(s).")
				tlast = time.time()
				num_run -= 1
			elif num_run > 0:
				num_run -= 1
			
			# Update plot if provided
			if (artist is not None) and (fig is not None):
				if not plt.fignum_exists(fig.number):
					self.log.info(f"Window has been manually closed. Aborting simulation.")
					break
			
			# Calculate next frame
			self.next_frame()
			
			# Update plot if provided
			if (artist is not None) and (fig is not None):
				self.log.lowdebug(f"Updateing graph: pos={self.waveform.positions}")
				self.log.lowdebug(f"Updateing graph: amp={self.waveform.amplitudes}")
				artist.set_data(self.waveform.positions, self.waveform.amplitudes)
				fig.canvas.draw()
				fig.canvas.flush_events()
			
			# Limit frame rate if requested
			if (self.limit_frame_rate) and (num_run != 0):
				# Wait until frame rate is hit
				t_proceed = tlast + frame_time
				while time.time() < t_proceed:
					time.sleep(0.001)
				tlast = time.time()
		
		self.log.info(f"Simulation finished in {time.time()-t0} s.")
	
	# def get_phase_velocity(self):
		
	# 	env = self.waveform.get_envelope()
		
		
		
	