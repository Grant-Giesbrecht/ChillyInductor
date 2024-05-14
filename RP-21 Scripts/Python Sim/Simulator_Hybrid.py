from base import *

import copy
from scipy.fft import fft, fftfreq
import pickle

from Simulator_ABCD import *
from Simulator_P0 import *

@dataclass
class SimoptHybrid:
	""" Contains simulation options"""
	
	convergence_sim = SIMULATOR_P0 # This is the simulator used to find Iac for convergence
	result_sim = SIMULATOR_ABCD # This is the simulator used to print the result
	
@dataclass
class LKSolutionHybrid:
	""" Contains data to represent a solution to the LKsystem problem
	
	Nomenclature:
	_g: guess that led to this result
	_c: Condition - system setup value
	_td: Time domain data
	_w: Spectral data at select frequencies
	_wf: spectral data at all frequencies
	
	"""
	
	# Scalar/configuration variables
	Iac_g = None
	Ibias_c = None
	Zin = None # Impedance looking into chip (from source side)
	harms_c = None
	freq_c = None # Frequency of fundamnetal tone
	Vgen_c = None # Gwenerator voltage
	
	# Time domain variables
	Vp = None # Not actually used
	betaL_td = None # Electrical length of chip in radians
	theta = None # TODO: not saved
	L_td = None # Inductance per unit length
	Z0_td = None # Characteristic impedance of chip
	
	# Spectrum Data
	Ix_wf = None
	Ig_wf = None
	Ix_w = None
	Vx_w = None
	IL_w = None
	Ig_w = None # Iac result as spectrum, shows fundamental, 2harm, and 3harm as touple (idx 0 = fund, ..., 2 = 3rd harm)
	freq_w = None
	freq_wf = None
	
	# Convergence data
	convergence_failure = None # Set as True if fails to converge
	num_iter = None # Number of iterations completed
	Iac_guess_history = None # List of all Iac guesses
	guess_coef_history = None # List of all guess_coefficeints
	error_history = None # List of all error values during converge
	error_history_pcnt = None # List of all error values during converge

class LKSimHybrid:
	""" This class represents a solution to the nonlinear chip system, give a set of input conditions (things
	like actual chip length, input power, etc)."""
	
	# Name of simulator
	NAME = "Simulator_Hybrid"
	
	def __init__(self, master_sim, sim_abcd, sim_p0):
		""" Initialize system with given conditions """
		
		# Simulations options
		self.opt = SimoptHybrid()
		
		self.sim_abcd = sim_abcd
		self.sim_p0 = sim_p0

		# # System Settings
		# self.Pgen = master_sim.Pgen
		# self.C_ = master_sim.C_
		# self.l_phys = master_sim.l_phys
		# self.freq = master_sim.freq
		# self.q = master_sim.q
		# self.L0 = master_sim.L0
		# self.ZL = master_sim.ZL # Impedance of load
		# self.Zg = master_sim.Zg # Impedance of generator
		# self.Vgen = master_sim.Vgen # Solve for Generator voltage from power
		# self.max_harm = master_sim.max_harm # Harmonic number to go up to in spectral domain (plus DC)
		# self.system_loss = master_sim.system_loss # Tuple containing system loss at each harmonic (linear scale, not dB)
		# self.Itickle = master_sim.Itickle # Amplitude (A) of tickle signal (Set to none to exclude tickle)
		# self.freq_tickle = master_sim.freq_tickle # Amplitude (A) of tickle signal (Set to none to exclude tickle)
		# self.harms = master_sim.harms # List of harmonic numbers to include in spectral analysis
				
		# # Time domain options
		# self.num_periods = master_sim.num_periods
		# self.num_periods_tickle = master_sim.num_periods_tickle # If tickle is included, this may not be none, in which case this will set the t_max time (greatly extending simulation time)
		# self.max_harm_td = master_sim.max_harm_td
		# self.min_points_per_wave = master_sim.min_points_per_wave
		# self.t = master_sim.t

		# Create solution object
		self.soln = LKSolutionHybrid() # Current solution data
		
		self.solution = [] # List of solution data
		self.bias_points = [] # List of bias values corresponding to solution data
		
		# self.configure_time_domain(1000, 3, 30)
	
	def configure_tickle(self, Itickle:float, freq_tickle:float, num_periods:float=20):
		""" This function configures the tickle variables, enabling a tickle signal to be
		included in the simulation. Current is the amplitude in amps, and freq is in Hz."""
		
		self.Itickle = Itickle
		self.freq_tickle = freq_tickle
		self.num_periods_tickle = num_periods
		
		# # Reconfigure time domain
		# self.configure_time_domain(self.num_periods, self.max_harm_td, self.min_points_per_wave)
		
		logging.info(f"Configured tickle signal with f={cspecial}{len(rd(self.freq_tickle/1e3))}{standard_color} KHz and Iac={cspecial}{rd(Itickle*1e3)}{standard_color} mA.")
		
	def configure_time_domain(self, num_periods:float, max_harm_td:int, min_points_per_wave:int=10):
		""" Configures the time domain settings
		 
		  num_periods: Minimum number of periods to simulate
		  max_harm_td: Number of harmoinics to simulate
		  min_points_per_wave: Minimum number of time points per wavelength (at all frequencies)
		  
		"""
		
		pass
		
	def configure_loss(self, file:str=None, sparam_data:dict=None):
		""" Reads a pkl file with a dictionary containing variables 'freq_Hz' and 'S21_dB'
		and calculates the loss at each of the simulated harmonics. Can provide the dictionary
		without specifying a filename by using the sparam_data argument.
		"""
		
		pass
			
	def fourier(self, y:list, loss_frac:float=0, plot_result:bool=False):
		""" Takes the fourier transform of 'y and returns both the full spectrum, and
		the spectral components at the frequencies indicated by self.freq and self.harms. 
		
		y: variable to take FFT of
		loss_frac: fraction of S21_loss to apply to spectral components. (0 = apply no loss, 1 = apply full S21 loss)
		
		Returns all data as a tuple:
			(fullspec, fullspec_freqs, spectrum, spectrum_freqs)
		
		fullspec: Entire spectrum, no S21 loss applied ever
		fullspec_freqs: corresponding frequencies for fullspec
		spectrum: Spectrum as specified frequencies WITH S21 loss applied, per loss_frac
		spectrum_freqs: Frequencies at which spectrum is defined
		"""
		# Initialize X and Y variables
		num_pts = len(y)
		dt = self.t[1]-self.t[0]
		
		# Run FFT
		spec_raw = fft(y)[:num_pts//2]
		
		# Fix magnitude to compensate for number of points
		fullspec = 2.0/num_pts*np.abs(spec_raw)
		
		# Get corresponding x axis (frequencies)
		fullspec_freqs = fftfreq(num_pts, dt)[:num_pts//2]
		
		# Iterate over all harmonics
		spectrum = []
		spectrum_freqs = []
		DC_idx = 0
		for h_idx, h in enumerate(self.harms):
			
			# Find closest datapoint to target frequency
			target_freq = self.freq*h
			idx = find_nearest(fullspec_freqs, target_freq)
			freq_err = np.abs(fullspec_freqs[idx]-target_freq)

			# Send warning if target frequency missed by substatial margin
			if target_freq == 0:
				if freq_err > self.opt.freq_tol_Hz:
					logging.warning(f"Failed to find spectral data within absolute tolerance of target frequency. (Error = {freq_err/1e6} MHz, target = DC")
			elif freq_err/target_freq*100 > self.opt.freq_tol_pcnt:
				logging.warning(f"Failed to find spectral data within (%) tolerance of target frequency. (Error = {freq_err/target_freq*100} %, target = {target_freq/1e9} GHz")
			elif freq_err > self.opt.freq_tol_Hz:
				logging.warning(f"Failed to find spectral data within absolute tolerance of target frequency. (Error = {freq_err/1e6} MHz, target = {target_freq/1e9} GHz")
			
			# Find index of peak, checking adjacent datapoints as well
			try:
				Iac_hx = np.max([ fullspec[idx-1], fullspec[idx], fullspec[idx+1] ])
			except:
				if h != 0:
					logging.warning("Spectrum selected edge-element for fundamental")
				Iac_hx = fullspec[idx]
			
			# Apply system loss
			if (self.opt.use_S21_loss) and (self.system_loss is not None):
				logging.error("Need to apply system loss to power, not LK!!!")
				Iac_hx *= 1 - loss_frac*(1 - self.system_loss[h_idx])
		
			# Save to solution set
			spectrum.append(abs(Iac_hx))
			spectrum_freqs.append(fullspec_freqs[idx])
			
		# DC term is doubled, per matching TD with reconstructed signal.
		# TODO: Explain this!
		spectrum[0] /= 2
		
		# Plot spectrum if requested
		if plot_result:
			
			# Create spectrum figure
			plt.figure(2)
			plt.semilogy(np.array(fullspec_freqs)/1e9, np.array(fullspec)*1e3, color=(0, 0, 0.7))
			plt.scatter(np.array(spectrum_freqs)/1e9, np.array(spectrum)*1e3, color=(00.7, 0, 0))
			plt.xlabel("Frequency (GHz)")
			plt.ylabel("AC Current (mA)")
			plt.title(f"Intermediate Result: Spectrum of 'fourier()'")
			plt.xlim((0, self.freq*10/1e9))
			plt.grid()
			
			plt.show()
			
		# Return tuple of all data
		return (np.array(fullspec), np.array(fullspec_freqs), np.array(spectrum), np.array(spectrum_freqs))
	
	def crunch(self, Iac:float, Idc:float, show_plot_td=False, show_plot_spec=False):
		""" Using the provided Iac guess, find the reuslting solution, and the error
		between the solution and the initial guess.
		
		Converts the resulting Iac time domain data into spectral components, saving the
		fundamental through 3rd hamonic as a touple, with idx=0 assigned to fundamental.
		"""
				
		# Update Iac in solution
		self.soln.Iac_g = Iac
		self.soln.Ibias_c = Idc
		self.soln.harms_c = self.harms
		self.soln.freq_c = self.freq
		self.soln.Vgen_c = self.Vgen
		
		# Solve for inductance (Lk)
		# Calculate input current waveform
		if (self.Itickle is not None) and (self.freq_tickle is not None):
			Iin_td = Idc + Iac*np.sin(self.freq*2*PI*self.t) + self.Itickle*np.sin(self.freq_tickle*2*PI*self.t)
		else:
			Iin_td = Idc + Iac*np.sin(self.freq*2*PI*self.t)
	
		# Calculate Lk
		Lk = self.L0 + self.L0/(self.q**2) * (Iin_td**2)
		self.soln.L_td = Lk/self.l_phys
		
		# Find Z0 of chip
		self.soln.Z0_td = np.sqrt(self.soln.L_td/self.C_)
		
		# Find electrical length of chip (from phase velocity)
		self.soln.betaL_td = 2*PI*self.l_phys*self.freq*np.sqrt(self.C_ * self.soln.L_td)
		
		# Define ABCD method s.t. calculate current at VNA
		meas_frac = 1 #Fractional distance from gen towards load at which to meas. Vx and Ix
		thetaA_td = self.soln.betaL_td*meas_frac # = betaL
		thetaB_td = self.soln.betaL_td*(1-meas_frac) # = 0
		j = complex(0, 1)
		
		# Solve for IL (Eq. 33,4 in Notebook TAE-33)
		M = (self.ZL*np.cos(thetaB_td) + j*self.soln.Z0_td*np.sin(thetaB_td)) * ( np.cos(thetaA_td) + j*self.Zg/self.soln.Z0_td*np.sin(thetaA_td))
		N = ( self.ZL*j/self.soln.Z0_td*np.sin(thetaB_td) + np.cos(thetaB_td) ) * ( j*self.soln.Z0_td*np.sin(thetaA_td) + self.Zg*np.cos(thetaA_td) )
		
		IL_t = self.Vgen/(M+N)
		Vx_t = IL_t*self.ZL*np.cos(thetaB_td) + IL_t*j*self.soln.Z0_td*np.sin(thetaB_td)
		Ix_t = IL_t*self.ZL*j/self.soln.Z0_td*np.sin(thetaB_td) + IL_t*np.cos(thetaB_td)
		Ig_t = Vx_t*j/self.soln.Z0_td*np.sin(thetaA_td) + Ix_t*np.cos(thetaA_td)
		
		#----------------------- CALCULATE SPECTRAL COMPONENTS OF V and I --------------------
		
		IL_tuple = self.fourier(IL_t, loss_frac=1)
		IL = IL_tuple[2]
		
		Vx_tuple = self.fourier(Vx_t, loss_frac=meas_frac)
		Vx = Vx_tuple[2]
		
		Ix_tuple = self.fourier(Ix_t, loss_frac=meas_frac)
		Ix = Ix_tuple[2]
		
		Ig_tuple = self.fourier(Ig_t, loss_frac=0)
		Ig = Ig_tuple[2]
		
		# Save to result
		self.soln.Ig_w = abs(Ig)
		# self.soln.spec_Ig_check = abs(Ig_zc)
		self.soln.Ix_w = abs(Ix)
		self.soln.Vx_w = abs(Vx)
		self.soln.IL_w = abs(IL)
		
		self.soln.Ix_wf = abs(Ix_tuple[0])
		self.soln.Ig_wf = abs(Ig_tuple[0])
		self.soln.freq_w = Ix_tuple[3]
		self.soln.freq_wf = Ix_tuple[1]
		
	def plot_solution(self, s:LKSolutionHybrid=None):
		
		# Pick last solution if none provided
		if s is None:
			s = self.soln
		
		# calculate end index
		plot_Ts = 5
		idx_end = find_nearest(self.t, plot_Ts/self.freq)
		
		# Limit plot window
		f_max_plot = self.freq*np.max([10, np.max(self.harms)])
		idx_end = find_nearest(s.freq_wf, f_max_plot)
		
		# Plot Spectrum
		SP_R = 2
		SP_C = 3
		fig1 = plt.figure(1)
		plt.subplot(SP_R, SP_C, 1)
		plt.semilogy(s.freq_wf[:idx_end]/1e9, self.soln.Ig_wf[:idx_end], label="Full Spectrum", color=(0, 0, 0.8))
		plt.scatter(s.freq_w/1e9, self.soln.Ig_w, label="Selected Points", color=(0.8, 0, 0))
		plt.xlabel("Frequency (GHz)")
		plt.ylabel("Ig [A]")
		plt.title("Solution Spectrum")
		plt.grid()
		plt.legend()
		
		# Plot convergence history
		plt.subplot(SP_R, SP_C, 2)
		plt.semilogy(np.array(s.Iac_guess_history)*1e3, linestyle='dashed', marker='+', color=(0.4, 0, 0.6))
		plt.xlabel("Iteration")
		plt.ylabel("Iac Guess (mA)")
		plt.grid()
		plt.title("Guess History")
		
		plt.subplot(SP_R, SP_C, 3)
		plt.semilogy(s.guess_coef_history, linestyle='dashed', marker='+', color=(0, 0.4, 0.7))
		plt.xlabel("Iteration")
		plt.ylabel("Convergence Coefficient")
		plt.title("Coeff. History")
		plt.grid()
		
		plt.subplot(SP_R, SP_C, 4)
		plt.plot(np.array(s.error_history)*1e3, linestyle='dashed', marker='+', color=(0, 0.7, 0.4))
		plt.xlabel("Iteration")
		plt.ylabel("Error (mA)")
		plt.title("Error History")
		plt.grid()
		
		plt.subplot(SP_R, SP_C, 5)
		plt.plot(np.array(s.error_history_pcnt), linestyle='dashed', marker='+', color=(0.7, 0.0, 0.7))
		plt.xlabel("Iteration")
		plt.ylabel("Error (%)")
		plt.title("Error History")
		plt.grid()
		
		# fig1.set_size_inches((14, 3))
		
		plt.show()
	
	def solve(self, Ibias_vals:list, show_plot_on_conv=False, show_plot_on_fail=False):
		""" Takes a list of bias values, plugs them in for Idc, and solves for
		the AC current s.t. error is within tolerance. """
		
		logging.info(f"Beginning iterative solve for {cspecial}{len(Ibias_vals)}{standard_color} bias points.")
		
		# Get convergence simulator
		if self.opt.convergence_sim == SIMULATOR_ABCD:
			conv_sim = self.sim_abcd
		elif self.opt.convergence_sim == SIMULATOR_P0:
			conv_sim = self.sim_p0
		
		# Get result simulator
		if self.opt.result_sim == SIMULATOR_ABCD:
			res_sim = self.sim_abcd
		elif self.opt.result_sim == SIMULATOR_P0:
			res_sim = self.sim_p0
			
		# Solve for this bias point using the convergence simulator
		conv_sim.solve(Ibias_vals, show_plot_on_conv=show_plot_on_conv, show_plot_on_fail=show_plot_on_fail)
		
		# Get solution data
		soln = conv_sim.get_solution()
		Iac_g = soln_extract(soln, "Iac_g", conv_only=True)
		Ibias_c = soln_extract(soln, "Ibias_c", conv_only=True)
		logging.info(f"Found {Fore.CYAN}{len(Iac_g)}{Fore.WHITE} converged values ranging from {Fore.CYAN}{rd(min(Iac_g*1e3))}{Fore.WHITE} mA to {Fore.CYAN}{rd(max(Iac_g*1e3))}{Fore.WHITE} mA.")
		
		# Run solution data through results simulator
		
		logging.info(f"Beginning results calculation with simulator: {Fore.LIGHTBLUE_EX}{simcode_to_str(self.opt.result_sim)}{Style.RESET_ALL}")
		Iac_r = []
		for idx, Iac in enumerate(Iac_g):
			
			# Crunch the numbers in the alternative simulator
			res_sim.crunch(Iac, Ibias_c[idx])
			Iac_r.append(res_sim.soln.Ig_w[1])
		
			# Add to result-simulator's solution
			new_soln = copy.deepcopy(res_sim.soln)
			new_soln.convergence_failure = False
			
			# Add solution to list
			res_sim.solution.append(new_soln)
			
			# Add bias point to list
			res_sim.bias_points.append(Ibias_c[idx])
		
		
		
	
	def get_solution(self):
		""" Returns the solution in generalized format """
		
		# Iterate over solution data, copy into new format
		fmt_solution = []
		for s in self.solution:
			
			# Create solution object
			ns = LKSolution()
			
			# Populate with solution data
			ns.Iac_g = s.Iac_g
			ns.Ibias_c = s.Ibias_c
			ns.Vgen_c = s.Vgen_c
			
			ns.Ig_w = s.Ig_w
			ns.Ig_wf = s.Ig_wf
			
			ns.IL_w = s.IL_w
			ns.IL_wf = []
			
			ns.VL_w = []
			ns.VL_wf = []
			
			ns.freq_w = s.freq_w
			ns.freq_wf = s.freq_wf
			
			ns.convergence_failure = s.convergence_failure
			ns.num_iter = s.num_iter
			
			ns.Iac_guess_history = s.Iac_guess_history
			ns.guess_coef_history = s.guess_coef_history
			ns.error_history = s.error_history
			ns.error_history_pcnt = s.error_history_pcnt
			
			# Append to output list
			fmt_solution.append(ns)
		
		return fmt_solution