from base import *

from dataclasses import dataclass
import copy
from scipy.fft import fft, fftfreq
import pickle

@dataclass
class SimoptABCD:
	""" Contains simulation options"""
	
	# Simulation options
	use_S21_loss = True # This option interprets S21 data to determine system loss and incorporates it in the converging simulation
	
	# Frequency tolerance flags
	#   These apply to finding frequencies for the loss estimate, FFT, etc. If
	#   these tolerances are exceeded, warnings are sent to the log.
	freq_tol_pcnt = 5 # If the estimate can't match the target freq. within this tolerance, it will send an error. Does not apply to DC
	freq_tol_Hz = 100e6 # Same as above, but as absolute vale and DOES apply to DC
	
	# Convergence options
	max_iter = 1000 # Max iterations for convergence
	tol_pcnt = 1 # Tolerance in percent between Iac guesses
	tol_abs = 0.1e-3 # Tolerance in mA between Iac guesses
	guess_update_coef = 0.1 # Fraction by which to compromise between guess and result Iac (0=remain at guess, 1=use result; 0.5 recommended)
	ceof_shrink_factor = 1 # Fraction by which to modify guess_update_coef when sign reverses (good starting point: 0.2)
	
	# How to pick initial Iac guess
	start_guess_method = GUESS_ZERO_REFLECTION
	
	# Data Save Options
	remove_td = False # Prevents all time domain data from being saved in solution data to save space
	
@dataclass
class LKSolutionABCD:
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
	spec_Ig_check = None
	freq_w = None
	freq_wf = None
	rmse = None # |Iac_result - Iac|
	
	# Convergence data
	convergence_failure = None # Set as True if fails to converge
	num_iter = None # Number of iterations completed
	Iac_guess_history = None # List of all Iac guesses
	guess_coef_history = None # List of all guess_coefficeints
	error_history = None # List of all error values during converge
	error_history_pcnt = None # List of all error values during converge

class LKSimABCD:
	""" This class represents a solution to the nonlinear chip system, give a set of input conditions (things
	like actual chip length, input power, etc)."""
	
	def __init__(self, Pgen_dBm:float, C_:float, l_phys:float, freq:float, q:float, L0:float, max_harm:int=6, ZL=50, Zg=50):
		""" Initialize system with given conditions """
		
		# Simulations options
		self.opt = SimoptABCD()

		# System Settings
		self.Pgen = (10**(Pgen_dBm/10))/1000
		self.C_ = C_
		self.l_phys = l_phys
		self.freq = freq
		self.q = q
		self.L0 = L0
		self.Zcable = 50 # Z0 of cable leading into chip
		self.ZL = ZL # Impedance of load
		self.Zg = Zg # Impedance of generator
		self.Vgen  = np.sqrt(self.Pgen*200) # Solve for Generator voltage from power
		self.max_harm = max_harm # Harmonic number to go up to in spectral domain (plus DC)
		self.system_loss = None # Tuple containing system loss at each harmonic (linear scale, not dB)
		self.Itickle = None # Amplitude (A) of tickle signal (Set to none to exclude tickle)
		self.freq_tickle = None # Amplitude (A) of tickle signal (Set to none to exclude tickle)
		self.harms = np.array(range(self.max_harm+1)) # List of harmonic numbers to include in spectral analysis
				
		# Time domain options
		self.num_periods = None
		self.num_periods_tickle = None # If tickle is included, this may not be none, in which case this will set the t_max time (greatly extending simulation time)
		self.max_harm_td = None
		self.min_points_per_wave = None
		self.t = []

		# Create solution object
		self.soln = LKSolutionABCD() # Current solution data
		
		self.solution = [] # List of solution data
		self.bias_points = [] # List of bias values corresponding to solution data
		
		self.configure_time_domain(1000, 3, 30)
	
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
		
		# Save options
		self.num_periods = num_periods
		self.max_harm_td = max_harm_td
		self.min_points_per_wave = min_points_per_wave
		
		# Calculate max 
		if self.num_periods_tickle is None: # Calculate t_max and num_points the standard way
			t_max = num_periods/self.freq
			num_points = min_points_per_wave*max_harm_td*num_periods+1
		else: # Greatly extend number of points to simulate low-freq tickle accurately
			t_max = self.num_periods_tickle/self.freq_tickle
			num_points = min_points_per_wave*int(max_harm_td*self.freq/self.freq_tickle)*num_periods+1
		
		# Create t array
		self.t = np.linspace(0, t_max, num_points)
		
		logging.info(f"Configured time domain with {cspecial}{len(self.t)}{standard_color} points.")
		
	def configure_loss(self, file:str=None, sparam_data:dict=None):
		""" Reads a pkl file with a dictionary containing variables 'freq_Hz' and 'S21_dB'
		and calculates the loss at each of the simulated harmonics. Can provide the dictionary
		without specifying a filename by using the sparam_data argument.
		"""
		
		# Read file if no data provided
		if sparam_data is None:
			
			# Open File
			with open(file, 'rb') as fh:
				sparam_data = pickle.load(fh)
		
		# Access frequency and S21 data
		try:
			freq = sparam_data['freq_Hz']
			S21 = sparam_data['S21_dB']
		except:
			logging.error(f"{Fore.RED}Invalid S-parameter data provided when configuring system loss!{Style.RESET_ALL}")
			logging.main("Simulating wihtout system loss")
			return
		
		# Scan over all included harmonics and find loss
		self.system_loss = []
		for h in self.harms:
			
			target_freq = self.freq*h
			
			f_idx = find_nearest(freq, target_freq) # Find index
			
			# Send warning if target frequency missed by substatial margin
			freq_err = np.abs(freq[f_idx]-target_freq)
			if target_freq == 0:
				if freq_err > self.opt.freq_tol_Hz:
					logging.warning(f"Failed to find loss data within absolute tolerance of target frequency. (Error = {freq_err/1e6} MHz, target = DC")
			elif freq_err/target_freq*100 > self.opt.freq_tol_pcnt:
				logging.warning(f"Failed to find loss data within (%) tolerance of target frequency. (Error = {freq_err/target_freq*100} %, target = {target_freq/1e9} GHz")
			elif freq_err > self.opt.freq_tol_Hz:
				logging.warning(f"Failed to find loss data within absolute tolerance of target frequency. (Error = {freq_err/1e6} MHz, target = {target_freq/1e9} GHz")
			
			loss = (10**(S21[f_idx]/20)) # Calculate loss (convert from dB)
			self.system_loss.append(loss) # Add to list
		
		logging.main(f"Configured system loss.")
			
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
		
	def plot_solution(self, s:LKSolutionABCD=None):
				
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
		
		Iac_crude_guess = np.abs(self.Vgen) / 50 # Crude guess for AC current (Assume 50 ohm looking into chip)
		
		# Scan over each bias value
		for Idc in Ibias_vals:
			
			# Reset solution data
			last_sign = None # Sign of last change
			self.soln.num_iter = 0	# Reset iteration counter
			self.soln.Iac_guess_history = []
			self.soln.guess_coef_history = []
			self.soln.error_history = []
			self.soln.error_history_pcnt = []
			guess_coef = self.opt.guess_update_coef # Reset guess_coef
			
			# Prepare initial guess
			if self.opt.start_guess_method == GUESS_ZERO_REFLECTION:
				Iac_guess = Iac_crude_guess
			elif self.opt.start_guess_method == GUESS_USE_LAST:
				if len(self.solution) > 0:
					Iac_guess = self.solution[-1].Iac
				else:
					Iac_guess = Iac_crude_guess
				
			# Loop until converge
			while True:
				
				#Add to solution history
				self.soln.guess_coef_history.append(guess_coef)
				self.soln.Iac_guess_history.append(Iac_guess)
				
				# Crunch the numbers of this guess value
				self.crunch(Iac_guess, Idc)
				self.soln.num_iter += 1
				
				# Calculate signed error, check if convergence conditions met
				# error = self.soln.Ig_w[0] + self.soln.Ig_w[1] + self.soln.Ig_w[2] - Iac_guess
				error = self.soln.Ig_w[1] - Iac_guess
				denom = np.min([self.soln.Ig_w[1], Iac_guess])
				if denom != 0:
					error_pcnt = (np.max([self.soln.Ig_w[1], Iac_guess])/denom-1)*100
					did_converge = (error_pcnt < self.opt.tol_pcnt) and ( abs(error) < self.opt.tol_abs )
				else:
					error_pcnt = None
					did_converge = ( abs(error) < self.opt.tol_abs )
				
				# Add to history
				self.soln.error_history.append(error)
				self.soln.error_history_pcnt.append(error_pcnt)
				
				# Check for convergence
				if did_converge: #------------- Solution has converged ---------------------------------
					
					# Add to logger
					logging.info(f"Datapoint ({cspecial}Idc={rd(Idc*1e3)} mA{standard_color}),({cspecial}Iac={rd(Iac_guess*1e3, 3)} mA{standard_color}) converged with {cspecial}error={rd(error_pcnt, 3)}%{standard_color} after {cspecial}{self.soln.num_iter}{standard_color} iterations ")
					
					# Create deep
					new_soln = copy.deepcopy(self.soln)
					new_soln.convergence_failure = False
					
					if self.opt.remove_td:
						new_soln.Lk = []
						new_soln.Vp = []
						new_soln.betaL = []
						new_solnZchip_td = []
						new_soln.L_td = []
						new_soln.P0 = []
						new_soln.theta = []
						new_soln.Zin = []
						
					if show_plot_on_conv:
						
						label_color = Fore.LIGHTBLUE_EX
						if new_soln.Iac_g < 1e-3:
							label_color = Fore.RED
						
						# print(f"{label_color}Solution:{Style.RESET_ALL}")
						# print(f"{label_color}\tharms:{Style.RESET_ALL} {rdl(new_soln.harms)}")
						# print(f"{label_color}\tZ0 (ohm)):{Style.RESET_ALL} {rdl(new_soln.spec_Z0)}")
						# print(f"{label_color}\tsqL (ohms):{Style.RESET_ALL} {rdl(new_soln.spec_sqL)}")
						# print(f"{label_color}\tbetaL (deg):{Style.RESET_ALL} {rdl(new_soln.spec_betaL)}")
						# print(f"{label_color}\tIL (mA):{Style.RESET_ALL} {rdl(new_soln.IL_w*1e3)}")
						# print(f"{label_color}\tIx (mA):{Style.RESET_ALL} {rdl(new_soln.Ix_w*1e3)}")
						# print(f"{label_color}\tIg (mA):{Style.RESET_ALL} {rdl(new_soln.Ig_w*1e3)}")
						
						# if new_soln.Iac < 1e-3:
						# 	self.plot_solution(new_soln)
					
					# Add solution to list
					self.solution.append(new_soln)
					
					# Add bias point to list
					self.bias_points.append(Idc)
					
					# Plot result if requested
					if show_plot_on_conv:
						self.plot_solution()
					
					# Exit convergence loop
					break
					
				# Check for exceed max iterations
				elif self.soln.num_iter >= self.opt.max_iter:  #-------- Not converged, has exceeded hax iterations! ---------
					
					# Add to logger
					logging.warning(f"Failed to converge for point ({cspecial}Idc={rd(Idc*1e3)} mA{standard_color}).")
					
					# Create deep copy
					new_soln = copy.deepcopy(self.soln)
					new_soln.convergence_failure = True
					
					# Purge time domain data if requested
					if self.opt.remove_td:
						new_soln.Lk = []
						new_soln.Vp = []
						new_soln.betaL = []
						new_solnZchip_td = []
						new_soln.L_td = []
						new_soln.P0 = []
						new_soln.theta = []
						new_soln.Zin = []
						# new_soln.Iac_result_rms = []
						# new_soln.Iac_result_td = []
					
					# Add solution to list
					self.solution.append(new_soln)
					
					# Print convergence data if requested
					if show_plot_on_fail:
						self.plot_solution(new_soln)
					
					# Exit convergence loop
					break
				
				# Else update guess
				else: #---------------------- Not converged, calcualte a new guess ------------------------------
					
					# Eror is positive, both agree
					if error > 0:
						
						# First iteration - set sign
						if last_sign is None:
							last_sign = np.sign(error)
						# Last change was in different direction
						elif last_sign < 0:
							guess_coef *= self.opt.ceof_shrink_factor # Change update size
							logging.info(f"iter: {self.soln.num_iter} Changing guess coef from {rd(guess_coef*1e5)}e-5 to {rd(guess_coef*1e5)}e-5. Shrink factor: {rd(self.opt.ceof_shrink_factor)}")
							last_sign = 1 # Update change direction
							logging.debug(f"Error sign changed. Changing guess update coefficient to {cspecial}{guess_coef}{standard_color}")
						
						# Update guess
						Iac_guess = Iac_guess + error * guess_coef
					
					# Error is negative, both agree
					elif error < 0:
												
						# First iteration - set sign
						if last_sign is None:
							last_sign = np.sign(error)
						# Last change was in different direction
						elif last_sign > 0:
							guess_coef *= self.opt.ceof_shrink_factor # Change update size
							logging.info(f"iter: {self.soln.num_iter} Changing guess coef from {rd(guess_coef*1e5)}e-5 to {rd(guess_coef*1e5)}e-5. Shrink factor: {rd(self.opt.ceof_shrink_factor)}")
							last_sign = -1 # Update change direction
							logging.debug(f"Error sign changed. Changing guess update coefficient to {cspecial}{guess_coef}{standard_color}")
							
						# Update guess
						Iac_guess = Iac_guess + error * guess_coef
					
					logging.debug(f"Last guess produced {cspecial}error={error}{standard_color}. Updated guess to {cspecial}Iac={Iac_guess}{standard_color}. [{cspecial}iter={self.soln.num_iter}{standard_color}]")
