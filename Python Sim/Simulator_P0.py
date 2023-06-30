from base import *

import copy
from scipy.fft import fft, fftfreq
import pickle

SIMULATOR_P0 = 20

@dataclass
class SimoptP0:
	""" Contains simulation options"""
	
	# Simulation options
	use_interp = False # This option allows FFT interpolation, but ***has NOT been implemented***
	use_S21_loss = True # This option interprets S21 data to determine system loss and incorporates it in the converging simulation
	use_Lk_expansion = False # This option calculates Lk from the standard I^2 approx as opposed to including all terms. Note: this option must be FALSE to include tickle. Only included because it was used in the original simulations
	
	# Convergence options
	max_iter = 1000 # Max iterations for convergence
	tol_pcnt = 1 # Tolerance in percent between Iac guesses
	guess_update_coef = 0.5 # Fraction by which to compromise between guess and result Iac (0=remain at guess, 1=use result; 0.5 recommended)
	ceof_shrink_factor = 0.2 # Fraction by which to modify guess_update_coef when sign reverses (good starting point: 0.2)
	
	# How to pick initial Iac guess
	start_guess_method = GUESS_ZERO_REFLECTION
	
	# Data Save Options
	remove_td = False # Prevents all time domain data from being saved in solution data to save space

@dataclass
class LKSolutionP0:
	""" Contains data to represent a solution to the LKsystem problem"""
	
	Lk = None
	
	Vp = None # Not actually used
	betaL = None # Electrical length of chip in radians
	P0 = None
	theta = None # TODO: not saved
	Iac = None
	Ibias = None
	Zin = None # Impedance looking into chip (from source side)
	
	L_ = None # From Lk
	Zchip = None # From L_, characteristic impedance of chip
	
	Iac_result_rms = None # Magnitude of Iac
	Iac_result_td = None # Use Iac to find solution and calculate Iac again, this is the result in time domain
	Iac_result_spec = None # Iac result as spectrum, shows fundamental, 2harm, and 3harm as touple (idx 0 = fund, ..., 2 = 3rd harm)
	rmse = None # |Iac_result - Iac|
	
	convergence_failure = None # Set as True if fails to converge
	num_iter = None # Number of iterations completed
	
	spec = None # Spectrum data [AC Current amplitude in Amps]
	spec_freqs = None # Spectrum frequencies in Hz

class LKSimP0:
	""" This class represents a solution to the nonlinear chip system, give a set of input conditions (things
	like actual chip length, input power, etc)."""
	
	# Name of simulator
	NAME = "Simulator_P0"
	
	def __init__(self, master_sim):
		""" Initialize system with given conditions """
		
		# Simulations options
		self.opt = SimoptP0()

		# System Settings
		self.Pgen = master_sim.Pgen
		self.C_ = master_sim.C_
		self.l_phys = master_sim.l_phys
		self.freq = master_sim.freq
		self.q = master_sim.q
		self.L0 = master_sim.L0
		self.Zcable = 50 # Z0 of cable leading into chip
		self.Rsrc = np.real(master_sim.Zg) # R of generator
		self.Xsrc = np.imag(master_sim.Zg) # X of generator
		self.Vgen  = np.sqrt(self.Pgen*200) # Solve for Generator voltage from power
		self.system_loss = master_sim.system_loss # Tuple containing system loss at each harmonic (linear scale, not dB)
		self.Itickle = master_sim.Itickle # Amplitude (A) of tickle signal (Set to none to exclude tickle)
		self.freq_tickle = master_sim.freq_tickle # Amplitude (A) of tickle signal (Set to none to exclude tickle)
		
		# Time domain options
		self.num_periods = master_sim.num_periods
		self.num_periods_tickle = master_sim.num_periods_tickle # If tickle is included, this may not be none, in which case this will set the t_max time (greatly extending simulation time)
		self.max_harm = master_sim.max_harm_td
		self.min_points_per_wave = master_sim.min_points_per_wave
		self.t = master_sim.t

		# Create solution object
		self.soln = LKSolutionP0() # Current solution data
		
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
		# self.configure_time_domain(self.num_periods, self.max_harm, self.min_points_per_wave)
		
		logging.info(f"Configured tickle signal with f={cspecial}{len(rd(self.freq_tickle/1e3))}{standard_color} KHz and Iac={cspecial}{rd(Itickle*1e3)}{standard_color} mA.")
		
	def configure_time_domain(self, num_periods:float, max_harm:int, min_points_per_wave:int=10):
		""" Configures the time domain settings
		 
		  num_periods: Minimum number of periods to simulate
		  max_harm: Number of harmoinics to simulate
		  min_points_per_wave: Minimum number of time points per wavelength (at all frequencies)
		  
		"""
		
		# Save options
		self.num_periods = num_periods
		self.max_harm = max_harm
		self.min_points_per_wave = min_points_per_wave
		
		# Calculate max 
		if self.num_periods_tickle is None: # Calculate t_max and num_points the standard way
			t_max = num_periods/self.freq
			num_points = min_points_per_wave*max_harm*num_periods+1
		else: # Greatly extend number of points to simulate low-freq tickle accurately
			t_max = self.num_periods_tickle/self.freq_tickle
			num_points = min_points_per_wave*int(max_harm*self.freq/self.freq_tickle)*num_periods+1
		
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
		
		# Find indecies
		f_idx1 = find_nearest(freq, self.freq)
		f_idx2 = find_nearest(freq, self.freq*2)
		f_idx3 = find_nearest(freq, self.freq*3)
		
		#TODO: add more error checking if data doesn't contain correct or close enoguh frequencies
		
		# Convert from dB to linear scaling - dB20 because this will be applied to Iac, not P0
		loss_Fund = (10**(S21[f_idx1]/20))
		loss_2H = (10**(S21[f_idx2]/20))
		loss_3H = (10**(S21[f_idx3]/20))
		
		# Save loss
		self.system_loss = (loss_Fund, loss_2H, loss_3H)
		
		logging.main(f"Configured system loss: {cspecial}[lf={rd(loss_Fund, 2)}, l2H={rd(loss_2H, 2)}, l3H={rd(loss_3H, 2)}]{standard_color}")
		
	def crunch(self, Iac:float, Idc:float, show_plot_td=False, show_plot_spec=False):
		""" Using the provided Iac guess, find the reuslting solution, and the error
		between the solution and the initial guess.
		
		Converts the resulting Iac time domain data into spectral components, saving the
		fundamental through 3rd hamonic as a touple, with idx=0 assigned to fundamental.
		"""
		
		# Update Iac in solution
		self.soln.Iac = Iac
		self.soln.Ibias = Idc
		
		# Solve for inductance (Lk)
		if self.opt.use_Lk_expansion: 
			# Use expansion for I=Idc+Iac*sin
			
			self.soln.Lk = self.L0 + self.L0/self.q**2 * ( Idc**2 + 2*Idc*Iac*np.sin(self.freq*2*PI*self.t) + Iac**2/2 - Iac**2/2*np.cos(2*self.freq*2*PI*self.t) ) 
		else:
			
			# Calculate input current waveform
			if (self.Itickle is not None) and (self.freq_tickle is not None):
				Iin = Idc + Iac*np.sin(self.freq*2*PI*self.t) + self.Itickle*np.sin(self.freq_tickle*2*PI*self.t)
			else:
				Iin = Idc + Iac*np.sin(self.freq*2*PI*self.t)
		
			# Calculate Lk
			self.soln.Lk = self.L0 + self.L0/self.q**2 * Iin**2 
			
		self.soln.L_ = self.soln.Lk/self.l_phys
		
		# # Update inductance estimate
		# self.solve_inductance(Iac, Idc)
		
		# Find Z0 of chip
		self.soln.Vp = 1/np.sqrt(self.C_ * self.soln.L_)
		self.soln.Zchip = np.sqrt(self.soln.L_ / self.C_)
		
		# Find electrical length of chip (from phase velocity)
		self.soln.betaL = 2*PI*self.l_phys*self.freq*np.sqrt(self.C_ * self.soln.L_)
		
		# Find input impedance to chip
		self.soln.Zin = xfmr(self.soln.Zchip, self.Zcable, self.soln.betaL)
		
		# Get real and imag components of Zin
		Rin = np.real(self.soln.Zin)
		Xin = np.imag(self.soln.Zin)
		
		# Find angle of Zin
		self.theta = np.angle(self.soln.Zin)
		
		# Calcualte transmitted power (to chip)
		self.soln.P0 = np.abs(self.Vgen)**2 * Rin * 0.5 / ((Rin + self.Rsrc)**2 + (Xin + self.Xsrc)**2)
		
		# Find resulting Iac and error
		logging.warning(f"{Fore.RED}ERROR: Power calculation ignored higher order terms.{standard_color}")
		self.soln.Iac_result_rms = np.sqrt(2*self.soln.P0/np.cos(self.theta)/self.soln.Zin)
		self.soln.Iac_result_td = np.sqrt(2)*self.soln.Iac_result_rms*np.sin(2*PI*self.freq*self.t) #TODO: Need a factor of sqrt(2)
		# err_list = np.abs(self.soln.Iac_result_rms - self.soln.Iac) # Error in signal *amplitude* at each time point
		# self.soln.rmse = np.sqrt(np.mean(err_list**2))
		
		# # Save to logger
		# logging.debug(f"Solution error: {round(self.soln.rmse*1000)/1000}")
		#TODO: This error is never used - Populate self.soln.rmse correctly!
		
		if show_plot_td:
			plot_Ts = 5
			idx_end = find_nearest(self.t, plot_Ts/self.freq)
			
			plt.plot(self.t*1e9, self.soln.Iac_result_td*1e3)
			plt.xlabel("Time (ns)")
			plt.ylabel("AC Current (mA)")
			plt.title("Solution Iteration Time Domain Plot")
			plt.show()
			
			# plt.plot(self.t[:idx_end]*1e9, self.soln.Lk[:idx_end]*1e9)
			# plt.xlabel("Time (ns)")
			# plt.ylabel("Lk (nH)")
			# plt.title("Solution Iteration Time Domain Plot")
			# plt.grid()
			# plt.show()
		
		############# Previously called get_spec_components() ###########
		
		y = self.soln.Iac_result_td
		num_pts = len(self.soln.Iac_result_td)
		dt = self.t[1]-self.t[0]
		
		# Run FFT
		spec_raw = fft(y)[:num_pts//2]
		
		# Fix magnitude to compensate for number of points
		spec = 2.0/num_pts*np.abs(spec_raw)
		self.soln.spec = spec
		
		# Get corresponding x axis (frequencies)
		spec_freqs = fftfreq(num_pts, dt)[:num_pts//2]
		self.soln.spec_freqs = spec_freqs
		
		if self.opt.use_interp:
			logging.warning("interp feature has not been implemented. See fft_demo.py to see how.")
		
		# Find index of peak - Fundamental
		idx = find_nearest(spec_freqs, self.freq)
		try:
			Iac_fund = np.max([ spec[idx-1], spec[idx], spec[idx+1] ])
		except:
			logging.warning("Spectrum selected edge-element for fundamental")
			Iac_fund = spec[idx]
			
		# Find index of peak - Fundamental
		idx = find_nearest(spec_freqs, self.freq*2)
		try:
			Iac_2H = np.max([ spec[idx-1], spec[idx], spec[idx+1] ])
		except:
			logging.warning("Spectrum selected edge-element for 2nd Harmonic")
			Iac_2H = spec[idx]
			
		# Find index of peak - Fundamental
		idx = find_nearest(spec_freqs, self.freq*3)
		try:
			Iac_3H = np.max([ spec[idx-1], spec[idx], spec[idx+1] ])
		except:
			logging.warning("Spectrum selected edge-element for 3rd Harmonic")
			Iac_3H = spec[idx]
			show_plot_spec = True
			
		# Apply system loss
		if self.opt.use_S21_loss and self.system_loss is not None:
			Iac_fund *= self.system_loss[0]
			Iac_2H *= self.system_loss[1]
			Iac_3H *= self.system_loss[2]
		
		# Save to solution set
		self.soln.Iac_result_spec = (Iac_fund, Iac_2H, Iac_3H)
		
		logging.debug(f"Calcualted Iac spectral components: fund={rd(Iac_fund*1e6, 3)}, 2H={rd(Iac_2H*1e6, 3)}, 3H={rd(Iac_3H*1e6, 3)} uA")
		
		if show_plot_spec:
			
			plt.plot(spec_freqs/1e9, spec*1e3)
			plt.xlabel("Frequency (GHz)")
			plt.ylabel("AC Current (mA)")
			plt.title("Solution Iteration Spectrum")
			
			plt.show()
		
	def plot_solution(self, s:LKSolutionP0=None):
		
		# Pick last solution if none provided
		if s is None:
			s = self.soln
		
		# calculate end index
		plot_Ts = 5
		idx_end = find_nearest(self.t, plot_Ts/self.freq)
		
		# Create time domain figure
		plt.figure(1)
		plt.plot(self.t[:idx_end]*1e9, np.real(s.Iac_result_td[:idx_end])*1e3, '-b')
		plt.plot(self.t[:idx_end]*1e9, np.abs(s.Iac_result_td[:idx_end])*1e3, '-r')
		plt.plot(self.t[:idx_end]*1e9, np.sqrt(2)*np.abs(s.Iac_result_rms[:idx_end])*1e3, '-g')
		plt.xlabel("Time (ns)")
		plt.ylabel("AC Current (mA)")
		plt.title(f"Time Domain Data, Idc = {rd(s.Ibias*1e3)} mA")
		plt.legend(["TD Real", "TD Abs.", "|Amplitude|"])
		plt.grid()
		
		# Create spectrum figure
		plt.figure(2)
		plt.semilogy(s.spec_freqs/1e9, s.spec*1e3)
		plt.xlabel("Frequency (GHz)")
		plt.ylabel("AC Current (mA)")
		plt.title(f"Current Spectrum, Idc = {rd(s.Ibias*1e3)} mA")
		plt.xlim((0, self.freq*5/1e9))
		plt.grid()
		
		plt.show()
	
	def solve(self, Ibias_vals:list, show_plot_on_conv=False):
		""" Takes a list of bias values, plugs them in for Idc, and solves for
		the AC current s.t. error is within tolerance. """
		
		logging.info(f"Beginning iterative solve for {cspecial}{len(Ibias_vals)}{standard_color} bias points.")
		
		Iac_crude_guess = np.abs(self.Vgen)**2 / 50 # Crude guess for AC current (Assume 50 ohm looking into chip)
		
		# Scan over each bias value
		for Idc in Ibias_vals:
			
			last_sign = None # Sign of last change
			self.soln.num_iter = 0	# Reset iteration counter
			guess_coef = self.opt.guess_update_coef # Reset guess_coef
			
			if self.opt.start_guess_method == GUESS_ZERO_REFLECTION:
				Iac_guess = Iac_crude_guess
			elif self.opt.start_guess_method == GUESS_USE_LAST:
				if len(self.solution) > 0:
					Iac_guess = self.solution[-1].Iac
				else:
					Iac_guess = Iac_crude_guess
			
			# Loop until converge
			while True:
				
				# Crunch the numbers of this guess value
				self.crunch(Iac_guess, Idc)
				self.soln.num_iter += 1
				
				# Calculate signed error
				error = self.soln.Iac_result_spec[0] - Iac_guess
				error_pcnt = (np.max([self.soln.Iac_result_spec[0], Iac_guess])/np.min([self.soln.Iac_result_spec[0], Iac_guess])-1)*100
				
				# Check for convergence
				if error_pcnt < self.opt.tol_pcnt:
					
					# Add to logger
					logging.info(f"Datapoint ({cspecial}Idc={rd(Idc*1e3)} mA{standard_color}),({cspecial}Iac={rd(Iac_guess*1e3, 3)} mA{standard_color}) converged with {cspecial}error={rd(error_pcnt, 3)}%{standard_color} after {cspecial}{self.soln.num_iter}{standard_color} iterations ")
					
					# Create deep
					new_soln = copy.deepcopy(self.soln)
					new_soln.convergence_failure = False
					
					if self.opt.remove_td:
						new_soln.Lk = []
						new_soln.Vp = []
						new_soln.betaL = []
						new_soln.Zchip = []
						new_soln.L_ = []
						new_soln.P0 = []
						new_soln.theta = []
						new_soln.Zin = []
						new_soln.Iac_result_rms = []
						new_soln.Iac_result_td = []
					
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
				elif self.soln.num_iter >= self.opt.max_iter:
					
					# Add to logger
					logging.warning(f"Failed to converge for point ({cspecial}Idc={rd(Idc*1e3)} mA{standard_color}).")
					
					# Exit convergence loop
					break
				
				# Else update guess
				else:
					
					# Error is positive
					if error > 0:
						
						# First iteration - set sign
						if last_sign is None:
							last_sign = np.sign(error)
						# Last change was in different direction
						elif last_sign < 1:
							guess_coef *= self.opt.ceof_shrink_factor # Change update size
							last_sign = -1 # Update change direction
							logging.debug(f"Error sign changed. Changing guess update coefficient to {cspecial}{guess_coef}{standard_color}")
						
						# Update guess
						Iac_guess = Iac_guess + error * guess_coef
					
					# Else error is negative
					else:
						
						# First iteration - set sign
						if last_sign is None:
							last_sign = np.sign(error)
						# Last change was in different direction
						elif last_sign > 1:
							guess_coef *= self.opt.ceof_shrink_factor # Change update size
							last_sign = 1 # Update change direction
							logging.debug(f"Error sign changed. Changing guess update coefficient to {cspecial}{guess_coef}{standard_color}")
							
						# Update guess
						Iac_guess = Iac_guess + error * guess_coef
					
					logging.debug(f"Last guess produced {cspecial}error={error}{standard_color}. Updated guess to {cspecial}Iac={Iac_guess}{standard_color}. [{cspecial}iter={self.soln.num_iter}{standard_color}]")
	
	def get_solution(self):
		""" Returns the solution in generalized format """
		
		# Iterate over solution data, copy into new format
		fmt_solution = []
		for s in self.solution:
			
			# Create solution object
			ns = LKSolution()
			
			# Populate with solution data
			ns.Iac_g = s.Iac
			ns.Ibias_c = s.Ibias
			ns.Vgen_c = self.Vgen
			
			ns.Ig_w = s.Iac_result_spec
			ns.Ig_wf = []
			
			ns.IL_w = []
			ns.IL_wf = []
			
			ns.VL_w = []
			ns.VL_wf = []
			
			ns.freq_w = s.spec_freqs
			ns.freq_wf = []
			
			ns.convergence_failure = s.convergence_failure
			ns.num_iter = s.num_iter
			
			ns.Iac_guess_history = []
			ns.guess_coef_history = []
			ns.error_history = []
			ns.error_history_pcnt = []
			
			# Append to output list
			fmt_solution.append(ns)
		
		return fmt_solution