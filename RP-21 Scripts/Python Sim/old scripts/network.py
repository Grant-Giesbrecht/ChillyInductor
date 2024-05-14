from core import *
import threading
import socket

class ClientAgent:
	""" This will run on each client. It will connect to the server, recieve a batch order, 
	run the simulation, and return the data.
	"""
	
	def __init__(self):
		
		self.L0_list = None
		self.q_list = None
	
	def run(self):
		
		# ASK FOR INSTRUCTIONS
		
		# SPAWN SIMTHREADS
		
		# RETURN INFO
			
		#TODO: Spawn a bunch of sim threads here
		pass
	

class ServerAgent:
	""" This runs on server and connects to clients. It will send out the data to crunch, 
	then wait for the client to come back with that info, then save it to the master list"""
	
	def __init__(self, L0_list:list, q_list:list):
		
		self.L0_list = L0_list
		self.q_list = q_list
	
	def run(self):
		
		#SEND OUT POINTS
		
		# WAIT FOR REPLY
		
		# SAVE TO MASTER
		
		# CLOSE
		
		pass


class SimThread(threading.Thread):
	""" Multiple of these will be spawned by each CLientAgent."""
	
	def __init__(self, L0_list:list, q_list:list, Pgen, C_, l_phys, freq):
		
		super().__init__()
		
		self.L0_list = L0_list
		self.q_list = q_list
		
		self.Pgen = Pgen
		self.C_ = C_
		self.l_phys = l_phys
		self.freq = freq
	
	def run(self):
		global master_mutex, master_coefs, master_conditions, master_Iac, master_rmse1, master_rmse2
		global Idc_A, S21_data
		
		# Get target values from MATLAB DraftScript-25 which reads data from DS5
		Idc_A = np.array([-0.03325, -0.030875, -0.0285, -0.026125, -0.02375, -0.021375, -0.019, -0.016625, -0.01425, -0.011875, -0.0095, -0.007125, -0.00475, -0.002375, 0, 0.002375, 0.00475, 0.007125, 0.0095, 0.011875, 0.01425, 0.016625, 0.019, 0.021375, 0.02375, 0.026125, 0.0285, 0.030875, 0.03325])
		Ifund_mA = np.array([1.5131, 1.6246, 1.4932, 1.6711, 1.7109, 1.817, 1.7177, 1.6502, 1.6852, 1.7652, 1.8488, 1.9048, 1.9497, 1.9705, 1.9756, 1.9717, 1.9531, 1.9116, 1.8525, 1.7779, 1.6928, 1.6577, 1.7247, 1.8184, 1.7306, 1.6805, 1.5131, 1.6378, 1.5375])
		Ifund_A = Ifund_mA/1e3
		V2h = np.array([0.022715, 0.019268, 0.018873, 0.020213, 0.023781, 0.017135, 0.01579, 0.014872, 0.012785, 0.0098328, 0.0081696, 0.0070002, 0.0055488, 0.003095, 1.5521e-05, 0.003104, 0.0056001, 0.0070699, 0.0082481, 0.0098724, 0.012929, 0.015455, 0.016188, 0.01738, 0.024912, 0.020964, 0.019668, 0.019591, 0.023608])
		mid_idx = int(np.floor(len(Idc_A)/2))
		I2h = V2h/50
		
		logging.main(f"{Fore.GREEN}[TID={hex(threading.get_ident())}]{standard_color} Beginning sweep.")
		
		# Create array of bias values
		Ibias = Idc_A
		
		Iac_results = []
		I2h_results = []
		rmse1_results = []
		rmse2_results = []
		rmse3_results = []
		coefs = []
		conditions = []
		
		for idx, L0 in enumerate(self.L0_list):
			for idx2, q in enumerate(self.q_list):
				
				logging.main(f"{Fore.GREEN}[TID={hex(threading.get_ident())}]{Fore.LIGHTBLUE_EX}Beginning simulation of L0 = {L0*1e9} nH{standard_color}")
				
				# Prepare simulation
				lks = LKSystem(self.Pgen, self.C_, self.l_phys, self.freq, q, L0)
				lks.opt.start_guess_method = GUESS_USE_LAST
				lks.configure_loss(sparam_data=S21_data)
				
				# Solve!
				lks.solve(Ibias, show_plot_on_conv=False)
				
				# Calculate current array
				Iac = np.array([x.Iac for x in lks.solution])
				I2h_sim = np.array([x.Iac_result_spec[1] for x in lks.solution])
				
				# Calculate error basic error
				rmse1 = np.sqrt( np.mean( np.abs(Ifund_A - Iac)**2 ) )
				
				# Calculate auto-scaled error
				coef = Ifund_A[mid_idx]/Iac[mid_idx]
				Iac_scaled = Iac * coef
				rmse2 = np.sqrt( np.mean( np.abs(Ifund_A - Iac_scaled)**2 ) )
				
				# Calculate harmonic error
				rmse3 = np.sqrt( np.mean( np.abs(I2h - I2h_sim)**2 ) )
				
				# Add to output data
				Iac_results.append(Iac)
				I2h_results.append(I2h_sim)
				rmse1_results.append(rmse1)
				rmse2_results.append(rmse2)
				rmse3_results.append(rmse3)
				coefs.append(coef)
				conditions.append( (q, L0) )
		
		logging.main(f"{Fore.GREEN}[TID={hex(threading.get_ident())}]{standard_color} Finished simulation sweep.")
		
		#TODO: Change this to send to server
		with master_mutex:
			logging.main(f"{Fore.GREEN}[TID={hex(threading.get_ident())}]{standard_color} Acquired mutex. Saving to global data arrays")
			master_Iac.extend(Iac_results)
			master_I2h.extend(I2h_results)
			master_rmse1.extend(rmse1_results)
			master_rmse2.extend(rmse2_results)
			master_rmse3.extend(rmse3_results)
			master_coefs.extend(coefs)
			master_conditions.extend(conditions)
		
		logging.main(f"{Fore.GREEN}[TID={hex(threading.get_ident())}]{standard_color} Exiting")