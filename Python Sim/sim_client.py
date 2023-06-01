class SimThread(threading.Thread):
	
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
				lks = LKSystem(Pgen, C_, l_phys, freq, q, L0)
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