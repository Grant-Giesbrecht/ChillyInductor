from core import *
import time
import threading

# Get target values from MATLAB DraftScript-25 which reads data from DS5
Idc_A = np.array([-0.03325, -0.030875, -0.0285, -0.026125, -0.02375, -0.021375, -0.019, -0.016625, -0.01425, -0.011875, -0.0095, -0.007125, -0.00475, -0.002375, 0, 0.002375, 0.00475, 0.007125, 0.0095, 0.011875, 0.01425, 0.016625, 0.019, 0.021375, 0.02375, 0.026125, 0.0285, 0.030875, 0.03325])
Ifund_mA = np.array([1.5131, 1.6246, 1.4932, 1.6711, 1.7109, 1.817, 1.7177, 1.6502, 1.6852, 1.7652, 1.8488, 1.9048, 1.9497, 1.9705, 1.9756, 1.9717, 1.9531, 1.9116, 1.8525, 1.7779, 1.6928, 1.6577, 1.7247, 1.8184, 1.7306, 1.6805, 1.5131, 1.6378, 1.5375])
Ifund_A = Ifund_mA/1e3
V2h = np.array([0.022715, 0.019268, 0.018873, 0.020213, 0.023781, 0.017135, 0.01579, 0.014872, 0.012785, 0.0098328, 0.0081696, 0.0070002, 0.0055488, 0.003095, 1.5521e-05, 0.003104, 0.0056001, 0.0070699, 0.0082481, 0.0098724, 0.012929, 0.015455, 0.016188, 0.01738, 0.024912, 0.020964, 0.019668, 0.019591, 0.023608])
mid_idx = int(np.floor(len(Idc_A)/2))
I2h = V2h/50

# Pgen = -4 # dBm
Pgen = 0 # dBm
C_ = 121e-12
l_phys = 0.5
freq = 10e9
q = 0.190

Nthread = 8
Nq = 3
NL0 = 6666

# Create L0 lists
L0_sub_lists = []
# L0_list = np.linspace(.05e-9, 333.35e-9, NL0)
# L0_list = np.linspace(333.4e-9, 666.7e-9, NL0)
L0_list = np.linspace(666.75e-9, 1000e-9, NL0)

nL = NL0//Nthread

# Break into sub lists
i = None
for idx in range(Nthread):
	
	# Get slice of list
	i = idx + 1
	L0_list_temp = L0_list[int(nL*(i-1)):int((nL*i))]
	
	# Add to list of sub lists
	L0_sub_lists.append(L0_list_temp)

# Create q list
q_list = np.linspace(0.18, 0.2, Nq)

print(f"Will create {Nthread} threads. Number of simulations per thread:")
for idx in range(Nthread):
	npts = len(L0_sub_lists[idx])*len(q_list)
	print(f"\t[thread {idx}]: {npts} sims")

# Read file if no data provided
# Open File
with open("cryostat_sparams.pkl", 'rb') as fh:
	S21_data = pickle.load(fh)

# Global data
master_mutex = threading.Lock()
master_Iac = []
master_I2h = []
master_rmse1 = []
master_rmse2 = []
master_rmse3 = []
master_coefs = []
master_conditions = []

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
				I2h_sim = np.array([x.spec_Ig[1] for x in lks.solution])
				
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

# Create threads
threads = []
for idx in range(Nthread):
	
	# Create thread and add to list
	new_thread = SimThread(L0_sub_lists[idx], q_list, Pgen, C_, l_phys, freq)
	threads.append(new_thread)

t0 = time.time()

# Begin all threads
for idx in range(Nthread):
	threads[idx].start()

# Wait for threads to complete
for idx in range(Nthread):
	threads[idx].join()

tf = time.time()

# Print stats
print(f"Finished sweep in {tf-t0} seconds")

npoints = 0
for idx in range(Nthread):
	npoints += len(L0_sub_lists[idx])*len(q_list)

print(f"Number of sweep points: {npoints}")

# Find minimum error 
idx_best1 = master_rmse1.index(min(master_rmse1))
idx_best2 = master_rmse2.index(min(master_rmse2))

# Get L0 conditions
L0_best1 = master_conditions[idx_best1][1]
L0_best2 = master_conditions[idx_best2][1]

# Get Pg conditions
q_best1 = master_conditions[idx_best1][0]
q_best2 = master_conditions[idx_best2][0]

coef_best = master_coefs[idx_best2]

plt.figure(1)
plt.plot(Idc_A, Ifund_mA, color='r', label="Measurement", linestyle='dashed', marker='o')
plt.plot(Idc_A, 1e3*master_Iac[idx_best1], color='b', label=f"Sim (L0={rd(L0_best1*1e9,1)} nH) (q = {q_best1} A)", linestyle='dashed', marker='+')
plt.plot(Idc_A, 1e3*master_Iac[idx_best2]*coef_best, color=(0.35, 0, 0.6), label=f"Scaled Sim (L0={rd(L0_best2*1e9, 1)} nH) (q = {q_best2} A)", linestyle='dashed', marker='x')
plt.grid()
plt.xlabel("Bias Current (mA)")
plt.ylabel("AC Current Amplitude (mA)")
plt.title(f"Fundamental - Closest fit")
plt.legend()

plt.figure(2)
plt.plot(Idc_A, 1e3*I2h, color='r', label="Measurement", linestyle='dashed', marker='o')
plt.plot(Idc_A, 1e3*master_I2h[idx_best1], color='b', label=f"Sim (L0={rd(L0_best1*1e9,1)} nH) (q = {q_best1} A)", linestyle='dashed', marker='+')
plt.grid()
plt.xlabel("Bias Current (mA)")
plt.ylabel("AC Current Amplitude (mA)")
plt.title(f"2nd Harmonics - Closest fit")
plt.legend()

print(f"Scaling coefficient used in plot: {coef_best}")

# Save data to Pickle for further analysis
master_data = {"Iac_results":master_Iac, "I2h_results":master_I2h, "rmse1_results":master_rmse1, "rmse2_results":master_rmse2, "rmse3_results":master_rmse3, "Idc_A":Idc_A, "coefs":master_coefs, "conditions":master_conditions}
with open("harm_sim_last_data.pkl", 'wb') as f:
	pickle.dump(master_data, f)

plt.show()